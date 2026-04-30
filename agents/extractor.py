"""
Extractor Agent

Responsibility: Given a product URL and a SelectorMap (generated once by
TemplateAgent), extract all schema fields using CSS selectors and JS window
vars. No LLM call per page.

LLM fallback fires ONLY when selector extraction returns fewer than half
the expected fields — meaning the page layout is unusual or broken.
"""
import json
import logging
import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup
from playwright.async_api import BrowserContext
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agents.llm_client import LLMClient
from agents.template import SelectorMap

logger = logging.getLogger(__name__)

FALLBACK_THRESHOLD = 0.4   # if <40% of fields extracted, use LLM fallback

FALLBACK_SYSTEM = "You are a product data extractor. Extract only what is on the page. Return JSON only."

FALLBACK_PROMPT = """Extract product fields from this page content.
URL: {url}
Content: {content}

Return JSON with these fields (null if not found):
product_name, brand, sku, price_raw, unit_size, availability,
description, specifications (dict), image_urls (list), alternative_products (list of {{name,url}})"""


class ExtractorAgent:
    def __init__(
        self,
        context: BrowserContext,
        llm: LLMClient,
        selector_map: SelectorMap | None = None,
        rate_limit_delay: float = 1.5,
        page_timeout_ms: int = 30000,
    ):
        self.context = context
        self.llm = llm
        self.selector_map = selector_map
        self.delay = rate_limit_delay
        self.timeout = page_timeout_ms
        self._fallback_count = 0
        self._total_count = 0

    def set_selector_map(self, selector_map: SelectorMap):
        self.selector_map = selector_map

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    async def extract(self, product_url: str, base_product: dict) -> dict:
        import asyncio

        page = await self.context.new_page()
        enriched = dict(base_product)
        enriched["scraped_at"] = datetime.now(timezone.utc).isoformat()

        try:
            await page.goto(product_url, wait_until="load", timeout=self.timeout)
            await page.wait_for_timeout(1500)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.6)")
            await page.wait_for_timeout(500)

            html = await page.content()
            soup = BeautifulSoup(html, "lxml")

            # Step 1: selector-based extraction (free, fast)
            extracted, hit_count, total_fields = self._extract_with_selectors(soup, page)

            # Step 2: JS window vars (related products etc.)
            js_fields = await self._extract_from_js(page)
            for k, v in js_fields.items():
                if v and not extracted.get(k):
                    extracted[k] = v
                    hit_count += 1

            self._total_count += 1
            fill_rate = hit_count / total_fields if total_fields else 0

            # Step 3: LLM fallback only if fill rate is too low
            if fill_rate < FALLBACK_THRESHOLD:
                logger.info(
                    "Extractor: low fill rate %.0f%% for %s — using LLM fallback",
                    fill_rate * 100, product_url
                )
                self._fallback_count += 1
                llm_fields = await self._llm_fallback(product_url, soup)
                for k, v in llm_fields.items():
                    if v and not extracted.get(k):
                        extracted[k] = v

            # Merge: only fill fields not already populated by upstream (e.g. Algolia)
            for field, value in extracted.items():
                if value is not None and not enriched.get(field):
                    enriched[field] = value

            # Normalize price
            if enriched.get("price_raw") and enriched.get("price_usd") is None:
                try:
                    enriched["price_usd"] = float(re.sub(r"[\$,]", "", str(enriched["price_raw"])).strip())
                except ValueError:
                    pass

            source = "llm_fallback" if fill_rate < FALLBACK_THRESHOLD else "selectors"
            enriched["extraction_source"] = source

        except Exception as e:
            logger.error("Extractor: failed for %s — %s", product_url, e)
            raise
        finally:
            await page.close()
            await asyncio.sleep(self.delay)

        return enriched

    def _extract_with_selectors(self, soup: BeautifulSoup, page) -> tuple[dict, int, int]:
        """Apply SelectorMap to BeautifulSoup. Returns (fields_dict, hit_count, total_fields)."""
        extracted = {}
        hit_count = 0

        if not self.selector_map:
            return extracted, 0, 1

        for field_name, fs in self.selector_map.fields.items():
            if fs.js_var:
                continue  # handled separately in _extract_from_js

            value = None

            if fs.selector:
                try:
                    if field_name == "image_urls":
                        els = soup.select(fs.selector)
                        urls = [el.get("src") or el.get("data-src", "") for el in els]
                        urls = [u for u in urls if u and "catalog/product" in u]
                        value = json.dumps(urls) if urls else None

                    elif field_name == "specifications":
                        value = self._extract_specs(soup, fs.selector)

                    elif field_name == "alternative_products":
                        # Try selector-based related products
                        els = soup.select(fs.selector)
                        alts = []
                        for el in els:
                            a = el.select_one("a[href]") or (el if el.name == "a" else None)
                            if a:
                                alts.append({"name": a.get_text(strip=True), "url": a.get("href", "")})
                        value = json.dumps(alts[:10]) if alts else None

                    else:
                        el = soup.select_one(fs.selector)
                        if el:
                            if fs.attr == "text":
                                raw = el.get_text(separator=" ", strip=True)
                            elif fs.attr == "table":
                                raw = self._extract_specs(soup, fs.selector)
                                value = raw
                                raw = None
                            else:
                                raw = el.get(fs.attr, "")

                            if raw and fs.regex:
                                m = re.search(fs.regex, raw)
                                value = m.group(0) if m else None
                            elif raw:
                                value = str(raw).strip() or None

                except Exception as e:
                    logger.debug("Extractor: selector error for %s (%s): %s", field_name, fs.selector, e)

            if value:
                extracted[field_name] = value
                hit_count += 1

        total_fields = len([f for f in self.selector_map.fields.values() if not f.js_var])
        return extracted, hit_count, max(total_fields, 1)

    def _extract_specs(self, soup: BeautifulSoup, selector: str) -> str | None:
        specs = {}
        container = soup.select_one(selector)
        if not container:
            return None
        for row in container.select("tr"):
            cells = row.select("th, td")
            if len(cells) >= 2:
                k = cells[0].get_text(strip=True)
                v = cells[1].get_text(strip=True)
                if k and v:
                    specs[k] = v
        # Also try dl/dt/dd
        for dt, dd in zip(container.select("dt"), container.select("dd")):
            k = dt.get_text(strip=True)
            v = dd.get_text(strip=True)
            if k and v:
                specs[k] = v
        return json.dumps(specs) if specs else None

    async def _extract_from_js(self, page) -> dict:
        """Extract fields that live in JavaScript window variables."""
        result = {}
        try:
            js_data = await page.evaluate("""() => {
                const out = {};
                // Related products from Algolia window vars
                const recs = window.algoliaFrequentlyBoughtProducts || window.algoliaPopularProducts;
                if (Array.isArray(recs) && recs.length > 0) {
                    out.alternative_products = recs.slice(0, 10).map(p => ({
                        name: p.name || p.family_title || '',
                        url: p.url || p.family_url || '',
                    })).filter(p => p.url);
                }
                return out;
            }""")
            if js_data.get("alternative_products"):
                result["alternative_products"] = json.dumps(js_data["alternative_products"])
        except Exception as e:
            logger.debug("Extractor: JS extraction error: %s", e)
        return result

    async def _llm_fallback(self, url: str, soup: BeautifulSoup) -> dict:
        """Last resort: send page text to LLM for field extraction."""
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        main = soup.select_one("main") or soup.body
        content = main.get_text(separator="\n", strip=True)[:4000] if main else ""

        prompt = FALLBACK_PROMPT.format(url=url, content=content)
        try:
            result = await self.llm.complete_json(prompt, max_tokens=1000, system=FALLBACK_SYSTEM)
            # Serialize complex fields
            for field in ("specifications", "image_urls", "alternative_products"):
                if isinstance(result.get(field), (dict, list)):
                    result[field] = json.dumps(result[field])
            return result
        except Exception as e:
            logger.warning("Extractor: LLM fallback failed for %s: %s", url, e)
            return {}

    def stats(self) -> dict:
        return {
            "total": self._total_count,
            "llm_fallbacks": self._fallback_count,
            "selector_hit_rate": f"{(self._total_count - self._fallback_count) / max(self._total_count, 1):.0%}",
        }
