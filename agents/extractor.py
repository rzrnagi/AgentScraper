"""
Extractor Agent

Responsibility: Given a product detail page URL, load the page with Playwright
and extract all fields not available from Algolia:
  - description
  - specifications / attributes table
  - unit / pack size
  - alternative / related products
  - full image gallery

Strategy:
  1. CSS selector extraction (primary — fast, no LLM cost)
  2. Claude Haiku fallback for pages where selectors fail (e.g. unusual layouts)
"""
import json
import logging
import re
from datetime import datetime, timezone

from anthropic import AsyncAnthropic
from bs4 import BeautifulSoup
from playwright.async_api import BrowserContext
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class ExtractorAgent:
    def __init__(
        self,
        context: BrowserContext,
        anthropic_client: AsyncAnthropic | None = None,
        use_llm_fallback: bool = True,
        rate_limit_delay: float = 1.5,
        page_timeout_ms: int = 30000,
    ):
        self.context = context
        self.client = anthropic_client
        self.use_llm = use_llm_fallback and anthropic_client is not None
        self.delay = rate_limit_delay
        self.timeout = page_timeout_ms

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    async def extract(self, product_url: str, base_product: dict) -> dict:
        """
        Load product detail page and extract enriched fields.
        Merges with base_product data from Algolia.
        Returns updated product dict.
        """
        import asyncio

        page = await self.context.new_page()
        enriched = dict(base_product)
        enriched["scraped_at"] = datetime.now(timezone.utc).isoformat()

        try:
            await page.goto(product_url, wait_until="load", timeout=self.timeout)
            await page.wait_for_timeout(2500)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.5)")
            await page.wait_for_timeout(1000)

            html = await page.content()
            soup = BeautifulSoup(html, "lxml")

            # Pull recommended products from window vars (populated by Algolia/Adobe)
            alt_from_js = await self._extract_related_from_js(page)

            fields = self._extract_selectors(soup, page)
            if alt_from_js and not fields.get("alternative_products"):
                fields["alternative_products"] = alt_from_js

            if not fields.get("description") and self.use_llm:
                logger.info("Extractor: primary selectors incomplete for %s — using LLM fallback", product_url)
                fields = await self._extract_llm(soup, product_url, fields)

            for k, v in fields.items():
                if v is not None:
                    enriched[k] = v

            enriched["extraction_source"] = "detail_page" if fields.get("description") else "algolia+partial"

        except Exception as e:
            logger.error("Extractor: failed for %s — %s", product_url, e)
            raise
        finally:
            await page.close()
            import asyncio
            await asyncio.sleep(self.delay)

        return enriched

    async def _extract_related_from_js(self, page) -> str | None:
        """Read related/popular product data from Algolia window variables."""
        try:
            products = await page.evaluate("""() => {
                const sources = [
                    window.algoliaFrequentlyBoughtProducts,
                    window.algoliaPopularProducts,
                ];
                for (const src of sources) {
                    if (Array.isArray(src) && src.length > 0) {
                        return src.slice(0, 10).map(p => ({
                            name: p.name || p.family_title || '',
                            url: p.url || p.family_url || '',
                            brand: p.manufacturer_name || '',
                        })).filter(p => p.url);
                    }
                }
                return null;
            }""")
            if products:
                return json.dumps(products)
        except Exception as e:
            logger.debug("Related JS extraction error: %s", e)
        return None

    def _extract_selectors(self, soup: BeautifulSoup, page=None) -> dict:
        fields = {
            "description": None,
            "specifications": None,
            "unit_size": None,
            "image_urls": None,
            "alternative_products": None,
        }

        # Description — Safco uses .product-description and .pdp-tabs-group for expanded text
        desc_el = (
            soup.select_one(".product-description") or
            soup.select_one(".product-description-wrapper") or
            soup.select_one(".product.attribute.description .value") or
            soup.select_one("[itemprop='description']") or
            soup.select_one(".product-info-description")
        )
        if desc_el:
            fields["description"] = desc_el.get_text(separator=" ", strip=True)

        # Try to get fuller description from the pdp tabs section
        tab_group = soup.select_one(".pdp-tabs-group")
        if tab_group:
            full_text = tab_group.get_text(separator=" ", strip=True)
            # Extract everything after "Description" label
            if "Description" in full_text:
                desc_part = full_text.split("Description", 1)[-1].strip()
                if len(desc_part) > len(fields.get("description") or ""):
                    fields["description"] = desc_part[:2000]

        # Specifications — look for attribute tables and key/value pairs
        specs = {}
        for table in soup.select("table.data, .additional-attributes, .product-attributes"):
            for row in table.select("tr"):
                cells = row.select("th, td")
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True)
                    val = cells[1].get_text(strip=True)
                    if key and val:
                        specs[key] = val

        # Also grab list-style attributes
        for attr in soup.select(".product.attribute, [class*='product-attribute']"):
            label = attr.select_one(".label, .type, strong")
            value = attr.select_one(".value, p, span:not(.label)")
            if label and value:
                key = label.get_text(strip=True)
                val = value.get_text(strip=True)
                if key and val and key != val and key not in specs:
                    specs[key] = val

        if specs:
            fields["specifications"] = json.dumps(specs, ensure_ascii=False)

        # Unit size from specs or title patterns
        unit_pattern = re.compile(
            r"(\d+[\s\-]?(?:pk|pack|ct|count|box|bx|bag|roll|each|pc|pcs|ml|g|mg|oz|lb)s?)",
            re.IGNORECASE,
        )
        for source in [fields.get("description") or "", str(specs)]:
            m = unit_pattern.search(source)
            if m:
                fields["unit_size"] = m.group(1)
                break

        # Image gallery
        images = []
        for img in soup.select(".product.media img, .gallery-placeholder img, [data-gallery-role] img"):
            src = img.get("src") or img.get("data-src") or ""
            if src and "catalog/product" in src and src not in images:
                images.append(src)
        if images:
            fields["image_urls"] = json.dumps(images)

        # Alternative / related products — Safco renders a carousel below the fold
        # Cards have .product-info class with a product link and name
        alt_products = []
        seen_alt_urls = set()
        for card in soup.select("div.product-info"):
            links = card.select("a[href*='/product/']")
            for a in links:
                href = a.get("href", "")
                if not href or href in seen_alt_urls:
                    continue
                seen_alt_urls.add(href)
                # Name is in a sibling or child element; fall back to link text
                name_el = card.select_one("p, strong, .name")
                name = (name_el.get_text(strip=True) if name_el else a.get_text(strip=True)) or href.split("/product/")[-1]
                if name:
                    alt_products.append({"name": name, "url": href})
        if alt_products:
            fields["alternative_products"] = json.dumps(alt_products[:10])

        return fields

    async def _extract_llm(self, soup: BeautifulSoup, url: str, partial: dict) -> dict:
        """Use Claude Haiku to extract fields from page text when selectors fail."""
        if not self.client:
            return partial

        text = soup.get_text(separator="\n", strip=True)[:3000]
        prompt = f"""Extract product information from this dental supply product page.
URL: {url}

Page text:
{text}

Return a JSON object with these fields (use null if not found):
{{
  "description": "product description",
  "unit_size": "pack/unit size e.g. '100/box'",
  "specifications": {{"key": "value"}}
}}
Only return the JSON, nothing else."""

        try:
            response = await self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text_out = response.content[0].text.strip()
            # Strip markdown code fences if present
            text_out = re.sub(r"^```(?:json)?\s*", "", text_out)
            text_out = re.sub(r"\s*```$", "", text_out)
            extracted = json.loads(text_out)

            if extracted.get("description") and not partial.get("description"):
                partial["description"] = extracted["description"]
            if extracted.get("unit_size") and not partial.get("unit_size"):
                partial["unit_size"] = extracted["unit_size"]
            if extracted.get("specifications") and not partial.get("specifications"):
                specs = extracted["specifications"]
                if isinstance(specs, dict):
                    partial["specifications"] = json.dumps(specs, ensure_ascii=False)

        except Exception as e:
            logger.warning("Extractor LLM fallback error for %s: %s", url, e)

        return partial
