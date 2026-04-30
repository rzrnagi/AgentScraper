"""
Template Agent

Responsibility: Given 2-3 sample product pages from a site, ask the LLM
to generate CSS selectors (and JS extraction hints) for each schema field.
This runs ONCE per site — the resulting SelectorMap is reused for every
product page with no further LLM calls.

Output: SelectorMap — a dict mapping field name → extraction config.
"""
import json
import logging
import random
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup
from playwright.async_api import BrowserContext

from agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

SCHEMA_FIELDS = [
    ("product_name",         "str",  "Full product name/title"),
    ("brand",                "str",  "Brand or manufacturer name"),
    ("sku",                  "str",  "SKU, item number, or product code(s)"),
    ("price_raw",            "str",  "Price as displayed, e.g. '$24.99'"),
    ("unit_size",            "str",  "Pack/unit size, e.g. '100/box', '50 ct'"),
    ("availability",         "str",  "Stock status, e.g. 'In stock'"),
    ("description",          "str",  "Full product description"),
    ("specifications",       "dict", "Specs/attributes as key-value pairs"),
    ("image_urls",           "list", "Product image URLs"),
    ("alternative_products", "list", "Related products as [{name, url}]"),
]

TEMPLATE_SYSTEM = """You are a CSS selector expert for web scraping.
Given rendered HTML from a product page (after JavaScript has executed), generate precise CSS selectors.
IMPORTANT: Use only selectors that match elements visible in the provided HTML. Do NOT guess or use
standard Magento/WooCommerce patterns — read the actual class names from the HTML.
Return only valid JSON."""

TEMPLATE_PROMPT = """Analyze this rendered product page and generate CSS selectors for extracting product data.

URL: {url}

VISIBLE TEXT on the page (what a user sees):
{visible_text}

RENDERED HTML (after JS execution — use these actual class names for selectors):
{html}

Rules:
- Only use selectors you can verify exist in the HTML above
- Use the visible text to identify WHICH element holds each field, then find its selector in the HTML
- For alternative_products: always use js_var (window.algoliaFrequentlyBoughtProducts or window.algoliaPopularProducts)
- If a field is genuinely absent, set selector to null

Return this JSON (include ALL fields, use null if not extractable):
{{
  "product_name":         {{"selector": "CSS selector or null", "attr": "text", "js_var": null, "regex": null, "notes": ""}},
  "brand":                {{"selector": "...", "attr": "text", "js_var": null, "regex": null, "notes": ""}},
  "sku":                  {{"selector": "...", "attr": "text", "js_var": null, "regex": null, "notes": ""}},
  "price_raw":            {{"selector": "...", "attr": "text", "js_var": null, "regex": null, "notes": ""}},
  "unit_size":            {{"selector": "...", "attr": "text", "js_var": null, "regex": null, "notes": ""}},
  "availability":         {{"selector": "...", "attr": "text", "js_var": null, "regex": null, "notes": ""}},
  "description":          {{"selector": "...", "attr": "text", "js_var": null, "regex": null, "notes": ""}},
  "specifications":       {{"selector": "table or dl selector", "attr": "table", "js_var": null, "regex": null, "notes": ""}},
  "image_urls":           {{"selector": "img selector", "attr": "src", "js_var": null, "regex": null, "notes": ""}},
  "alternative_products": {{"selector": null, "attr": null, "js_var": "algoliaFrequentlyBoughtProducts", "regex": null, "notes": "from JS window var"}}
}}"""


@dataclass
class FieldSelector:
    selector: str | None = None
    attr: str = "text"
    js_var: str | None = None
    regex: str | None = None
    notes: str = ""
    validated: bool = False
    hit_rate: float = 0.0


@dataclass
class SelectorMap:
    fields: dict[str, FieldSelector] = field(default_factory=dict)
    site_url: str = ""
    sample_count: int = 0

    def to_dict(self) -> dict:
        return {k: vars(v) for k, v in self.fields.items()}


class TemplateAgent:
    def __init__(self, context: BrowserContext, llm: LLMClient):
        self.context = context
        self.llm = llm

    async def build_selector_map(self, product_urls: list[str], site_url: str = "") -> SelectorMap:
        """
        Sample 2-3 product pages, ask LLM to generate selectors,
        validate them, and return a consolidated SelectorMap.
        """
        n_samples = min(3, len(product_urls))
        samples = random.sample(product_urls, n_samples)
        logger.info("Template: sampling %d product pages to build selector map", n_samples)

        all_proposals: list[dict] = []

        for url in samples:
            logger.info("Template: loading sample page %s", url)
            html = await self._get_rendered_html(url)
            if not html:
                continue
            proposal = await self._ask_llm_for_selectors(url, html)
            if proposal:
                all_proposals.append(proposal)

        if not all_proposals:
            logger.error("Template: no proposals generated — using empty selector map")
            return SelectorMap(site_url=site_url, sample_count=0)

        # Score selector candidates against all sample pages and keep the best
        selector_map = await self._select_best_selectors(all_proposals, samples)
        selector_map.site_url = site_url
        selector_map.sample_count = n_samples

        logger.info("Template: selector map built — %d/%d fields have selectors",
                    sum(1 for f in selector_map.fields.values() if f.selector or f.js_var),
                    len(selector_map.fields))
        return selector_map

    async def _get_rendered_html(self, url: str) -> str | None:
        page = await self.context.new_page()
        try:
            await page.goto(url, wait_until="load", timeout=30000)
            await page.wait_for_timeout(3000)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.5)")
            await page.wait_for_timeout(1000)
            html = await page.content()
            visible_text = await page.evaluate(
                "() => document.body.innerText.slice(0, 1500)"
            )
        except Exception as e:
            logger.warning("Template: failed to load %s: %s", url, e)
            return None
        finally:
            await page.close()

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "noscript", "head"]):
            tag.decompose()
        main = soup.select_one("main") or soup.body
        if not main:
            return None
        focused_html = " ".join(str(main).split())[:8000]

        return {"html": focused_html, "visible_text": visible_text}

    async def _ask_llm_for_selectors(self, url: str, page_data) -> dict | None:
        if isinstance(page_data, dict):
            html = page_data.get("html", "")
            visible_text = page_data.get("visible_text", "")
        else:
            html = page_data or ""
            visible_text = ""
        prompt = TEMPLATE_PROMPT.format(url=url, html=html, visible_text=visible_text)
        try:
            result = await self.llm.complete_json(prompt, max_tokens=1500, system=TEMPLATE_SYSTEM)
            logger.info("Template: LLM returned selectors for %s", url)
            return result
        except Exception as e:
            logger.warning("Template: LLM selector generation failed for %s: %s", url, e)
            return None

    async def _select_best_selectors(self, proposals: list[dict], sample_urls: list[str]) -> SelectorMap:
        """
        Score all proposed selectors against all sample pages and keep the
        candidate with the highest hit rate for each field.
        """
        selector_map = SelectorMap()
        all_fields = set().union(*[p.keys() for p in proposals])
        sample_soups = await self._load_sample_soups(sample_urls)

        for field_name in all_fields:
            candidates = self._collect_candidates(proposals, field_name)
            if not candidates:
                selector_map.fields[field_name] = FieldSelector()
                continue

            best = max(
                candidates,
                key=lambda fs: (
                    self._compute_hit_rate(fs, sample_soups),
                    1 if fs.js_var else 0,
                    1 if fs.selector else 0,
                ),
            )
            best.hit_rate = self._compute_hit_rate(best, sample_soups)
            best.validated = best.hit_rate > 0 or bool(best.js_var)
            if not best.validated:
                best.selector = None
                best.regex = None
            selector_map.fields[field_name] = best

        return selector_map

    async def _load_sample_soups(self, sample_urls: list[str]) -> list[BeautifulSoup]:
        soups: list[BeautifulSoup] = []
        for url in sample_urls:
            page = await self.context.new_page()
            try:
                await page.goto(url, wait_until="load", timeout=30000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    logger.debug("Template: validation networkidle timeout for %s", url)
                await page.wait_for_timeout(2500)
                html = await page.content()
                soups.append(BeautifulSoup(html, "lxml"))
            except Exception as e:
                logger.warning("Template: validation page load failed for %s: %s", url, e)
            finally:
                await page.close()
        return soups

    def _collect_candidates(self, proposals: list[dict], field_name: str) -> list[FieldSelector]:
        seen = set()
        candidates: list[FieldSelector] = []
        for proposal in proposals:
            config = proposal.get(field_name)
            if not isinstance(config, dict):
                continue
            fs = FieldSelector(
                selector=config.get("selector"),
                attr=config.get("attr", "text") or "text",
                js_var=config.get("js_var"),
                regex=config.get("regex"),
                notes=config.get("notes", ""),
            )
            key = (fs.selector, fs.attr, fs.js_var, fs.regex)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(fs)
        return candidates

    def _compute_hit_rate(self, fs: FieldSelector, soups: list[BeautifulSoup]) -> float:
        if fs.js_var:
            return 1.0
        if not fs.selector or not soups:
            return 0.0

        hits = 0
        for soup in soups:
            try:
                if self._selector_matches(fs, soup):
                    hits += 1
            except Exception as e:
                logger.debug("Template: selector scoring error for %s: %s", fs.selector, e)
        return hits / len(soups)

    def _selector_matches(self, fs: FieldSelector, soup: BeautifulSoup) -> bool:
        if not fs.selector:
            return False
        if fs.attr == "table":
            return bool(self._extract_table_like_value(soup, fs.selector))
        if fs.attr == "src":
            return any(
                (el.get("src") or el.get("data-src"))
                for el in soup.select(fs.selector)
            )
        el = soup.select_one(fs.selector)
        if not el:
            return False
        if fs.attr == "text":
            raw = el.get_text(separator=" ", strip=True)
        else:
            raw = el.get(fs.attr, "")
        if not raw:
            return False
        if fs.regex:
            return bool(re.search(fs.regex, raw))
        return True

    def _extract_table_like_value(self, soup: BeautifulSoup, selector: str) -> bool:
        container = soup.select_one(selector)
        if not container:
            return False
        if container.select("tr") or (container.select("dt") and container.select("dd")):
            return True
        return False
