"""
Page Classifier Agent

Responsibility: Given a loaded Playwright page, determine its type:
  - "listing"  → category/search results page
  - "detail"   → single product page
  - "unknown"  → neither

Strategy: fast heuristic first (URL pattern + DOM markers), LLM fallback only
if heuristic is inconclusive. This keeps LLM costs low — most pages are unambiguous.
"""
import logging
import re
from typing import Literal

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

PageType = Literal["listing", "detail", "unknown"]

DETAIL_URL_PATTERN = re.compile(r"/catalog/product/view/id/\d+/")
LISTING_URL_PATTERN = re.compile(r"/catalog/[a-z0-9-]+/?$")


class ClassifierAgent:
    def __init__(self, anthropic_client: AsyncAnthropic | None = None, use_llm: bool = True):
        self.client = anthropic_client
        self.use_llm = use_llm and anthropic_client is not None

    async def classify(self, url: str, page=None) -> PageType:
        """Classify page type. Fast heuristic first, LLM fallback if ambiguous."""
        heuristic = self._classify_by_url(url)
        if heuristic != "unknown":
            return heuristic

        if page:
            dom_result = await self._classify_by_dom(page)
            if dom_result != "unknown":
                return dom_result

        if self.use_llm and page:
            return await self._classify_by_llm(url, page)

        return "unknown"

    def _classify_by_url(self, url: str) -> PageType:
        if DETAIL_URL_PATTERN.search(url):
            return "detail"
        if LISTING_URL_PATTERN.search(url):
            return "listing"
        return "unknown"

    async def _classify_by_dom(self, page) -> PageType:
        try:
            result = await page.evaluate("""() => {
                const detailMarkers = [
                    '.product-info-main', '#product_addtocart_form',
                    '[itemprop="sku"]', '.product-info-price',
                    '.product.attribute.description'
                ];
                const listingMarkers = [
                    '.products-grid', '.products-list',
                    '[class*="product-items"]', '#algolia-hits-container'
                ];
                for (const sel of detailMarkers) {
                    if (document.querySelector(sel)) return 'detail';
                }
                for (const sel of listingMarkers) {
                    if (document.querySelector(sel)) return 'listing';
                }
                return 'unknown';
            }""")
            return result
        except Exception as e:
            logger.debug("Classifier DOM check error: %s", e)
            return "unknown"

    async def _classify_by_llm(self, url: str, page) -> PageType:
        try:
            title = await page.title()
            snippet = await page.evaluate(
                "() => document.body.innerText.slice(0, 800)"
            )
            prompt = (
                f"URL: {url}\nPage title: {title}\nPage text snippet:\n{snippet}\n\n"
                "Is this a product LISTING page (shows multiple products) or a product DETAIL page "
                "(shows a single product)? Reply with exactly one word: 'listing' or 'detail'."
            )
            response = await self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.content[0].text.strip().lower()
            if "detail" in answer:
                return "detail"
            if "listing" in answer:
                return "listing"
        except Exception as e:
            logger.warning("Classifier LLM error: %s", e)
        return "unknown"
