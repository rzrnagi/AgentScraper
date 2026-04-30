"""
Discovery Agent

Responsibility: Load a site entry page, discover category links, filter them
via LLM to product-only categories, and return structured targets ready to
write into config.yaml.
"""
import logging
from urllib.parse import urljoin, urlparse

from playwright.async_api import BrowserContext

from agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

DISCOVERY_SYSTEM = "You are a web scraping assistant. Return only valid JSON."

DISCOVERY_PROMPT = """Given these navigation links found on an e-commerce site, select only the ones
that are product category listing pages (pages that show multiple products for sale).

Exclude: brand pages, blog/news, services, contact, about, account, login, checkout,
         cart, wishlist, and any non-product-listing pages.

Links:
{links}

Return a JSON array of selected links only:
[{{"url": "...", "category": "Clean title-case category name"}}, ...]"""


class DiscoveryAgent:
    def __init__(self, context: BrowserContext, llm: LLMClient | None = None):
        self.context = context
        self.llm = llm

    async def discover_categories(
        self,
        start_url: str,
        keywords: list[str] | None = None,
        use_llm: bool = True,
    ) -> list[dict]:
        """
        Discover product category links from a site page.
        If use_llm is True and llm is set, filters via LLM.

        Returns:
            [{"url": "https://...", "category": "Human label"}, ...]
        """
        logger.info("Discovery: loading %s", start_url)
        start_host = urlparse(start_url).netloc

        page = await self.context.new_page()
        try:
            await page.goto(start_url, wait_until="load", timeout=30000)
            await page.wait_for_timeout(2500)
            links = await page.eval_on_selector_all(
                "a[href]",
                """els => els.map(el => ({
                    href: el.href || '',
                    text: (el.textContent || '').trim()
                }))""",
            )
        finally:
            await page.close()

        # Pre-filter: same host, /catalog/ path, has text
        candidates = []
        seen = set()
        for item in links:
            href = (item.get("href") or "").strip()
            text = " ".join((item.get("text") or "").split())
            if not href or not text:
                continue
            absolute_url = urljoin(start_url, href)
            parsed = urlparse(absolute_url)
            if parsed.netloc != start_host:
                continue
            if "/catalog/" not in parsed.path:
                continue
            if parsed.path.rstrip("/") in ("/catalog", ""):
                continue
            if absolute_url in seen:
                continue
            seen.add(absolute_url)
            candidates.append({"url": absolute_url, "category": text})

        logger.info("Discovery: %d raw /catalog/ links found", len(candidates))

        if not candidates:
            return []

        # Optional keyword pre-filter before sending to LLM
        if keywords:
            kw = [k.lower() for k in keywords]
            candidates = [
                c for c in candidates
                if any(k in c["url"].lower() or k in c["category"].lower() for k in kw)
            ]
            logger.info("Discovery: %d candidates after keyword filter %s", len(candidates), keywords)

        if not candidates:
            return []

        if use_llm and self.llm:
            return await self._llm_filter(candidates)

        return candidates

    async def _llm_filter(self, candidates: list[dict]) -> list[dict]:
        lines = "\n".join(
            f'- url: {c["url"]}  text: {c["category"]}' for c in candidates[:80]
        )
        prompt = DISCOVERY_PROMPT.format(links=lines)
        try:
            result = await self.llm.complete_json(prompt, max_tokens=2000, system=DISCOVERY_SYSTEM)
            if isinstance(result, list):
                logger.info("Discovery: LLM selected %d categories", len(result))
                return result
        except Exception as e:
            logger.warning("Discovery: LLM filter failed (%s) — returning all candidates", e)
        return candidates
