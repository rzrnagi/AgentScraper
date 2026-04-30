"""
Navigator Agent

Responsibility: Given an ExtractionStrategy from the Analyst, collect all
product URLs (and basic fields where available) for a category.

For API-based sites (Algolia, REST, GraphQL): replays the raw request
template captured by the analyst, only mutating the page/offset parameter.
This works regardless of where auth credentials are located (headers, URL
params, body) and regardless of encoding format.

For HTML sites: crawls listing pages following pagination links.
"""
import asyncio
import json
import logging
import re
import urllib.parse
from typing import Any

import httpx
from playwright.async_api import BrowserContext

from agents.analyst import ExtractionStrategy
from agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

HITS_PER_PAGE = 100


class NavigatorAgent:
    def __init__(self, context: BrowserContext, llm: LLMClient, rate_limit_delay: float = 1.5):
        self.context = context
        self.llm = llm
        self.delay = rate_limit_delay

    async def collect_products(
        self,
        category_url: str,
        category_name: str,
        strategy: ExtractionStrategy,
    ) -> list[dict]:
        logger.info("Navigator: strategy=%s", strategy.data_source)

        if strategy.data_source == "algolia":
            if strategy.raw_request_template:
                return await self._paginate_from_template(
                    strategy.raw_request_template, category_name, api_type="algolia"
                )
            logger.warning("Navigator: no raw template — falling back to HTML")
            return await self._crawl_html(category_url, category_name, strategy)

        elif strategy.data_source in ("rest_api", "graphql"):
            if strategy.raw_request_template:
                return await self._paginate_from_template(
                    strategy.raw_request_template, category_name, api_type="generic"
                )
            return await self._navigate_api_via_llm(category_url, category_name, strategy)

        else:
            return await self._crawl_html(category_url, category_name, strategy)

    # ------------------------------------------------------------------ Template replay

    async def _paginate_from_template(
        self, template: dict, category_name: str, api_type: str
    ) -> list[dict]:
        """
        Replay the captured request for every page, incrementing the page counter.
        Works for Algolia and generic REST/GraphQL — no assumptions about auth location.
        """
        url = template["url"]
        method = template.get("method", "POST").upper()
        headers = dict(template.get("headers", {}))
        headers.setdefault("Content-Type", "application/json")
        original_body = template.get("body", {})

        all_hits = []
        page_num = 0
        nb_pages = 1

        async with httpx.AsyncClient(timeout=20.0) as client:
            while page_num < nb_pages:
                body = _set_page(original_body, page_num, HITS_PER_PAGE)
                try:
                    if method == "POST":
                        resp = await client.post(url, headers=headers, json=body)
                    else:
                        resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (401, 403):
                        logger.error(
                            "Navigator: %s on page %d — token may have expired. "
                            "Re-running the analyst would refresh it.",
                            e.response.status_code, page_num
                        )
                    else:
                        logger.error("Navigator: HTTP %s on page %d", e.response.status_code, page_num)
                    break

                data = resp.json()

                if api_type == "algolia":
                    results = data.get("results", [])
                    if not results:
                        logger.error("Navigator: Algolia returned empty results. body=%s",
                                     json.dumps(body)[:400])
                        break
                    result = results[0]
                    nb_pages = result.get("nbPages", 1)
                    hits = result.get("hits", [])
                    logger.info("Navigator: Algolia page %d/%d — %d hits",
                                page_num + 1, nb_pages, len(hits))
                    for hit in hits:
                        all_hits.append(self._normalize_algolia_hit(hit, category_name))
                else:
                    # Generic: collect any string URLs or objects with a url key
                    items = _extract_list(data)
                    if not items:
                        break
                    for item in items:
                        product_url = item if isinstance(item, str) else item.get("url", "")
                        if product_url:
                            all_hits.append({
                                "product_url": product_url,
                                "category": category_name,
                                "extraction_source": "api",
                            })

                page_num += 1
                if page_num < nb_pages:
                    await asyncio.sleep(self.delay)

        logger.info("Navigator: collected %d products via template replay", len(all_hits))
        return all_hits

    def _normalize_algolia_hit(self, hit: dict, category_name: str) -> dict:
        price_data = hit.get("price", {}).get("USD", {})
        cats = hit.get("categories", {})
        level2 = cats.get("level2")
        subcategory = None
        if isinstance(level2, list) and level2:
            subcategory = level2[0].split("///")[-1].strip()
        elif isinstance(level2, str):
            subcategory = level2.split("///")[-1].strip()

        raw_skus = hit.get("matching_skus") or []
        if not raw_skus:
            sku_field = hit.get("sku") or []
            raw_skus = sku_field if isinstance(sku_field, list) else ([sku_field] if sku_field else [])
        flat_skus = self._flatten_to_strings(raw_skus)
        sku = ", ".join(flat_skus) if flat_skus else hit.get("meta_keyword", "")

        images = []
        for key in ("image_url", "thumbnail_url"):
            val = hit.get(key, "")
            if val and val not in images:
                images.append(val)

        return {
            "product_name": hit.get("name", ""),
            "brand": hit.get("manufacturer_name", ""),
            "sku": sku,
            "category": category_name,
            "subcategory": subcategory,
            "product_url": hit.get("url") or hit.get("family_url", ""),
            "price_raw": price_data.get("default_formated", ""),
            "price_usd": price_data.get("default"),
            "unit_size": None,
            "availability": hit.get("stock_availability", ""),
            "description": None,
            "specifications": None,
            "image_url": hit.get("image_url", ""),
            "image_urls": json.dumps(images),
            "alternative_products": None,
            "meta_keywords": hit.get("meta_keyword", ""),
            "object_id": hit.get("objectID", ""),
            "on_offer": hit.get("on_offer", ""),
            "scraped_at": None,
            "extraction_source": "algolia",
        }

    def _flatten_to_strings(self, val: Any) -> list[str]:
        result = []
        if isinstance(val, list):
            for item in val:
                result.extend(self._flatten_to_strings(item))
        elif val is not None:
            result.append(str(val))
        return result

    # ------------------------------------------------------------------ Generic API (LLM fallback)

    async def _navigate_api_via_llm(
        self, category_url: str, category_name: str, strategy: ExtractionStrategy
    ) -> list[dict]:
        logger.info("Navigator: using LLM to plan %s API navigation", strategy.data_source)
        prompt = f"""E-commerce scraping task. Category URL: {category_url}
API endpoint: {strategy.api_endpoint}
Notes: {strategy.notes}

Return JSON describing how to paginate this API to get all product URLs:
{{
  "request_url": "...",
  "request_method": "GET or POST",
  "headers": {{}},
  "body": null,
  "product_url_path": "dot.notation path to URL list in response",
  "page_param": "URL param for page number or null"
}}"""
        try:
            plan = await self.llm.complete_json(prompt, max_tokens=512)
            return await self._execute_api_plan(plan, category_name)
        except Exception as e:
            logger.error("Navigator: API navigation failed: %s — falling back to HTML", e)
            return await self._crawl_html(category_url, category_name, strategy)

    async def _execute_api_plan(self, plan: dict, category_name: str) -> list[dict]:
        all_products = []
        url = plan.get("request_url", "")
        method = plan.get("request_method", "GET").upper()
        headers = plan.get("headers", {})
        body = plan.get("body")
        url_path = plan.get("product_url_path", "")
        page_param = plan.get("page_param")
        page = 0

        async with httpx.AsyncClient(timeout=20.0) as client:
            while True:
                req_url = f"{url}&{page_param}={page}" if page_param else url
                resp = await (client.post(req_url, headers=headers, json=body)
                              if method == "POST" else client.get(req_url, headers=headers))
                resp.raise_for_status()
                data = resp.json()
                items = data
                for key in url_path.split("."):
                    items = items.get(key, []) if isinstance(items, dict) else []
                if not items:
                    break
                for item in items:
                    product_url = item if isinstance(item, str) else item.get("url", "")
                    if product_url:
                        all_products.append({
                            "product_url": product_url,
                            "category": category_name,
                            "extraction_source": "api",
                        })
                if not page_param:
                    break
                page += 1
                await asyncio.sleep(self.delay)

        return all_products

    # ------------------------------------------------------------------ HTML crawl

    async def _crawl_html(
        self, category_url: str, category_name: str, strategy: ExtractionStrategy
    ) -> list[dict]:
        logger.info("Navigator: crawling HTML listing pages")
        page = await self.context.new_page()
        all_products = []
        url = category_url
        pattern = re.compile(strategy.product_url_pattern) if strategy.product_url_pattern else None
        seen = set()

        while url:
            await page.goto(url, wait_until="load", timeout=35000)
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                logger.debug("Navigator: networkidle timeout for %s", url)
            await page.wait_for_timeout(2500)
            links = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")

            for link in links:
                if link in seen:
                    continue
                if pattern:
                    is_product = bool(pattern.search(link))
                else:
                    is_product = "/product/" in link and "/catalog/" not in link
                if not is_product:
                    continue
                seen.add(link)
                all_products.append({
                    "product_url": link,
                    "category": category_name,
                    "extraction_source": "html_crawl",
                })

            next_url = None
            if strategy.pagination_selector:
                el = await page.query_selector(strategy.pagination_selector)
                if el:
                    next_url = await el.get_attribute("href")

            url = next_url
            if url:
                await asyncio.sleep(self.delay)

        await page.close()
        logger.info("Navigator: collected %d product URLs via HTML", len(all_products))
        return all_products


# ------------------------------------------------------------------ Helpers

def _set_page(body: dict, page_num: int, hits_per_page: int) -> dict:
    """
    Return a copy of the Algolia request body with page/hitsPerPage updated.
    Handles URL-encoded params string inside body.requests[].params.
    Falls back gracefully if body structure is unexpected.
    """
    if not isinstance(body, dict):
        return body
    body = json.loads(json.dumps(body))  # deep copy
    reqs = body.get("requests")
    if not isinstance(reqs, list) or not reqs:
        return body
    first = reqs[0]
    params_str = first.get("params", "")
    if params_str:
        params = dict(urllib.parse.parse_qsl(params_str, keep_blank_values=True))
        params["page"] = str(page_num)
        params["hitsPerPage"] = str(hits_per_page)
        first["params"] = urllib.parse.urlencode(params)
    else:
        first["page"] = page_num
        first["hitsPerPage"] = hits_per_page
    return body


def _extract_list(data: Any) -> list:
    """Walk a JSON response to find the first non-empty list (generic API fallback)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            result = _extract_list(v)
            if result:
                return result
    return []
