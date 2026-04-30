"""
Navigator Agent

Responsibility: For a given category URL, intercept the Algolia API key + query params
from the rendered page, then paginate the Algolia API to collect all product hits.
Returns a list of product dicts with basic fields populated from Algolia data.
No LLM needed here — the interception + API call is deterministic.
"""
import asyncio
import json
import logging
import re
import urllib.parse
from typing import Any

import httpx
from playwright.async_api import BrowserContext

logger = logging.getLogger(__name__)

ALGOLIA_APP_ID = "A5ULKNTM8N"
ALGOLIA_HOST = f"https://{ALGOLIA_APP_ID.lower()}-dsn.algolia.net"
ALGOLIA_INDEX = "safco_prod_default_products"
HITS_PER_PAGE = 100


class NavigatorAgent:
    def __init__(self, context: BrowserContext, rate_limit_delay: float = 1.0):
        self.context = context
        self.delay = rate_limit_delay

    async def collect_products(self, category_url: str, category_name: str) -> list[dict]:
        """
        Load the category page, intercept the Algolia API key and facet filter,
        then paginate Algolia to return all product hits for this category.
        """
        logger.info("Navigator: loading %s", category_url)
        api_key, facet_filter = await self._intercept_algolia_params(category_url)

        if not api_key:
            logger.error("Navigator: failed to intercept Algolia API key from %s", category_url)
            return []

        logger.info("Navigator: got Algolia key (expires in ~24h), filter=%s", facet_filter)

        products = await self._paginate_algolia(api_key, facet_filter, category_name)
        logger.info("Navigator: collected %d products for '%s'", len(products), category_name)
        return products

    async def _intercept_algolia_params(self, url: str) -> tuple[str, str]:
        """Open the page, wait for the Algolia /queries request, extract key + facetFilters."""
        page = await self.context.new_page()
        api_key = None
        facet_filter = None
        event = asyncio.Event()

        async def on_request(request):
            nonlocal api_key, facet_filter
            req_url = request.url
            if "algolia.net" in req_url and "/indexes/" in req_url and "queries" in req_url:
                key_match = re.search(r"x-algolia-api-key=([^&]+)", req_url)
                if key_match:
                    api_key = urllib.parse.unquote(key_match.group(1))

                body_raw = request.post_data
                if body_raw:
                    try:
                        body = json.loads(body_raw)
                        for req in body.get("requests", []):
                            params_str = req.get("params", "")
                            params = dict(urllib.parse.parse_qsl(params_str))
                            ff = params.get("facetFilters", "")
                            if ff and "categories.level1" in ff:
                                # Extract the raw filter value
                                facet_filter = ff
                                break
                    except Exception as e:
                        logger.debug("Navigator: param parse error: %s", e)

                if api_key and facet_filter:
                    event.set()

        page.on("request", on_request)

        try:
            await page.goto(url, wait_until="networkidle", timeout=35000)
            await asyncio.wait_for(event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Navigator: timeout waiting for Algolia request on %s", url)
        except Exception as e:
            logger.error("Navigator: page load error: %s", e)
        finally:
            await page.close()

        return api_key or "", facet_filter or ""

    async def _paginate_algolia(self, api_key: str, facet_filter: str, category_name: str) -> list[dict]:
        """Paginate Algolia index and return all hits normalized to product dicts."""
        all_hits = []
        page_num = 0
        nb_pages = 1

        headers = {
            "X-Algolia-Application-Id": ALGOLIA_APP_ID,
            "X-Algolia-API-Key": api_key,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            while page_num < nb_pages:
                payload = {
                    "requests": [{
                        "indexName": ALGOLIA_INDEX,
                        "params": urllib.parse.urlencode({
                            "facetFilters": facet_filter,
                            "numericFilters": '["visibility_catalog=1"]',
                            "hitsPerPage": HITS_PER_PAGE,
                            "page": page_num,
                            "query": "",
                            "attributesToRetrieve": ",".join([
                                "name", "brand", "manufacturer_name", "sku",
                                "matching_skus", "meta_keyword", "categories",
                                "categories_without_path", "url", "family_url",
                                "price", "stock_availability", "image_url",
                                "thumbnail_url", "objectID", "on_offer",
                                "clearance", "is_new", "type_id",
                                "family_title", "promo_description",
                            ]),
                        }),
                    }]
                }

                resp = await client.post(
                    f"{ALGOLIA_HOST}/1/indexes/*/queries",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

                result = data["results"][0]
                nb_pages = result.get("nbPages", 1)
                hits = result.get("hits", [])
                logger.info("Navigator: Algolia page %d/%d — %d hits", page_num + 1, nb_pages, len(hits))

                for hit in hits:
                    all_hits.append(self._normalize_hit(hit, category_name))

                page_num += 1
                if page_num < nb_pages:
                    await asyncio.sleep(self.delay)

        return all_hits

    def _flatten_to_strings(self, val) -> list[str]:
        """Recursively flatten any nested list/value to a list of strings."""
        result = []
        if isinstance(val, list):
            for item in val:
                result.extend(self._flatten_to_strings(item))
        elif val is not None:
            result.append(str(val))
        return result

    def _normalize_hit(self, hit: dict, category_name: str) -> dict:
        """Map an Algolia hit to our product schema (basic fields only)."""
        price_data = hit.get("price", {}).get("USD", {})
        price_raw = price_data.get("default_formated", "")
        price_usd = price_data.get("default")

        cats = hit.get("categories", {})
        subcategory = None
        level2 = cats.get("level2")
        if isinstance(level2, list) and level2:
            subcategory = level2[0].split("///")[-1].strip()
        elif isinstance(level2, str):
            subcategory = level2.split("///")[-1].strip()

        # matching_skus has clean catalog numbers; sku field is also a list in Algolia
        # (sku list includes the internal image code as first element + catalog numbers)
        raw_skus = hit.get("matching_skus") or []
        if not raw_skus:
            sku_field = hit.get("sku") or []
            raw_skus = sku_field if isinstance(sku_field, list) else ([sku_field] if sku_field else [])
        flat_skus = self._flatten_to_strings(raw_skus)
        sku = ", ".join(flat_skus) if flat_skus else hit.get("meta_keyword", "")

        image_urls = []
        if hit.get("image_url"):
            image_urls.append(hit["image_url"])
        if hit.get("thumbnail_url") and hit["thumbnail_url"] not in image_urls:
            image_urls.append(hit["thumbnail_url"])

        product_url = hit.get("url") or hit.get("family_url", "")

        return {
            "product_name": hit.get("name", ""),
            "brand": hit.get("manufacturer_name", ""),
            "sku": sku,
            "category": category_name,
            "subcategory": subcategory,
            "product_url": product_url,
            "price_raw": price_raw,
            "price_usd": price_usd,
            "unit_size": None,
            "availability": hit.get("stock_availability", ""),
            "description": None,
            "specifications": None,
            "image_url": hit.get("image_url", ""),
            "image_urls": json.dumps(image_urls),
            "alternative_products": None,
            "meta_keywords": hit.get("meta_keyword", ""),
            "object_id": hit.get("objectID", ""),
            "on_offer": hit.get("on_offer", ""),
            "scraped_at": None,
            "extraction_source": "algolia",
        }
