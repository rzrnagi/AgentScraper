"""
Site Analyst Agent

Responsibility: Given a category URL, analyze the page once to determine
how product data is served and capture everything needed to replay it.

For API-based sites: captures the full raw request (URL + headers + body)
so the navigator can replay it verbatim — no guessing about key location,
body format, or encoding scheme.

Returns an ExtractionStrategy that the Navigator and Extractor consume.
"""
import json
import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Literal

from playwright.async_api import BrowserContext

from agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Headers that indicate a request cannot be safely replayed:
#   - signed/timestamped requests will break if body changes
#   - short-lived tokens will expire between analyst and navigator
_EPHEMERAL_HEADERS = {
    "x-signature", "x-timestamp", "x-nonce", "x-request-signature",
    "x-amz-security-token",
}
_JWT_PREFIX = "bearer eyj"  # lowercase; JWT = short-lived, unsafe to replay


def _is_safe_to_replay(headers: dict) -> bool:
    lower = {k.lower(): v for k, v in headers.items()}
    for h in _EPHEMERAL_HEADERS:
        if h in lower:
            logger.warning("Analyst: unsafe header detected (%s) — template replay disabled", h)
            return False
    auth = lower.get("authorization", "")
    if auth.lower().startswith(_JWT_PREFIX):
        logger.warning("Analyst: JWT bearer token detected — template replay disabled")
        return False
    return True


def _safe_headers(headers: dict) -> dict:
    """Strip browser-specific / session headers that would break or leak when replayed."""
    drop = {"cookie", "origin", "referer", "sec-fetch-site", "sec-fetch-mode",
            "sec-fetch-dest", "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform",
            "user-agent", "accept-encoding", "accept-language"}
    return {k: v for k, v in headers.items() if k.lower() not in drop}


@dataclass
class ExtractionStrategy:
    data_source: Literal["algolia", "graphql", "rest_api", "html"]
    # Raw request template — set when analyst intercepts a live API request.
    # Navigator replays this verbatim, only mutating the page/offset param.
    raw_request_template: dict = field(default_factory=dict)
    # Convenience fields parsed from the raw request (used for logging / fallback)
    api_endpoint: str = ""
    api_key_value: str = ""
    api_app_id: str = ""
    api_index_name: str = ""
    api_facet_filter: str = ""
    # HTML-driven fields
    product_link_selector: str = ""
    pagination_selector: str = ""
    pagination_param: str = ""
    # Shared
    product_url_pattern: str = ""
    notes: str = ""


ANALYST_SYSTEM = """You are a web scraping analyst. Your job is to examine a website's
network traffic and page structure, then produce a precise JSON extraction strategy.
Be concise and technical. Only return valid JSON."""

ANALYST_PROMPT = """Analyze this e-commerce category page and determine how product data is served.

URL: {url}

Page title: {title}

Network requests intercepted (method, URL, response summary):
{network_summary}

Page HTML snippet (first 3000 chars of main content):
{html_snippet}

Classification rules — apply these in order:
1. If ANY intercepted request URL contains "algolia.net" → data_source = "algolia"
2. If ANY intercepted request URL contains "graphql" or the body has a "query" field → data_source = "graphql"
3. If there are other JSON API calls returning product lists → data_source = "rest_api"
4. Otherwise → data_source = "html"

Return JSON in this exact format:
{{
  "data_source": "algolia" | "graphql" | "rest_api" | "html",
  "api_endpoint": "full URL of the primary API endpoint, empty if html",
  "product_link_selector": "CSS selector for product links (html only)",
  "pagination_selector": "CSS selector for next-page link (html only)",
  "pagination_param": "URL param for page number e.g. 'page' (html only)",
  "product_url_pattern": "regex matching product detail page URLs",
  "notes": "one sentence summary of what you found"
}}"""


class AnalystAgent:
    def __init__(self, context: BrowserContext, llm: LLMClient):
        self.context = context
        self.llm = llm

    async def analyze(self, category_url: str) -> ExtractionStrategy:
        logger.info("Analyst: analyzing %s", category_url)

        page = await self.context.new_page()
        network_log = []
        raw_api_request: dict = {}  # first intercepted API request, stored verbatim

        async def on_request(request):
            nonlocal raw_api_request
            if raw_api_request:
                return
            url = request.url
            # Capture the first meaningful API request (Algolia, GraphQL, REST)
            is_api = (
                ("algolia.net" in url and "queries" in url)
                or ("graphql" in url.lower() and request.method == "POST")
            )
            if not is_api:
                return

            headers = dict(request.headers)
            if not _is_safe_to_replay(headers):
                return

            body = {}
            try:
                raw = request.post_data  # raw string — available at request-fire time
                if raw:
                    body = json.loads(raw)
            except Exception:
                pass

            raw_api_request = {
                "url": url,
                "method": request.method,
                "headers": _safe_headers(headers),
                "body": body,
            }
            logger.info("Analyst: captured raw API request — %s %s",
                        request.method, url[:100])

        async def on_response(response):
            url = response.url
            ct = response.headers.get("content-type", "")
            if "json" in ct and not any(s in url for s in ["hubspot", "nr-data", "yotpo", "hubapi"]):
                try:
                    body = await response.json()
                    network_log.append(f"{response.request.method} {url[:120]}\n  -> {str(body)[:300]}")
                except Exception:
                    pass

        page.on("request", on_request)
        page.on("response", on_response)

        try:
            await page.goto(category_url, wait_until="load", timeout=35000)
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                logger.debug("Analyst: networkidle timeout for %s", category_url)
            await page.wait_for_timeout(4000)
            title = await page.title()
            main_html = await page.evaluate("""() => {
                const el = document.querySelector('main') || document.body;
                return el.innerHTML.replace(/<script[\\s\\S]*?<\\/script>/gi, '')
                                   .replace(/<style[\\s\\S]*?<\\/style>/gi, '')
                                   .replace(/\\s+/g, ' ')
                                   .slice(0, 3000);
            }""")
        finally:
            await page.close()

        network_summary = "\n".join(network_log[:20]) or "No JSON responses captured."
        prompt = ANALYST_PROMPT.format(
            url=category_url,
            title=title,
            network_summary=network_summary,
            html_snippet=main_html,
        )

        logger.info("Analyst: sending page analysis to LLM (%d chars)", len(prompt))
        try:
            result = await self.llm.complete_json(prompt, max_tokens=1024, system=ANALYST_SYSTEM)

            # Build strategy from LLM result — exclude raw_request_template (not from LLM)
            llm_fields = {k: result.get(k, "")
                          for k in ExtractionStrategy.__dataclass_fields__
                          if k != "raw_request_template"}
            strategy = ExtractionStrategy(**llm_fields)

            # Attach raw request template and parse convenience fields from it
            if raw_api_request:
                strategy.raw_request_template = raw_api_request
                strategy = _parse_api_fields(strategy, raw_api_request)
                logger.info("Analyst: raw request template attached — app=%s index=%s",
                            strategy.api_app_id, strategy.api_index_name)

            logger.info("Analyst: strategy=%s notes=%s",
                        strategy.data_source, strategy.notes[:80])
            return strategy

        except Exception as e:
            logger.error("Analyst: LLM analysis failed: %s", e)
            return ExtractionStrategy(
                data_source="html",
                product_url_pattern=r"/product/[a-z0-9-]+",
                notes=f"LLM failed ({e}), defaulting to HTML crawl",
            )


def _parse_api_fields(strategy: ExtractionStrategy, raw: dict) -> ExtractionStrategy:
    """Extract convenience fields from a raw request for logging and fallback use."""
    url = raw.get("url", "")
    headers = raw.get("headers", {})
    body = raw.get("body", {})

    strategy.api_endpoint = url.split("?")[0]

    # API key: headers → URL params
    qs = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
    strategy.api_key_value = (
        headers.get("x-algolia-api-key", "")
        or qs.get("x-algolia-api-key", "")
    )

    # App ID: headers → URL hostname → URL params
    app_id = (
        headers.get("x-algolia-application-id", "")
        or qs.get("x-algolia-application-id", "")
    )
    if not app_id:
        m = re.search(r"https://([A-Z0-9]+)-dsn\.algolia\.net", url, re.IGNORECASE)
        if m:
            app_id = m.group(1).upper()
    strategy.api_app_id = app_id

    # Index name + facet filter from Algolia body
    reqs = body.get("requests", [{}]) if isinstance(body, dict) else [{}]
    first = reqs[0] if reqs else {}
    strategy.api_index_name = first.get("indexName", "")
    params_str = first.get("params", "")
    params = dict(urllib.parse.parse_qsl(params_str))
    strategy.api_facet_filter = params.get("facetFilters", "")

    return strategy
