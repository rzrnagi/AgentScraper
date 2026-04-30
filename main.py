"""
AgentScrape — Orchestrator (V2, AI-native)

Pipeline per category:
  1. Analyst   — LLM analyzes the page once, determines extraction strategy
  2. Navigator — collects all product URLs using that strategy
  3. Template  — LLM samples 2-3 product pages once, generates CSS selector map
  4. Extractor — applies selector map to every page (no LLM per page)
                 LLM fallback fires only if fill rate < 40% on a given page
  5. Validator — normalizes + deduplicates
  6. Storage   — SQLite + CSV/JSON export

Run:  python3 main.py [--config config.yaml] [--limit N]
Setup: python3 setup.py  (first time, to configure API key)
"""
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from playwright.async_api import async_playwright

from agents.analyst import AnalystAgent
from agents.navigator import NavigatorAgent
from agents.template import TemplateAgent
from agents.extractor import ExtractorAgent
from agents.validator import ValidatorAgent
from agents.llm_client import LLMClient
from storage.db import Database
from storage.export import export_csv, export_json

load_dotenv()


def setup_logging(cfg: dict):
    log_dir = Path(cfg.get("file", "./logs/scraper.log")).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, cfg.get("level", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cfg.get("file", "./logs/scraper.log"), encoding="utf-8"),
        ],
    )


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str) -> dict:
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def check_setup():
    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        print("\nNo LLM provider configured. Run: python3 setup.py\n")
        sys.exit(1)
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("\nANTHROPIC_API_KEY not set. Run: python3 setup.py\n")
        sys.exit(1)
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("\nOPENAI_API_KEY not set. Run: python3 setup.py\n")
        sys.exit(1)


async def run(config_path: str = "config.yaml", limit: int = None, no_llm: bool = False):
    cfg = load_config(config_path)
    setup_logging(cfg["logging"])
    logger = logging.getLogger("orchestrator")

    if not no_llm:
        check_setup()

    run_id = str(uuid.uuid4())[:8]
    logger.info("Run started — id=%s provider=%s", run_id, os.getenv("LLM_PROVIDER", "none"))

    checkpoint = load_checkpoint(cfg["checkpoint"]["file"])
    db = Database(cfg["output"]["db_file"])
    validator = ValidatorAgent()
    crawl_cfg = cfg["crawl"]
    llm_cfg = cfg.get("llm", {})

    analyst_llm = LLMClient.from_env(tier=llm_cfg.get("analyst_tier", "smart"))
    extractor_llm = LLMClient.from_env(tier=llm_cfg.get("extractor_tier", "fast"))

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=crawl_cfg.get("headless", True))
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
            viewport={"width": 1280, "height": 900},
        )

        analyst   = AnalystAgent(context=context, llm=analyst_llm)
        navigator = NavigatorAgent(context=context, llm=extractor_llm,
                                   rate_limit_delay=crawl_cfg.get("request_delay_seconds", 1.5))
        template  = TemplateAgent(context=context, llm=analyst_llm)
        extractor = ExtractorAgent(context=context, llm=extractor_llm,
                                   rate_limit_delay=crawl_cfg.get("request_delay_seconds", 1.5),
                                   page_timeout_ms=crawl_cfg.get("page_load_timeout_ms", 30000))

        for target in cfg["targets"]:
            cat_url   = target["url"]
            cat_name  = target["category"]
            cat_key   = cat_url.rstrip("/").split("/")[-1]

            logger.info("=== Category: %s ===", cat_name)

            # Step 1: Analyst determines how the site serves product data
            strategy = await analyst.analyze(cat_url)
            logger.info("Strategy: %s | %s", strategy.data_source, strategy.notes[:80])

            # Step 2: Navigator collects all products using that strategy
            products = await navigator.collect_products(cat_url, cat_name, strategy)
            if not products:
                logger.error("No products found for %s — skipping", cat_name)
                continue

            logger.info("Found %d products in '%s'", len(products), cat_name)

            # Step 3: Template agent samples a few pages and builds the selector map (LLM, once)
            product_urls = [p["product_url"] for p in products if p.get("product_url")]
            selector_map = await template.build_selector_map(product_urls, site_url=cat_url)
            extractor.set_selector_map(selector_map)
            logger.info("Template: selector map ready (%d fields mapped)", len(selector_map.fields))

            # Step 4: Extractor enriches each product page using selector map
            done_urls = set(checkpoint.get(cat_key, []))
            to_process = [p for p in products if p.get("product_url") and p["product_url"] not in done_urls]
            if limit:
                to_process = to_process[:limit]
                logger.info("Limit applied: processing %d of %d", len(to_process), len(products))

            logger.info("%d to process, %d already done (checkpoint)", len(to_process), len(done_urls))

            processed_urls = list(done_urls)
            enriched_batch = []

            for i, product in enumerate(to_process):
                url = product["product_url"]
                logger.info("[%d/%d] Extracting: %s", i + 1, len(to_process), url)
                try:
                    enriched = await extractor.extract(url, product)
                    enriched_batch.append(enriched)
                    db.log_scrape(run_id, cat_name, url, "success")
                    processed_urls.append(url)
                except Exception as e:
                    logger.error("Extraction failed for %s: %s", url, e)
                    db.log_scrape(run_id, cat_name, url, "error", str(e))
                    enriched_batch.append(product)

                if (i + 1) % 10 == 0:
                    checkpoint[cat_key] = processed_urls
                    save_checkpoint(cfg["checkpoint"]["file"], checkpoint)
                    logger.info("Checkpoint saved (%d done)", len(processed_urls))

            # Step 5: Validate + deduplicate
            valid, _ = validator.validate_batch(enriched_batch)
            logger.info("Extractor stats: %s", extractor.stats())

            # Step 6: Store
            new_count = sum(1 for p in valid if db.upsert_product(p))
            logger.info("Stored %d products (%d new) for '%s'", len(valid), new_count, cat_name)

            checkpoint[cat_key] = processed_urls
            save_checkpoint(cfg["checkpoint"]["file"], checkpoint)

        await browser.close()

    all_products = db.get_all_products()
    export_csv(all_products, cfg["output"]["csv_file"])
    export_json(all_products, cfg["output"]["json_file"])

    db.close()
    logger.info("Run complete — %d total products", len(all_products))
    return all_products


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AgentScrape — dental supply product scraper")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Max products per category (testing)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM check (not recommended)")
    args = parser.parse_args()
    asyncio.run(run(config_path=args.config, limit=args.limit, no_llm=args.no_llm))
