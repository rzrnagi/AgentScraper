"""
AgentScrape — Orchestrator

Wires Navigator → Extractor → Validator → Storage for each target category.
Supports resumability via checkpoint file so interrupted runs pick up where they left off.
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

from agents.navigator import NavigatorAgent
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
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.get("file", "./logs/scraper.log"), encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=handlers,
    )


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str) -> dict:
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


async def run(config_path: str = "config.yaml", limit: int = None):
    cfg = load_config(config_path)
    setup_logging(cfg["logging"])
    logger = logging.getLogger("orchestrator")

    run_id = str(uuid.uuid4())[:8]
    logger.info("Run started — id=%s", run_id)

    checkpoint = load_checkpoint(cfg["checkpoint"]["file"])
    db = Database(cfg["output"]["db_file"])
    validator = ValidatorAgent()

    llm_client = None
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider:
        try:
            llm_client = LLMClient.from_env()
        except Exception as e:
            logger.warning("LLM setup invalid for provider=%s — %s", provider, e)
    else:
        logger.warning("LLM_PROVIDER not set — LLM fallback disabled")

    crawl_cfg = cfg["crawl"]

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=crawl_cfg.get("headless", True))
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/124.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )

        navigator = NavigatorAgent(
            context=context,
            rate_limit_delay=crawl_cfg.get("request_delay_seconds", 1.5),
        )
        extractor = ExtractorAgent(
            context=context,
            llm_client=llm_client,
            use_llm_fallback=cfg["llm"].get("use_for_extraction_fallback", True),
            rate_limit_delay=crawl_cfg.get("request_delay_seconds", 1.5),
            page_timeout_ms=crawl_cfg.get("page_load_timeout_ms", 30000),
        )

        for target in cfg["targets"]:
            cat_name = target["name"]
            cat_url = target["url"]
            category_label = target["category"]

            logger.info("=== Category: %s ===", category_label)

            # Step 1: Collect all products from Algolia via Navigator
            products = await navigator.collect_products(cat_url, category_label)
            if not products:
                logger.error("No products found for %s — skipping", category_label)
                continue

            logger.info("Found %d products in '%s'", len(products), category_label)

            # Step 2: For each product, enrich via detail page Extractor
            done_urls = set(checkpoint.get(cat_name, []))
            to_process = [p for p in products if p["product_url"] not in done_urls]

            if limit:
                to_process = to_process[:limit]
                logger.info("Limit applied: processing %d of %d", len(to_process), len(products))

            logger.info(
                "%d to process, %d already done (checkpoint)",
                len(to_process), len(done_urls)
            )

            processed_urls = list(done_urls)
            enriched_batch = []

            for i, product in enumerate(to_process):
                url = product["product_url"]
                logger.info("[%d/%d] Extracting: %s", i + 1, len(to_process), url)

                try:
                    enriched = await extractor.extract(url, product)
                    enriched_batch.append(enriched)
                    db.log_scrape(run_id, category_label, url, "success")
                    processed_urls.append(url)
                except Exception as e:
                    logger.error("Extraction failed for %s: %s", url, e)
                    db.log_scrape(run_id, category_label, url, "error", str(e))
                    # Still save the Algolia-only data
                    enriched_batch.append(product)

                # Checkpoint every 10 products
                if (i + 1) % 10 == 0:
                    checkpoint[cat_name] = processed_urls
                    save_checkpoint(cfg["checkpoint"]["file"], checkpoint)
                    logger.info("Checkpoint saved (%d done)", len(processed_urls))

            # Step 3: Validate + deduplicate
            valid, rejected = validator.validate_batch(enriched_batch)

            # Step 4: Write to DB
            new_count = sum(1 for p in valid if db.upsert_product(p))
            logger.info("Stored %d products (%d new) for '%s'", len(valid), new_count, category_label)

            # Save final checkpoint for this category
            checkpoint[cat_name] = processed_urls
            save_checkpoint(cfg["checkpoint"]["file"], checkpoint)

        await browser.close()

    # Export full dataset
    all_products = db.get_all_products()
    export_csv(all_products, cfg["output"]["csv_file"])
    export_json(all_products, cfg["output"]["json_file"])

    db.close()
    logger.info("Run complete — %d total products exported", len(all_products))
    return all_products


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AgentScrape — Safco Dental product scraper")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--limit", type=int, default=None, help="Max products per category (for testing)")
    args = parser.parse_args()

    asyncio.run(run(config_path=args.config, limit=args.limit))
