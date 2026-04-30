"""
Category Discoverer

Discovers product category URLs from a site homepage, filters them via LLM,
and writes the results into config.yaml targets — ready for main.py to consume.

Run:  python3 discover.py --site https://www.safcodental.com
      python3 discover.py --site https://www.safcodental.com --config config.yaml
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from playwright.async_api import async_playwright

from agents.discovery import DiscoveryAgent
from agents.llm_client import LLMClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("discover")


async def run(site_url: str, config_path: str = "config.yaml", keywords: list[str] | None = None):
    cfg_file = Path(config_path)
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)

    llm = LLMClient.from_env(tier="fast")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
            viewport={"width": 1280, "height": 900},
        )
        agent = DiscoveryAgent(context=context, llm=llm)
        categories = await agent.discover_categories(site_url, keywords=keywords, use_llm=True)
        await browser.close()

    if not categories:
        logger.error("No product categories found on %s", site_url)
        sys.exit(1)

    logger.info("Discovered %d categories — writing to %s", len(categories), config_path)
    for cat in categories:
        logger.info("  %-40s %s", cat["category"], cat["url"])

    cfg["targets"] = [{"url": c["url"], "category": c["category"]} for c in categories]

    with open(cfg_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nDone — {len(categories)} categories written to {config_path}")
    print("Run:  python3 main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover product categories and update config.yaml")
    parser.add_argument("--site", required=True, help="Site homepage or catalog index URL")
    parser.add_argument("--config", default="config.yaml", help="Config file to update (default: config.yaml)")
    parser.add_argument("--keywords", nargs="+", default=None, help="Optional keywords to narrow category discovery")
    args = parser.parse_args()
    asyncio.run(run(site_url=args.site, config_path=args.config, keywords=args.keywords))
