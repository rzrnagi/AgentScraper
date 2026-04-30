import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "id", "product_name", "brand", "sku", "category", "subcategory",
    "product_url", "price_raw", "price_usd", "unit_size", "availability",
    "description", "specifications", "image_url", "image_urls",
    "alternative_products", "meta_keywords", "object_id", "on_offer",
    "scraped_at", "extraction_source",
]


def export_csv(products: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(products)
    logger.info("CSV exported: %s (%d rows)", path, len(products))


def export_json(products: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    logger.info("JSON exported: %s (%d records)", path, len(products))
