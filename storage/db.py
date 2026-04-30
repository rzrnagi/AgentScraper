import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name        TEXT NOT NULL,
    brand               TEXT,
    sku                 TEXT,
    category            TEXT,
    subcategory         TEXT,
    product_url         TEXT UNIQUE NOT NULL,
    price_raw           TEXT,
    price_usd           REAL,
    unit_size           TEXT,
    availability        TEXT,
    description         TEXT,
    specifications      TEXT,
    image_url           TEXT,
    image_urls          TEXT,
    alternative_products TEXT,
    meta_keywords       TEXT,
    object_id           TEXT,
    on_offer            TEXT,
    scraped_at          TEXT,
    extraction_source   TEXT
);

CREATE TABLE IF NOT EXISTS scrape_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT,
    category    TEXT,
    url         TEXT,
    status      TEXT,
    error       TEXT,
    ts          TEXT
);
"""


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        logger.info("Database initialized at %s", db_path)

    def upsert_product(self, product: dict) -> bool:
        """Insert or replace product by URL. Returns True if new, False if updated."""
        sql = """
        INSERT INTO products (
            product_name, brand, sku, category, subcategory, product_url,
            price_raw, price_usd, unit_size, availability, description,
            specifications, image_url, image_urls, alternative_products,
            meta_keywords, object_id, on_offer, scraped_at, extraction_source
        ) VALUES (
            :product_name, :brand, :sku, :category, :subcategory, :product_url,
            :price_raw, :price_usd, :unit_size, :availability, :description,
            :specifications, :image_url, :image_urls, :alternative_products,
            :meta_keywords, :object_id, :on_offer, :scraped_at, :extraction_source
        )
        ON CONFLICT(product_url) DO UPDATE SET
            product_name = excluded.product_name,
            brand = excluded.brand,
            sku = excluded.sku,
            price_raw = excluded.price_raw,
            price_usd = excluded.price_usd,
            unit_size = excluded.unit_size,
            availability = excluded.availability,
            description = excluded.description,
            specifications = excluded.specifications,
            image_url = excluded.image_url,
            image_urls = excluded.image_urls,
            alternative_products = excluded.alternative_products,
            on_offer = excluded.on_offer,
            scraped_at = excluded.scraped_at,
            extraction_source = excluded.extraction_source
        """
        row = self._conn.execute(
            "SELECT id FROM products WHERE product_url = ?", (product["product_url"],)
        ).fetchone()
        self._conn.execute(sql, product)
        self._conn.commit()
        return row is None

    def log_scrape(self, run_id: str, category: str, url: str, status: str, error: str = None):
        self._conn.execute(
            "INSERT INTO scrape_log (run_id, category, url, status, error, ts) VALUES (?,?,?,?,?,?)",
            (run_id, category, url, status, error, datetime.now(timezone.utc).isoformat())
        )
        self._conn.commit()

    def get_all_products(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM products").fetchall()
        return [dict(r) for r in rows]

    def product_url_exists(self, url: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM products WHERE product_url = ?", (url,)
        ).fetchone() is not None

    def close(self):
        self._conn.close()
