"""
Validator / Deduplicator Agent

Responsibility:
  - Validate required fields
  - Normalize price strings, strip HTML entities, clean whitespace
  - Deduplicate within a batch by product_url
  - Log validation failures with reason
"""
import html
import json
import logging
import re
from typing import NamedTuple

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["product_name", "product_url"]
PRICE_RE = re.compile(r"[\$,]")


class ValidationResult(NamedTuple):
    valid: bool
    reason: str


class ValidatorAgent:
    def __init__(self):
        self._seen_urls: set[str] = set()

    def reset(self):
        """Clear dedup state between runs if needed."""
        self._seen_urls.clear()

    def validate_and_clean(self, product: dict) -> tuple[dict, ValidationResult]:
        """
        Returns (cleaned_product, result).
        If result.valid is False, caller should skip this product.
        """
        p = dict(product)

        # Deduplicate by URL
        url = p.get("product_url", "").strip()
        if not url:
            return p, ValidationResult(False, "missing product_url")
        if url in self._seen_urls:
            return p, ValidationResult(False, f"duplicate url: {url}")
        self._seen_urls.add(url)

        # Required field check
        for field in REQUIRED_FIELDS:
            if not p.get(field):
                return p, ValidationResult(False, f"missing required field: {field}")

        # Clean string fields
        for field in ["product_name", "brand", "description", "unit_size", "availability"]:
            if p.get(field):
                p[field] = html.unescape(str(p[field])).strip()
                p[field] = re.sub(r"\s+", " ", p[field])

        # Normalize price
        if p.get("price_raw"):
            try:
                cleaned = PRICE_RE.sub("", str(p["price_raw"])).strip()
                p["price_usd"] = float(cleaned)
            except ValueError:
                pass

        # Ensure price_usd is float or None
        if p.get("price_usd") is not None:
            try:
                p["price_usd"] = float(p["price_usd"])
            except (TypeError, ValueError):
                p["price_usd"] = None

        # Validate JSON fields are valid JSON strings
        for json_field in ["image_urls", "alternative_products", "specifications"]:
            val = p.get(json_field)
            if val and isinstance(val, str):
                try:
                    json.loads(val)
                except json.JSONDecodeError:
                    logger.debug("Validator: invalid JSON in %s for %s — clearing", json_field, url)
                    p[json_field] = None

        return p, ValidationResult(True, "ok")

    def validate_batch(self, products: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Process a list of products.
        Returns (valid_products, rejected_products).
        """
        valid, rejected = [], []
        for p in products:
            cleaned, result = self.validate_and_clean(p)
            if result.valid:
                valid.append(cleaned)
            else:
                logger.warning(
                    "Validator: rejected '%s' — %s",
                    p.get("product_name", "?"),
                    result.reason,
                )
                rejected.append({**cleaned, "_rejection_reason": result.reason})
        logger.info("Validator: %d valid, %d rejected", len(valid), len(rejected))
        return valid, rejected
