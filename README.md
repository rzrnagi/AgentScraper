# AgentScrape

An agent-based product scraper that targets e-commerce sites, discovers category pages, extracts product data at scale, and exports to SQLite, CSV, and JSON.

The design is deliberately lightweight: each agent owns one responsibility, and LLMs are used where they add real value — strategy analysis, selector generation, and fallback extraction — while the rest stays deterministic.

## Agent Workflow

```
discover.py  →  main.py
                  │
                  ├── AnalystAgent      inspect page + network → extraction strategy
                  ├── NavigatorAgent    collect all product URLs using that strategy
                  ├── TemplateAgent     sample 2-3 pages → reusable CSS selector map
                  ├── ExtractorAgent    apply selector map to every product page
                  ├── ValidatorAgent    normalize, validate, deduplicate
                  └── Storage           SQLite + CSV/JSON export
```

### How LLMs are used

| Step | Agent | LLM call | Purpose |
|------|-------|----------|---------|
| Discovery | DiscoveryAgent | once per site | filter nav links to product categories |
| Analysis | AnalystAgent | once per category | identify API vs HTML, capture request template |
| Templating | TemplateAgent | once per category | generate CSS selectors from rendered HTML |
| Extraction | ExtractorAgent | only on fallback | extract fields when selector fill rate < 40% |

Everything else — request replay, CSS extraction, pagination, storage — is plain code.

## Key Design Decisions

**Raw request replay instead of credential parsing** — the analyst captures the full live API request (URL + headers + body) during its page load and stores it as a template. The navigator replays it verbatim, only incrementing the page number. This works regardless of where auth credentials are placed (headers, URL params, body) and is resilient to format changes.

**Selector map built once per category** — the template agent samples 2-3 product pages, sends rendered HTML + visible text to the LLM, and generates a reusable CSS selector map. All subsequent product pages use that map with no LLM calls. Fallback only fires if fill rate drops below 40%.

**LLM sees rendered HTML** — Alpine.js and similar frameworks populate content client-side after page load. The template agent waits for JS to execute and sends both `page.content()` and `page.innerText` to the LLM so it can match visible content to actual DOM class names rather than guessing framework conventions.

## Results

Tested against Safco Dental Supply:

- **157 products** extracted across 2 categories (56 sutures, 101 gloves)
- **100% selector hit rate** — zero LLM fallbacks during extraction
- **0 products rejected** by the validator
- Algolia API intercepted automatically; full catalog fetched via paginated replay

## Project Structure

```
discover.py           category discovery entry point
main.py               scrape orchestrator
setup.py              interactive LLM provider setup
config.yaml           runtime configuration

agents/
  discovery.py        discover and filter category URLs
  analyst.py          determine extraction strategy from live page + network
  navigator.py        collect all product URLs using the strategy
  template.py         build CSS selector map from sample pages
  extractor.py        extract fields using selector map
  validator.py        normalize, validate, deduplicate
  llm_client.py       provider-agnostic LLM wrapper (Anthropic / OpenAI / Ollama)

storage/
  db.py               SQLite persistence
  export.py           CSV and JSON export
```

## Requirements

- Python 3.11+
- Playwright Chromium
- One configured LLM provider: OpenAI, Anthropic, or Ollama

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 setup.py
```

`setup.py` installs dependencies, downloads the Playwright browser, and validates your API key before writing `.env`.

## Usage

### Discover categories

```bash
python3 discover.py --site https://www.safcodental.com
python3 discover.py --site https://www.safcodental.com --keywords gloves sutures
```

Writes discovered targets into `config.yaml`. Run once before scraping.

### Run the scraper

```bash
python3 main.py
python3 main.py --limit 5        # test run, 5 products per category
python3 main.py --config custom.yaml
```

Resumes from checkpoint automatically if interrupted.

## Output

| File | Contents |
|------|----------|
| `output/products.db` | SQLite database |
| `output/products.csv` | flat CSV export |
| `output/products.json` | JSON export |
| `logs/scraper.log` | structured run log |
| `checkpoint.json` | resumable progress state |

### Schema

`product_name`, `brand`, `sku`, `category`, `subcategory`, `product_url`, `price_raw`, `price_usd`, `unit_size`, `availability`, `description`, `specifications`, `image_url`, `image_urls`, `alternative_products`, `meta_keywords`, `object_id`, `on_offer`, `scraped_at`, `extraction_source`

## Limitations

- Category discovery assumes `/catalog/`-style URL patterns; other site structures would need a more general nav heuristic.
- Selector maps can degrade on sites with heavily inconsistent product page layouts.
- Ephemeral auth tokens (JWTs, signed requests) are detected and refused for replay — those sites fall back to HTML crawl.

## Demo

## Quickstart

```bash
git clone https://github.com/rzrnagi/AgentScraper.git
cd AgentScraper
python3 -m venv .venv && source .venv/bin/activate
python3 setup.py                                                                   # install deps + configure LLM
python3 discover.py --site https://www.safcodental.com --keywords gloves sutures   # populate config.yaml
python3 main.py --limit 2                                                          # verify pipeline end-to-end
python3 main.py                                                                    # full run
```
