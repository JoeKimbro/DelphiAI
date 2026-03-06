"""
Scrapy settings for UFC scraper project

For more info see: https://docs.scrapy.org/en/latest/topics/settings.html
"""

BOT_NAME = "ufc_scraper"

SPIDER_MODULES = ["ufc_scraper.spiders"]
NEWSPIDER_MODULE = "ufc_scraper.spiders"

# Crawl responsibly by identifying yourself
USER_AGENT = "DelphiAI Research Bot (+https://github.com/JoeKimbro/DelphiAI)"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests (default: 16)
CONCURRENT_REQUESTS = 1

# Configure a delay for requests (default: 0)
# Be polite - 2 seconds between requests
DOWNLOAD_DELAY = 2

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Override the default request headers
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en",
}

# =============================================================================
# SPIDER MIDDLEWARES
# =============================================================================
SPIDER_MIDDLEWARES = {
    "ufc_scraper.middlewares.UfcScraperSpiderMiddleware": 543,
}

# =============================================================================
# DOWNLOADER MIDDLEWARES
# =============================================================================
DOWNLOADER_MIDDLEWARES = {
    # Core middleware
    "ufc_scraper.middlewares.UfcScraperDownloaderMiddleware": 543,
    
    # Error logging (always enabled)
    "ufc_scraper.middlewares.ErrorLoggingMiddleware": 100,
    
    # Retry with exponential backoff
    "ufc_scraper.middlewares.RetryWithBackoffMiddleware": 550,
    
    # Random user agent rotation (optional - uncomment to enable)
    # "ufc_scraper.middlewares.RandomUserAgentMiddleware": 400,
    
    # Proxy rotation (optional - uncomment and configure PROXY_LIST to enable)
    # "ufc_scraper.middlewares.ProxyRotationMiddleware": 350,
}

# =============================================================================
# PROXY SETTINGS (Optional)
# =============================================================================
# Uncomment and add your proxies to enable proxy rotation
# PROXY_LIST = [
#     'http://proxy1:port',
#     'http://proxy2:port',
#     'http://user:pass@proxy3:port',
# ]
# Or load from file:
# PROXY_LIST_FILE = 'proxies.txt'

# =============================================================================
# USER AGENT LIST (Optional)
# =============================================================================
# Custom user agents for rotation (uses defaults if not set)
# USER_AGENT_LIST = [
#     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
#     # Add more...
# ]

# =============================================================================
# ITEM PIPELINES
# =============================================================================
# Lower number = higher priority (runs first)
ITEM_PIPELINES = {
    "ufc_scraper.pipelines.ValidationPipeline": 100,           # Validate required fields
    "ufc_scraper.pipelines.DataCleaningPipeline": 120,         # Clean data & calculate age
    "ufc_scraper.pipelines.DataMergePipeline": 125,            # Merge UFC.com + UFCStats data
    "ufc_scraper.pipelines.ActiveFighterFilterPipeline": 130,  # Filter active fighters (optional)
    "ufc_scraper.pipelines.DataRangeValidationPipeline": 150,  # Validate data ranges
    "ufc_scraper.pipelines.DuplicateFilterPipeline": 200,      # Filter duplicates
    "ufc_scraper.pipelines.FightStatsCalculationPipeline": 250,# Calculate KO/SUB round percentages
    "ufc_scraper.pipelines.CsvExportPipeline": 300,            # Export to CSV
    "ufc_scraper.pipelines.DatabasePipeline": 350,             # Upload to PostgreSQL (optional)
    "ufc_scraper.pipelines.ScrapeCompletionStatsPipeline": 999,# Final stats & summary (runs last)
}

# =============================================================================
# DATABASE PIPELINE (Optional)
# =============================================================================
# Set to True to upload scraped data directly to PostgreSQL
# Database connection uses settings from .env file (DB_HOST, DB_PORT, etc.)
DATABASE_PIPELINE_ENABLED = False

# =============================================================================
# ACTIVE FIGHTER FILTER (Optional)
# =============================================================================
# Set to True to only keep fighters who have fought recently
ACTIVE_FIGHTER_FILTER_ENABLED = False

# Number of years to consider a fighter "active" (default: 2 years)
ACTIVE_FIGHTER_YEARS = 2

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
# CSV output directory (relative to scrapers folder)
CSV_OUTPUT_DIR = None  # Uses default: ../output/

# Error log directory
ERROR_LOG_DIR = None  # Uses default: ../output/

# =============================================================================
# AUTOTHROTTLE SETTINGS
# =============================================================================
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
AUTOTHROTTLE_DEBUG = False

# =============================================================================
# HTTP CACHE SETTINGS
# =============================================================================
# Useful during development to avoid hammering servers
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400  # 24 hours
HTTPCACHE_DIR = "httpcache"
HTTPCACHE_IGNORE_HTTP_CODES = []
HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# =============================================================================
# RETRY SETTINGS
# =============================================================================
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_BASE_DELAY = 1  # Used by RetryWithBackoffMiddleware
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# =============================================================================
# PLAYWRIGHT SETTINGS (for JavaScript-rendered pages)
# =============================================================================
# Uncomment to enable Playwright for JavaScript rendering
# Note: Requires scrapy-playwright: pip install scrapy-playwright
#       Then run: playwright install chromium
#
# DOWNLOAD_HANDLERS = {
#     "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
#     "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
# }
# 
# PLAYWRIGHT_BROWSER_TYPE = "chromium"
# PLAYWRIGHT_LAUNCH_OPTIONS = {
#     "headless": True,
#     "timeout": 30000,  # 30 seconds
# }
# PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 30000  # 30 seconds

# =============================================================================
# TWISTED SETTINGS
# =============================================================================
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"

# =============================================================================
# LOGGING SETTINGS
# =============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_DATEFORMAT = "%Y-%m-%d %H:%M:%S"

# Log to file as well (optional)
# LOG_FILE = "../output/scrapy.log"
# LOG_FILE_APPEND = True
