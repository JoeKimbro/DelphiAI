"""
Scrapy middlewares for UFC scraper

Includes:
- Proxy rotation for avoiding IP blocks
- User agent rotation
- Error logging and tracking
- Request/response handling

See documentation in:
https://docs.scrapy.org/en/latest/topics/spider-middleware.html
https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
"""

import logging
import random
import json
from datetime import datetime
from pathlib import Path

from scrapy import signals
from scrapy.exceptions import NotConfigured


logger = logging.getLogger(__name__)


class UfcScraperSpiderMiddleware:
    """Spider middleware for handling responses and exceptions."""

    def __init__(self):
        self.crawler = None

    @classmethod
    def from_crawler(cls, crawler):
        s = cls()
        s.crawler = crawler
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response):
        return None

    def process_spider_output(self, response, result):
        for i in result:
            yield i

    def process_spider_exception(self, response, exception):
        logger.error(f"Spider exception on {response.url}: {exception}")

    async def process_start(self, start):
        async for r in start:
            yield r

    def spider_opened(self, spider):
        spider.logger.info(f"Spider opened: {spider.name}")


class UfcScraperDownloaderMiddleware:
    """Downloader middleware for handling requests and responses."""

    def __init__(self):
        self.crawler = None

    @classmethod
    def from_crawler(cls, crawler):
        s = cls()
        s.crawler = crawler
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request):
        return None

    def process_response(self, request, response):
        return response

    def process_exception(self, request, exception):
        logger.error(f"Download exception on {request.url}: {exception}")

    def spider_opened(self, spider):
        spider.logger.info(f"Spider opened: {spider.name}")


class ProxyRotationMiddleware:
    """
    Middleware for rotating through a list of proxy servers.
    
    Configure in settings.py:
        PROXY_LIST = ['http://proxy1:port', 'http://proxy2:port']
        # Or load from file:
        PROXY_LIST_FILE = 'proxies.txt'
    
    Note: For production, consider using a proxy service like:
    - ScraperAPI
    - Bright Data
    - Oxylabs
    """
    
    def __init__(self, proxy_list):
        self.proxies = proxy_list
        self.proxy_index = 0
        self.failed_proxies = set()
    
    @classmethod
    def from_crawler(cls, crawler):
        proxy_list = crawler.settings.getlist('PROXY_LIST', [])
        
        # Try loading from file if no list provided
        if not proxy_list:
            proxy_file = crawler.settings.get('PROXY_LIST_FILE')
            if proxy_file:
                proxy_path = Path(proxy_file)
                if proxy_path.exists():
                    with open(proxy_path, 'r') as f:
                        proxy_list = [line.strip() for line in f if line.strip()]
        
        if not proxy_list:
            raise NotConfigured("No proxies configured. Set PROXY_LIST or PROXY_LIST_FILE in settings.")
        
        logger.info(f"Loaded {len(proxy_list)} proxies for rotation")
        return cls(proxy_list)
    
    def process_request(self, request):
        if not self.proxies:
            return None
        
        # Get next proxy (round-robin with skip for failed ones)
        attempts = 0
        while attempts < len(self.proxies):
            proxy = self.proxies[self.proxy_index % len(self.proxies)]
            self.proxy_index += 1
            
            if proxy not in self.failed_proxies:
                request.meta['proxy'] = proxy
                logger.debug(f"Using proxy: {proxy} for {request.url}")
                return None
            
            attempts += 1
        
        # All proxies failed, reset and try anyway
        logger.warning("All proxies have failed, resetting failed list")
        self.failed_proxies.clear()
        proxy = self.proxies[0]
        request.meta['proxy'] = proxy
        return None
    
    def process_exception(self, request, exception):
        proxy = request.meta.get('proxy')
        if proxy:
            logger.warning(f"Proxy {proxy} failed: {exception}")
            self.failed_proxies.add(proxy)


class RandomUserAgentMiddleware:
    """
    Middleware for rotating user agents to avoid detection.
    """
    
    USER_AGENTS = [
        # Chrome on Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        # Chrome on Mac
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        # Firefox on Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        # Firefox on Mac
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
        # Safari on Mac
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
        # Edge on Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
    ]
    
    def __init__(self, user_agents=None):
        self.user_agents = user_agents or self.USER_AGENTS
    
    @classmethod
    def from_crawler(cls, crawler):
        user_agents = crawler.settings.getlist('USER_AGENT_LIST', cls.USER_AGENTS)
        return cls(user_agents)
    
    def process_request(self, request):
        # Don't override if explicitly set
        if 'User-Agent' not in request.headers:
            ua = random.choice(self.user_agents)
            request.headers['User-Agent'] = ua


class ErrorLoggingMiddleware:
    """
    Middleware for comprehensive error logging and tracking.
    
    Logs all errors to a JSON file for later analysis.
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_log_path = self.output_dir / 'scrape_errors.json'
        self.errors = []
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retried_requests': 0,
            'errors_by_type': {},
            'errors_by_domain': {},
        }
    
    @classmethod
    def from_crawler(cls, crawler):
        output_dir = crawler.settings.get(
            'ERROR_LOG_DIR',
            Path(__file__).parent.parent.parent / 'output'
        )
        middleware = cls(output_dir)
        crawler.signals.connect(middleware.spider_closed, signal=signals.spider_closed)
        return middleware
    
    def process_request(self, request):
        self.stats['total_requests'] += 1
        return None
    
    def process_response(self, request, response):
        if response.status >= 400:
            # Ignore robots.txt 404 - this is normal (site has no robots.txt)
            if response.status == 404 and 'robots.txt' in request.url:
                self.stats['successful_requests'] += 1
                return response
            
            self._log_error(
                url=request.url,
                error_type=f"HTTP_{response.status}",
                message=f"HTTP Error: {response.status}",
                response_body=response.text[:500] if hasattr(response, 'text') else None
            )
            self.stats['failed_requests'] += 1
        else:
            self.stats['successful_requests'] += 1
        return response
    
    def process_exception(self, request, exception):
        error_type = type(exception).__name__
        self._log_error(
            url=request.url,
            error_type=error_type,
            message=str(exception)
        )
        self.stats['failed_requests'] += 1
        
        # Track by type
        self.stats['errors_by_type'][error_type] = \
            self.stats['errors_by_type'].get(error_type, 0) + 1
        
        # Track by domain
        from urllib.parse import urlparse
        domain = urlparse(request.url).netloc
        self.stats['errors_by_domain'][domain] = \
            self.stats['errors_by_domain'].get(domain, 0) + 1
    
    def _log_error(self, url, error_type, message, response_body=None):
        error_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'url': url,
            'error_type': error_type,
            'message': message,
        }
        if response_body:
            error_entry['response_preview'] = response_body
        
        self.errors.append(error_entry)
        logger.error(f"[{error_type}] {url}: {message}")
    
    def spider_closed(self, spider):
        """Save error log when spider closes."""
        output = {
            'spider': spider.name,
            'crawl_ended': datetime.utcnow().isoformat(),
            'stats': self.stats,
            'errors': self.errors
        }
        
        with open(self.error_log_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Error log saved to {self.error_log_path}")
        logger.info(f"Stats: {self.stats['successful_requests']}/{self.stats['total_requests']} successful, "
                   f"{self.stats['failed_requests']} failed")
        
        if self.stats['errors_by_type']:
            logger.info(f"Errors by type: {self.stats['errors_by_type']}")


class RetryWithBackoffMiddleware:
    """
    Enhanced retry middleware with exponential backoff.
    """
    
    def __init__(self, max_retries=3, base_delay=1):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    @classmethod
    def from_crawler(cls, crawler):
        max_retries = crawler.settings.getint('RETRY_TIMES', 3)
        base_delay = crawler.settings.getfloat('RETRY_BASE_DELAY', 1)
        return cls(max_retries, base_delay)
    
    def process_response(self, request, response):
        if response.status in [429, 503, 502, 500]:
            retries = request.meta.get('retry_count', 0)
            if retries < self.max_retries:
                import time
                delay = self.base_delay * (2 ** retries)  # Exponential backoff
                logger.warning(f"Got {response.status}, retrying in {delay}s (attempt {retries + 1})")
                time.sleep(delay)
                
                new_request = request.copy()
                new_request.meta['retry_count'] = retries + 1
                new_request.dont_filter = True
                return new_request
        return response
