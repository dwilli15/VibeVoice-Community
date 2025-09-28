"""
Web Scraping Module for VibeVoice Community
Extracts content from URLs and converts to audiobook format
"""

import os
import re
import sys
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import logging

# Web scraping imports
try:
    import requests
    from bs4 import BeautifulSoup
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

# Add Firecrawl support if available
try:
    # Check if we have the Firecrawl MCP tools available
    # This would be available if mcp_firecrawl is installed
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False

@dataclass
class WebContent:
    """Represents extracted web content"""
    url: str
    title: str
    content: str
    author: str = ""
    publish_date: str = ""
    word_count: int = 0
    language: str = "en"
    content_type: str = "article"  # article, blog, documentation, forum
    
@dataclass
class ScrapingConfig:
    """Configuration for web scraping"""
    max_depth: int = 3
    max_pages: int = 50
    include_images_alt: bool = True
    include_links: bool = False
    respect_robots_txt: bool = True
    delay_between_requests: float = 1.0
    user_agent: str = "VibeVoice-Community/1.0 (Educational TTS Converter)"
    content_filters: List[str] = None  # CSS selectors to exclude
    
class WebScraper:
    """Advanced web scraping for content extraction"""
    
    def __init__(self, config: ScrapingConfig = None):
        self.config = config or ScrapingConfig()
        self.session = None
        self.logger = self._setup_logging()
        
        if not WEB_AVAILABLE:
            raise ImportError("Web scraping requires: pip install requests beautifulsoup4")
        
        # Initialize session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for scraping process"""
        logger = logging.getLogger('WebScraper')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def scrape_single_url(self, url: str) -> Optional[WebContent]:
        """Scrape content from a single URL"""
        try:
            self.logger.info(f"🌐 Scraping: {url}")
            
            # Check robots.txt if required
            if self.config.respect_robots_txt and not self._check_robots_txt(url):
                self.logger.warning(f"❌ Robots.txt disallows scraping: {url}")
                return None
            
            # Fetch the page
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract structured content
            content = self._extract_main_content(soup, url)
            
            if content:
                self.logger.info(f"✅ Extracted {content.word_count} words from {url}")
                return content
            else:
                self.logger.warning(f"⚠️ No content extracted from {url}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to scrape {url}: {e}")
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup, url: str) -> Optional[WebContent]:
        """Extract main content from HTML soup"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Apply custom content filters
        if self.config.content_filters:
            for selector in self.config.content_filters:
                for element in soup.select(selector):
                    element.decompose()
        
        # Try to find the main content area
        main_content = self._find_main_content_area(soup)
        
        if not main_content:
            return None
        
        # Extract metadata
        title = self._extract_title(soup)
        author = self._extract_author(soup)
        publish_date = self._extract_publish_date(soup)
        content_type = self._detect_content_type(soup, url)
        
        # Clean and format text
        text_content = self._clean_text_content(main_content)
        
        if len(text_content.strip()) < 100:  # Too short to be useful
            return None
        
        word_count = len(text_content.split())
        
        return WebContent(
            url=url,
            title=title,
            content=text_content,
            author=author,
            publish_date=publish_date,
            word_count=word_count,
            content_type=content_type
        )
    
    def _find_main_content_area(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Find the main content area using various strategies"""
        # Strategy 1: Look for common content selectors
        content_selectors = [
            'article',
            '[role="main"]',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.content',
            '#content',
            '.post-body',
            '.story-body',
            'main'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                # Return the largest element
                return max(elements, key=lambda x: len(x.get_text()))
        
        # Strategy 2: Find the element with the most text
        potential_containers = soup.find_all(['div', 'section', 'article'])
        if potential_containers:
            # Filter out elements that are likely navigation/ads
            filtered = []
            for container in potential_containers:
                text_length = len(container.get_text().strip())
                link_count = len(container.find_all('a'))
                text_to_link_ratio = text_length / max(link_count, 1)
                
                # Good content has a high text-to-link ratio
                if text_length > 200 and text_to_link_ratio > 50:
                    filtered.append(container)
            
            if filtered:
                return max(filtered, key=lambda x: len(x.get_text()))
        
        # Strategy 3: Use the body as fallback
        return soup.find('body')
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        # Try different title sources
        title_selectors = [
            'h1',
            '.post-title',
            '.entry-title',
            '.article-title',
            'title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text().strip()
                if title and len(title) > 3:
                    return title
        
        return "Untitled"
    
    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extract author information"""
        author_selectors = [
            '.author',
            '.byline',
            '.post-author',
            '[rel="author"]',
            '.writer',
            '.contributor'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                author = element.get_text().strip()
                if author:
                    return author
        
        # Try meta tags
        meta_author = soup.find('meta', {'name': 'author'})
        if meta_author and meta_author.get('content'):
            return meta_author['content']
        
        return "Unknown"
    
    def _extract_publish_date(self, soup: BeautifulSoup) -> str:
        """Extract publication date"""
        date_selectors = [
            'time',
            '.date',
            '.publish-date',
            '.post-date',
            '.created-date'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                # Try datetime attribute first
                if element.get('datetime'):
                    return element['datetime']
                # Then try text content
                date_text = element.get_text().strip()
                if date_text:
                    return date_text
        
        # Try meta tags
        meta_date = soup.find('meta', {'property': 'article:published_time'})
        if meta_date and meta_date.get('content'):
            return meta_date['content']
        
        return ""
    
    def _detect_content_type(self, soup: BeautifulSoup, url: str) -> str:
        """Detect the type of content"""
        url_lower = url.lower()
        
        # Check URL patterns
        if any(pattern in url_lower for pattern in ['blog', 'post', 'article']):
            return "blog"
        elif any(pattern in url_lower for pattern in ['docs', 'documentation', 'wiki']):
            return "documentation"
        elif any(pattern in url_lower for pattern in ['forum', 'discussion', 'reddit']):
            return "forum"
        elif any(pattern in url_lower for pattern in ['news', 'press', 'release']):
            return "news"
        
        # Check meta tags
        meta_type = soup.find('meta', {'property': 'og:type'})
        if meta_type and meta_type.get('content'):
            return meta_type['content']
        
        return "article"
    
    def _clean_text_content(self, content_element: BeautifulSoup) -> str:
        """Clean and format text content for TTS"""
        # Extract text while preserving structure
        text_parts = []
        
        for element in content_element.descendants:
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Add heading markers
                text = element.get_text().strip()
                if text:
                    text_parts.append(f"\n\n{text}\n")
            elif element.name == 'p':
                # Add paragraph breaks
                text = element.get_text().strip()
                if text:
                    text_parts.append(f"{text}\n\n")
            elif element.name == 'br':
                text_parts.append("\n")
            elif element.name in ['li']:
                # List items
                text = element.get_text().strip()
                if text:
                    text_parts.append(f"• {text}\n")
            elif element.name == 'img' and self.config.include_images_alt:
                # Include alt text for images
                alt_text = element.get('alt', '').strip()
                if alt_text:
                    text_parts.append(f"[Image: {alt_text}] ")
            elif element.name == 'a' and self.config.include_links:
                # Include link text and URL
                link_text = element.get_text().strip()
                href = element.get('href', '')
                if link_text and href:
                    text_parts.append(f"{link_text} (link: {href}) ")
            elif hasattr(element, 'string') and element.string:
                # Regular text nodes
                text = element.string.strip()
                if text:
                    text_parts.append(text + " ")
        
        # Join and clean up
        full_text = ''.join(text_parts)
        
        # Clean up whitespace and formatting
        full_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', full_text)  # Multiple newlines
        full_text = re.sub(r' +', ' ', full_text)  # Multiple spaces
        full_text = re.sub(r'\n ', '\n', full_text)  # Space after newline
        
        return full_text.strip()
    
    def _check_robots_txt(self, url: str) -> bool:
        """Check if robots.txt allows scraping this URL"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            response = self.session.get(robots_url, timeout=5)
            if response.status_code != 200:
                return True  # No robots.txt, assume allowed
            
            # Simple robots.txt parsing (basic implementation)
            lines = response.text.split('\n')
            disallowed_paths = []
            current_user_agent = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('User-agent:'):
                    current_user_agent = line.split(':', 1)[1].strip()
                elif line.startswith('Disallow:') and current_user_agent in ['*', 'VibeVoice-Community']:
                    path = line.split(':', 1)[1].strip()
                    if path:
                        disallowed_paths.append(path)
            
            # Check if current URL path is disallowed
            url_path = parsed_url.path
            for disallowed in disallowed_paths:
                if url_path.startswith(disallowed):
                    return False
            
            return True
            
        except Exception:
            return True  # If we can't check, assume allowed
    
    def scrape_multiple_urls(self, urls: List[str]) -> List[WebContent]:
        """Scrape content from multiple URLs"""
        results = []
        
        for i, url in enumerate(urls):
            self.logger.info(f"📄 Processing {i+1}/{len(urls)}: {url}")
            
            content = self.scrape_single_url(url)
            if content:
                results.append(content)
            
            # Respect rate limiting
            if i < len(urls) - 1:  # Don't delay after the last URL
                import time
                time.sleep(self.config.delay_between_requests)
        
        return results
    
    def scrape_website_sitemap(self, base_url: str) -> List[WebContent]:
        """Scrape content from a website using its sitemap"""
        try:
            # Try to find sitemap.xml
            sitemap_urls = [
                urljoin(base_url, '/sitemap.xml'),
                urljoin(base_url, '/sitemap_index.xml'),
                urljoin(base_url, '/sitemaps.xml')
            ]
            
            urls_to_scrape = []
            
            for sitemap_url in sitemap_urls:
                try:
                    response = self.session.get(sitemap_url, timeout=10)
                    if response.status_code == 200:
                        # Parse sitemap XML
                        soup = BeautifulSoup(response.content, 'xml')
                        urls = soup.find_all('url')
                        
                        for url_elem in urls:
                            loc = url_elem.find('loc')
                            if loc:
                                urls_to_scrape.append(loc.text)
                        
                        break  # Found a working sitemap
                except:
                    continue
            
            if not urls_to_scrape:
                self.logger.warning(f"No sitemap found for {base_url}")
                return []
            
            # Limit the number of URLs to respect max_pages
            urls_to_scrape = urls_to_scrape[:self.config.max_pages]
            
            return self.scrape_multiple_urls(urls_to_scrape)
            
        except Exception as e:
            self.logger.error(f"Failed to scrape sitemap for {base_url}: {e}")
            return []

class WebToAudiobookConverter:
    """Convert web content to audiobook format"""
    
    def __init__(self, scraper: WebScraper = None):
        self.scraper = scraper or WebScraper()
        self.logger = logging.getLogger('WebToAudiobook')
    
    def convert_url_to_audiobook(self, url: str, output_dir: str, 
                                config) -> Dict:
        """Convert a single URL to audiobook"""
        from ebook_converter import EbookToAudiobookConverter, ConversionConfig as EbookConfig
        
        # Scrape content
        web_content = self.scraper.scrape_single_url(url)
        if not web_content:
            return {"error": f"Failed to scrape content from {url}"}
        
        # Create temporary text file
        temp_file = Path(output_dir) / f"web_content_{urlparse(url).netloc}.txt"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(f"Title: {web_content.title}\n")
            f.write(f"Author: {web_content.author}\n")
            f.write(f"Source: {web_content.url}\n")
            f.write(f"Date: {web_content.publish_date}\n\n")
            f.write(web_content.content)
        
        # Convert using existing ebook converter
        ebook_config = EbookConfig(
            input_file=str(temp_file),
            output_dir=output_dir,
            voice_name=config.voice_name if hasattr(config, 'voice_name') else "bf_isabella",
            speed=config.speed if hasattr(config, 'speed') else 1.3,
            format=config.format if hasattr(config, 'format') else "wav",
            engine=config.engine if hasattr(config, 'engine') else "auto",
            title=web_content.title,
            author=web_content.author
        )
        
        converter = EbookToAudiobookConverter()
        results = converter.convert_to_audiobook(ebook_config)
        
        # Cleanup temp file
        temp_file.unlink(missing_ok=True)
        
        return results
    
    def convert_multiple_urls(self, urls: List[str], output_dir: str,
                            config) -> Dict:
        """Convert multiple URLs to a combined audiobook"""
        # Scrape all URLs
        web_contents = self.scraper.scrape_multiple_urls(urls)
        
        if not web_contents:
            return {"error": "No content could be scraped from provided URLs"}
        
        # Combine into single text file with chapters
        temp_file = Path(output_dir) / "web_compilation.txt"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(f"Web Content Compilation\n")
            f.write(f"Generated by VibeVoice Community\n\n")
            
            for i, content in enumerate(web_contents, 1):
                f.write(f"Chapter {i}: {content.title}\n")
                f.write(f"Source: {content.url}\n")
                f.write(f"Author: {content.author}\n\n")
                f.write(content.content)
                f.write("\n\n" + "="*50 + "\n\n")
        
        # Convert using ebook converter
        from ebook_converter import EbookToAudiobookConverter, ConversionConfig as EbookConfig
        
        ebook_config = EbookConfig(
            input_file=str(temp_file),
            output_dir=output_dir,
            voice_name=config.voice_name if hasattr(config, 'voice_name') else "bf_isabella",
            speed=config.speed if hasattr(config, 'speed') else 1.3,
            format=config.format if hasattr(config, 'format') else "wav",
            engine=config.engine if hasattr(config, 'engine') else "auto",
            title="Web Content Compilation",
            author="Various"
        )
        
        converter = EbookToAudiobookConverter()
        results = converter.convert_to_audiobook(ebook_config)
        
        # Cleanup temp file
        temp_file.unlink(missing_ok=True)
        
        return results

def main():
    """CLI interface for web scraping and conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert web content to audiobooks")
    parser.add_argument("url", help="URL to scrape and convert")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-v", "--voice", default="bf_isabella", help="Voice name")
    parser.add_argument("-s", "--speed", type=float, default=1.3, help="Speech speed")
    parser.add_argument("-f", "--format", choices=["wav", "mp3", "m4b"], default="wav")
    parser.add_argument("-e", "--engine", choices=["vibevoice", "coqui", "auto"], default="auto")
    parser.add_argument("--max-pages", type=int, default=10, help="Maximum pages to scrape")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    
    args = parser.parse_args()
    
    # Setup scraping config
    scraping_config = ScrapingConfig(
        max_pages=args.max_pages,
        delay_between_requests=args.delay
    )
    
    # Create converter
    scraper = WebScraper(scraping_config)
    converter = WebToAudiobookConverter(scraper)
    
    # Simple config object for conversion
    class SimpleConfig:
        def __init__(self):
            self.voice_name = args.voice
            self.speed = args.speed
            self.format = args.format
            self.engine = args.engine
    
    config = SimpleConfig()
    
    # Convert URL to audiobook
    print(f"🌐 Converting {args.url} to audiobook...")
    results = converter.convert_url_to_audiobook(args.url, args.output, config)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"✅ Conversion complete!")
        print(f"📁 Output directory: {results['output_dir']}")
        print(f"🎵 Audio files: {len(results['audio_files'])}")

if __name__ == "__main__":
    main()
