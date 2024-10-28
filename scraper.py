import re
from urllib.parse import urlparse, urljoin, parse_qs, urldefrag, urlencode, urlunparse
from bs4 import BeautifulSoup
from collections import defaultdict


class Scraper:
    # Subdomains of uci.edu to crawl within the styx web cache.
    _allowed_domains = {
        "ics.uci.edu",
        "cs.uci.edu",
        "informatics.uci.edu",
        "stat.uci.edu",
    }

    # URL Query parameters that definitely indicate crawler traps.
    _trap_params = {
        "reply", "comment", "message", "print", "format", "output",
        "preview", "draft", "share", "invite", "action", "do"
    }

    # URL Query parameters that might unintentionally cause crawler traps.
    # Still visit these pages, just strip these parameters from URLs.
    _ordering_params = {
        "sort", "order", "filter"
    }

    # Set of English words to ignore. Pulled from the resource linked in the
    # Canvas assignment documentation: https://www.ranks.nl/stopwords
    _stopwords = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
        "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
        "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each",
        "few", "for", "from", "further",
        "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
        "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
        "let's",
        "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not",
        "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
        "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
        "under", "until", "up",
        "very",
        "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
        "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
    }


    def __init__(self):
        self.visited_urls = set() # Used for reporting how many unique pages were found.
        self.subdomain_counts = defaultdict(int) # Used for reporting how many subdomains were found.
        self.site_counts = defaultdict(int) # Used for checking if the crawler has been trapped.
        self.token_counts = defaultdict(int) # Used for reporting the top 50 most common words and for similarity detection.
        self.max_page_len = 0 # Used for reporting the longest page by measure of word count.
        self.site_hashes = set() # Used for similarity detection. Many membership operations, so use a set.
        self.site_uncommon_tokens = [] # Used for similarity detection. Many iterations, so use a list.


    def extract_next_links(self, url, resp):
        # Implementation required.
        # url: the URL that was used to get the page
        # resp.url: the actual url of the page
        # resp.status: the status code returned by the server. 200 is OK, you got the page. Other numbers mean that there was some kind of problem.
        # resp.error: when status is not 200, you can check the error here, if needed.
        # resp.raw_response: this is where the page actually is. More specifically, the raw_response has two parts:
        #         resp.raw_response.content: the content of the page!
        #         resp.raw_response.url: the url, again
        # Return a list with the hyperlinks (as strings) scrapped from resp.raw_response.content

        # Webpage uniqueness is based on the URL, not the content.
        # The total number of unique pages is the sum of either the site_counts or the subdomain_counts.
        parsed_url = urlparse(url)
        self.visited_urls.add(url)
        self.subdomain_counts[parsed_url.hostname] += 1
        self.site_counts[parsed_url.netloc + parsed_url.path] += 1

        if url != resp.url: # If we are redirected, we count both domain names.
            if not is_valid(resp.url): # First ensure the new URL is valid.
                return []
            
            resp_parsed_url = urlparse(resp.url)
            self.visited_urls.add(resp.url)
            self.subdomain_counts[resp_parsed_url.hostname] += 1
            self.site_counts[resp_parsed_url.netloc + resp_parsed_url.path] += 1
        
        # Ignore any HTTP responses that are not 200 OK.
        if resp.status != 200:
            return []

        links = set()
        soup = BeautifulSoup(resp.raw_response.content, 'lxml')

        # URLs will still be counted as visited regardless, but:
        #   - If a webpage's raw HTML is greater than 500 KB, regardless of token count,
        #     it is too large to be worth extracting new links from.
        #
        #   OR
        # 
        #   - If a webpage has less than 50 tokens, regardless of HTML size, it
        #     has low information value and is not worth extracting new links from.
        #
        #   OR
        #
        #   - If a webpage's raw HTML is greater than 300 KB AND its content contains
        #     less than 100 tokens, it is fairly large while also likely having low
        #     information value, so it is not worth extracting new links from.
        content_length = resp.raw_response.headers.get('Content-Length')
        html_size = int(content_length) if content_length else len(resp.raw_response.content.decode('utf-8'))

        text = soup.get_text()
        text.lower()
        tokens = re.findall(r'[^\W_]+', text)

        MAX_HTML_SIZE = 500000
        MIN_TOKENS = 50
        html_too_large = html_size > MAX_HTML_SIZE
        not_enough_tokens = not soup.body or len(tokens) < MIN_TOKENS
        not_enough_tokens_for_large_html = html_size > MAX_HTML_SIZE - 200000 and (not soup.body or len(tokens) < MIN_TOKENS * 2)

        if html_too_large or not_enough_tokens or not_enough_tokens_for_large_html:
            return []

        # Detect and avoid similarity.
        if self.is_similar(tokens):
            return []

        # Update the statistics for the length of the longest page and the token counts.
        self.max_page_len = max(self.max_page_len, len(tokens))
        for token in tokens:
            if token not in self._stopwords:
                self.token_counts[token] += 1

        # Extract all links in the webpage.
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href is not None:
                # Join relative links to the base URL.
                full_url = urldefrag(urljoin(resp.url, href, allow_fragments=False))

                parsed_full_url = urlparse(full_url)
                parsed_full_url.get_url()

                # Remove ordering query parameters from the URL.
                queries = parse_qs(parsed_full_url.query)
                queries_to_remove = []

                for param in queries.keys():
                    if param.split('[')[0] in self._ordering_params:
                        queries_to_remove.append(param)
                
                for param in queries_to_remove: # Can only modify dictionary after loop using it.
                    del queries[param]

                # Reconstruct the URL.
                query_string = urlencode(queries, doseq=True)
                full_url = urlunparse((parsed_full_url.scheme, parsed_full_url.netloc, parsed_full_url.path,
                                       parsed_full_url.params, query_string, parsed_full_url.fragment))

                # Only extract a link if the exact same URL has not been visited
                # and the URL is valid.
                # Query parameters may cause trap links to be extracted, but
                # is_valid() uses is_trap() to exclude likely traps.
                if full_url not in self.visited_urls and is_valid(full_url):
                    links.add(full_url)

        return list(links)
    

    def is_similar(self, tokens):
        # If a given webpage has less than 50 tokens, it has low information value
        # and will already be filtered out. The webpage may have more than 50
        # tokens, in which case take the 50 with the lowest count within the webpage.
        # This guarantees the webpage's hashed fingerprint will be built from the 50
        # most unique tokens within the webpage.
        
        # Build token counts for this webpage.
        page_token_counts = defaultdict(int)
        for token in tokens:
            if token not in self._stopwords:
                page_token_counts[token] += 1
        
        # Extract just the uncommon tokens from this webpage. Use frozenset for
        # immutability to enable hashing and for set intersection.
        sorted_page_token_counts = sorted(page_token_counts.items(), key=lambda item: item[1])
        uncommon_tokens = frozenset(token_count[0] for token_count in sorted_page_token_counts[:51])

        # Detect exact similarity. If the hashed fingerprint of this webpage is the same as any other
        # webpage, then this webpage is likely an exact duplicate of the other webpage.
        site_hash = hash(uncommon_tokens)
        if site_hash in self.site_hashes:
            return True
        
        # Detect near similarity. If 85% or more of the most unique tokens are shared in common
        # with any webpage, then this webpage is likely a near duplicate of the other webpage.
        for other_page_uncommon_tokens in self.site_uncommon_tokens:
            similarity = len(uncommon_tokens & other_page_uncommon_tokens) / len(other_page_uncommon_tokens)
            if similarity >= 0.85:
                return True
        
        # If the webpage is unique (not sufficiently similar to other webpages),
        # update the containers which hold the data used for similarity detection.
        self.site_hashes.add(site_hash)
        self.site_uncommon_tokens.append(uncommon_tokens)

        return False


    def is_valid_domain(self, parsed_url):
        # Return True if the URL's domain is within the set of allowed domains/subdomains.
        domain = parsed_url.hostname
        path = parsed_url.path

        if not domain:
            return False

        if any(domain.endswith(d) for d in self._allowed_domains):
            return True
        # Special case for "today.uci.edu/department/information_computer_sciences/*".
        elif domain == "today.uci.edu" and path.startswith("/department/information_computer_sciences/"):
            return True
        
        return False


    def is_trap(self, parsed_url):
        # Return True if any of the URL's query parameters indicate a potential crawler trap.
        query_string = parsed_url.query
        queries = parse_qs(query_string)
        if any(query_param.split('[')[0] in self._trap_params for query_param in queries.keys()):
            return True

        # Some query parameters (e.g. "page", "start", "offset", "limit", and "idx") do not
        # necessarily indicate crawler traps, but if the same URL (parameters notwithstanding)
        # has been visited more than 10 times, it is likely a trap.
        site = parsed_url.netloc + parsed_url.path
        if self.site_counts[site] > 10:
            return True

        return False
# End class Scraper


s = Scraper()


def scraper(url, resp):
    return s.extract_next_links(url, resp)


def is_valid(url):
    # Decide whether to crawl this url or not. 
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in set(["http", "https"]):
            return False

        if re.match(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
                + r"|png|tiff?|mid|mp2|mp3|mp4"
                + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
                + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
                + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
                + r"|epub|dll|cnf|tgz|sha1"
                + r"|thmx|mso|arff|rtf|jar|csv"
                + r"|rm|smil|wmv|swf|wma|zip|rar|gz)$", parsed_url.path.lower()):
            return False

        if not s.is_valid_domain(parsed_url) or s.is_trap(parsed_url):
            return False

        return True
    except TypeError:
        print ("TypeError for ", parsed_url)
        raise

