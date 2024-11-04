import re
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict
from urllib.parse import urlparse, urljoin, parse_qs, urlencode, urlunparse


class Scraper:
    # Subdomains of uci.edu to crawl within the styx web cache.
    _allowed_domains = [ "ics.uci.edu", "cs.uci.edu", "informatics.uci.edu", "stat.uci.edu" ]

    # Immutable set of query parameters which indicate dynamic pages.
    # To avoid traps, strip any parameters not in this set from all URLs.
    _good_params = frozenset( [ "p", "page", "paged", "baldiPage", "page_id", "id", "seminar_id", "attachment_id" "archive_year", "year", "limit", "people", "start", "offset", "limit", "idx", "s", "search", "q", "query", "eventDisplay", "tribe-bar-date", "redirect_to"] )

    # Set of English words to ignore. Pulled from the resource linked in the
    # Canvas assignment documentation: https://www.ranks.nl/stopwords
    _stopwords = frozenset( [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ] )


    def __init__(self):
        # Attributes used for reporting statistics.
        self.visited_urls = set() # Number of unique pages found. Also used to avoid duplicate pages.
        self.subdomain_counts = defaultdict(int) # Number of subdomains found, and number of unique pages in them.
        self.token_counts = defaultdict(int) # The top 50 most common words.
        self.max_page_url = "" # URL of the longest page.
        self.max_page_len = 0 # Longest page by measure of word count.

        # Attribute used for checking if the crawler has been trapped.
        self.site_counts = defaultdict(int)

        # Attributes used for checking document similarity via tf-idf.
        self.MAX_DOCUMENTS = 20
        # Dictionary that maps unique subdomain URLs to lists of [number of
        # documents, token-frequency mapping, and document fingerprints], where
        # each list contains data specific to the subdomain it is mapped with.
        self.subdomain_similarity = {}
    

    def get_top_words(self):
        # Sort all tokens by descending count and take the first 50.
        sorted_tokens = sorted(self.token_counts.items(), key=lambda item: item[1], reverse=True)
        return sorted_tokens[:50]


    def scrape_page(self, url, resp):
        # Implementation required.
        # url: the URL that was used to get the page
        # resp.url: the actual url of the page
        # resp.status: the status code returned by the server. 200 is OK, you got the page. Other numbers mean that there was some kind of problem.
        # resp.error: when status is not 200, you can check the error here, if needed.
        # resp.raw_response: this is where the page actually is. More specifically, the raw_response has two parts:
        #         resp.raw_response.content: the content of the page!
        #         resp.raw_response.url: the url, again
        # Return a list with the hyperlinks (as strings) scrapped from resp.raw_response.content

        parsed_url = self._get_parsed_url(url, resp)
        if parsed_url is None:
            # If a new, valid URL cannot be parsed, do not crawl.
            return []

        # Parse this page's content.
        content_type = resp.raw_response.headers.get('Content-Type', '')
        charset = re.search(r'charset=([^;\s]+)', content_type)
        encoding = charset.group(1) if charset else 'utf-8'
        html_content = resp.raw_response.content.decode(encoding, errors='replace')

        # If the HTML content is empty, do not crawl.
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'lxml')
        if not soup or not soup.html:
            # If the document is not valid HTML or is from a dead 200 page, do not crawl.
            return []
        
        # Find anchor tags to extract next links from. They are usually menu elements or otherwise low
        # information value, so separate them from informational tokens.
        anchors = soup.find_all('a', href=True)
        if not anchors and not soup.find('div'):
            # If the document does not contain anchor or division tags, it is likely not valid HTML.
            return []
        for element in anchors:
            element.extract()

        term_frequencies = defaultdict(int) # Frequency of each token in the document.

        # Count isolated informational tokens (alphanumeric sequences of length 2 or more with no underscores).
        informational_tokens = re.findall(r'[^\W_]{2,}', soup.get_text().lower())
        num_info_tokens = 0
        for token in informational_tokens:
            if token not in self._stopwords:
                term_frequencies[token] += 1
                num_info_tokens += 1

        # Still include anchor text tokens in the total count since page layouts often share anchors.
        total_num_tokens = len(informational_tokens)
        for anchor in anchors:
            anchor_tokens = re.findall(r'[^\W_]{2,}', anchor.get_text().lower())
            total_num_tokens += len(anchor_tokens)
            for token in anchor_tokens:
                if token not in self._stopwords:
                    term_frequencies[token] += 1

        # If the document has low information value or is simply too large, do not crawl.
        if not self._has_high_information_value(html_content, num_info_tokens):
            return []

        if parsed_url.hostname not in self.subdomain_similarity:
            # Initialize similarity record for new subdomains with 0 documents, an empty token mapping, and no fingerprints.
            self.subdomain_similarity[parsed_url.hostname] = [0, defaultdict(int), []]

        # Scrape 20 pages/documents from this subdomain to capture a foundation of common words
        # and page layouts. Only begin fingerprinting after this training period and within the 
        # same subdomain for greater reliability.
        similarity = self.subdomain_similarity[parsed_url.hostname]
        if similarity[0] < self.MAX_DOCUMENTS:
            similarity[0] += 1
            for token in term_frequencies.keys():
                similarity[1][token] += 1
        elif self._is_similar(term_frequencies, total_num_tokens, similarity[1], similarity[2]):
            # Do not crawl exact or near duplicate pages after training period.
            return []

        # Update longest page length and top 50 token counts statistics.
        if total_num_tokens > self.max_page_len:
            self.max_page_url = resp.url
            self.max_page_len = total_num_tokens
        for token, count in term_frequencies.items():
            self.token_counts[token] += count
        
        return self._extract_next_links(resp.url, anchors)


    def _get_parsed_url(self, url, resp):
        # If the HTTP status is 404, the URL will not even be counted as it does not exist.
        if resp.status == 404:
            return None

        # The statistic counting the number of unique pages found is based on the URL, not the content.
        parsed_url = urlparse(url)
        self.visited_urls.add(url)
        self.subdomain_counts[parsed_url.hostname] += 1

        # Increment times visited count used to check for traps.
        self.site_counts[parsed_url.netloc + parsed_url.path] += 1

        # Case encountered when the crawler is redirected to a different URL.
        if url != resp.url:
            # If this page has already been visited, do not crawl.
            if resp.url in self.visited_urls:
                return None

            # If the URL redirected to is invalid, check its deparameterized
            # version. If that is also invalid, do not crawl.
            if not is_valid(resp.url):
                # Strip all query parameters not in the _good_params set,
                # standardizing our URL and avoiding traps.
                redirect_url = self._remove_query_params(resp.url)

                if redirect_url in self.visited_urls or not is_valid(redirect_url):
                    return None
                
                # If the deparameterized version of the URL redirected to is valid,
                # use it to replace parsed_url.
                resp.url = redirect_url

            # If the URL redirected to is valid, count it towards statistics too.
            parsed_url = urlparse(resp.url) # Replace parsed_url.
            self.visited_urls.add(resp.url)
            self.subdomain_counts[parsed_url.hostname] += 1

            # Increment times visited count used to check for traps.
            self.site_counts[parsed_url.netloc + parsed_url.path] += 1

        # If the HTTP status is not 200 OK, do not crawl.
        if resp.status != 200:
            return None

        # If this page has been visited too many times with different query parameters,
        # it is a trap, so do not crawl.
        if self._is_trap(parsed_url):
            return None
        
        return parsed_url
    

    def _has_high_information_value(self, html_content, num_info_tokens):
        # URLs will still be counted as visited regardless, but:
        #   - If a page's raw HTML is greater than 500 KB, regardless of token
        #     count, it is too large to be worth extracting new links from.
        #
        #   OR
        # 
        #   - If a page has less than 50 informational tokens, regardless of HTML size,
        #     it has low information value and is not worth extracting new links from.
        #
        #   OR
        #
        #   - If a page's raw HTML is greater than 300 KB AND its content contains
        #     less than 100 informational tokens, it is fairly large while also likely
        #     having low information value, so it is not worth extracting new links from.
        MAX_HTML_SIZE = 500000
        MIN_TOKENS = 50
        html_size = len(html_content) # Size of the HTML content in bytes.

        html_too_large = html_size > MAX_HTML_SIZE
        not_enough_tokens = num_info_tokens < MIN_TOKENS
        not_enough_tokens_for_large_html = (html_size > MAX_HTML_SIZE - 200000) and (num_info_tokens < MIN_TOKENS * 2)

        if html_too_large or not_enough_tokens or not_enough_tokens_for_large_html:
            return False


    def _is_similar(self, term_frequencies, num_terms, document_frequencies, fingerprints):
        # SimHash algorithm, fixing binary hash width to 64 bits. Weight words using
        # term frequency -- inverse document frequency (tf-idf).
        WIDTH = 64
        THRESHOLD = 0.80

        # Build 64-dimensional vector V to hold weighted components.
        vec_v = np.zeros(WIDTH, dtype=np.float64)

        for token, frequency in term_frequencies.items():
            if frequency == 0: # Exclude uncounted tokens.
                continue

            tf = frequency / num_terms
            idf = np.log10(self.MAX_DOCUMENTS / (1 + document_frequencies[token]))

            # If the token appeared in 19 or 20 documents, set a minimum above 0
            # so the weight is still captured, just minimally.
            if idf <= 0:
                idf = 0.001

            # Token weight.
            tf_idf = tf * idf

            # Convert the hash of the current token to a binary string.
            hashed_token = hash(token)
            hashed_str = '{:b}'.format(hashed_token)

            # Oftentimes the width of the binary-formatted string of the token's hash
            # will be slightly less than 64 bits, so manually sign extend it.
            extend_bit = '0'
            if hashed_token < 0:
                hashed_str = hashed_str[1:]
                extend_bit = '1'
            
            for _ in range(64 - len(hashed_str)):
                hashed_str = extend_bit + hashed_str

            bits = np.array([1.0 if bit == '1' else -1.0 for bit in hashed_str], dtype=np.float64)
            vec_v += bits * tf_idf

        # Reduce V back to binary vased on whether V[i] is positive or negative.
        # V is now the fingerprint of this webpage.
        vec_v = np.where(vec_v >= 0, 1, 0)

        for fingerprint in fingerprints:
            # Check for similarity by comparing the number of bits that are the same between the fingerprints.
            same_bits = np.sum(vec_v == fingerprint)
            similarity = same_bits / WIDTH
            if similarity >= THRESHOLD: 
                return True

        # If the webpage is unique (not sufficiently similar to other webpages),
        # update the list of fingerprints for visited webpages.
        fingerprints.append(vec_v)

        return False


    def _extract_next_links(self, base_url, anchors):
        # Extract all linked URLs from the webpage.
        links = set()
        for link in anchors:
            href = link.get('href')
            if href is not None:
                # Join relative links to the base URL.
                joined_url = urljoin(base_url, href, allow_fragments=False)
                # Strip all query parameters not in the set of known good parameters.
                link = self._remove_query_params(joined_url)

                if link not in self.visited_urls and is_valid(link):
                    links.add(link)

        return list(links)


    def _is_valid_domain(self, parsed_url):
        # Return True if the URL's domain is within the set of allowed domains/subdomains.
        domain = parsed_url.hostname
        path = parsed_url.path

        if not domain:
            return False

        for ad in self._allowed_domains:
            if domain == ad or domain.endswith("." + ad):
                return True

        # Special case for "today.uci.edu/department/information_computer_sciences/*".
        if domain == "today.uci.edu" and path.startswith("/department/information_computer_sciences/"):
            return True

        return False


    def _is_trap(self, parsed_url):
        # If the same page with different query parameters has been visited more than 5 times, it is likely a trap.
        site = parsed_url.netloc + parsed_url.path
        return self.site_counts[site] > 5


    def _remove_query_params(self, url):
        # Remove any query parameters that are known to cause traps.
        # Also remove any fragments.
        parsed_url = urlparse(url)
        queries = parse_qs(parsed_url.query)

        params_to_remove = []
        for param in queries.keys():
            if param.split('[')[0] not in self._good_params:
                params_to_remove.append(param)
        for param in params_to_remove: # Modify dictionary after identifying bad parameters.
            del queries[param]

        query_string = urlencode(queries, doseq=True)

        return urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, query_string, ''))

# End class Scraper


s = Scraper()


def scraper(url, resp):
    return s.scrape_page(url, resp)


def is_valid(url):
    # Decide whether to crawl this url or not. 
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in set(["http", "https"]):
            return False

        # Filter out files that are not HTML/webpages.
        if re.match(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
                + r"|png|tiff?|mid|mp2|mp3|mp4"
                + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
                + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
                + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
                + r"|epub|dll|cnf|tgz|sha1"
                + r"|thmx|mso|arff|rtf|jar|csv"
                + r"|rm|smil|wmv|swf|wma|zip|rar|gz"
                + r"|ppsx|pps|txt|bib|sql|xml|pov|tsv|mat|in|out|scm|db"
                + r"|1_manual|2_manual|mpg|img|svg|webp|heic|lif|hqx|fig"
                + r"|lsp|java|war|c|h|cpp|hpp|cp|sh|ss|pl|rss|ff"
                + r"|rle|Z|shar|ova|edelsbrunner|class|prn"
                + r"|conf|cls|can|odp|results|sas|odc|ma|pd|mol|grm|nb)$", parsed_url.path.lower()):
            return False

        if not s._is_valid_domain(parsed_url) or s._is_trap(parsed_url):
            return False

        return True
    except TypeError:
        print ("TypeError for url")
        return False

