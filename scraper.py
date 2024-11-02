import re
from urllib.parse import urlparse, urljoin, parse_qs, urlencode, urlunparse
from bs4 import BeautifulSoup
from collections import defaultdict
import numpy as np


class Scraper:
    # Subdomains of uci.edu to crawl within the styx web cache.
    _allowed_domains = [ "ics.uci.edu", "cs.uci.edu", "informatics.uci.edu", "stat.uci.edu" ]


    # Still visit these pages, just strip these parameters from URLs.
    # URL Query parameters that crawlers should avoid due to either not returning html or causing traps.
    _bad_params = frozenset( [ "reply", "comment", "message", "print", "format", "output",
                              "preview", "draft", "share", "invite", "action", "do", "ical", 
                              ] )
    _order_params = frozenset( [ "sort", "order", "filter"  ] )

    # Set of English words to ignore. Pulled from the resource linked in the
    # Canvas assignment documentation: https://www.ranks.nl/stopwords
    _stopwords = frozenset( [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ] )


    def __init__(self):
        self.visited_urls = set() # Used for reporting how many unique pages were found.
        self.subdomain_counts = defaultdict(int) # Used for reporting how many subdomains were found + unique pages in them.
        self.site_counts = defaultdict(int) # Used for checking if the crawler has been trapped.
        self.token_counts = defaultdict(int) # Used for reporting the top 50 most common words.
        self.max_page_len = 0 # Used for reporting the longest page by measure of word count.

        # Used for tf-idf
        self.MAX_DOCUMENTS = 20
        self.subdomain_similarity = {}  # (n_documents, document_frequency, fingerprints)
        # subdomain_dfs contains the n_documents int and the document_frequency dict for each subdomain


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

        # If status is 404, we don't count the page because it doesn't exist
        if resp.status == 404:
            return []

        # Webpage uniqueness is based on the URL, not the content.
        # The total number of unique pages is len(visited_urls)
        parsed_url = urlparse(url)

        self.visited_urls.add(url)
        self.subdomain_counts[parsed_url.hostname] += 1
        self.site_counts[parsed_url.netloc + parsed_url.path] += 1

        if url != resp.url: # If we are redirected, we count both domain names.
            # replace the parsed_url
            if not is_valid(resp.url) or resp.url in self.visited_urls: # First ensure the new URL is valid.
                # If the current url is invalid because of its parameters, try to visit the non-parameterized url. 
                # Otherwise its irredeemable.

                full_url = self._remove_query_params(resp.url, self._bad_params | self._order_params)
                if full_url not in self.visited_urls and is_valid(full_url):
                    return [full_url]
                return []

            parsed_url = urlparse(resp.url)
            self.visited_urls.add(resp.url)
            self.subdomain_counts[parsed_url.hostname] += 1
            self.site_counts[parsed_url.netloc + parsed_url.path] += 1

        # Ignore any HTTP responses that are not 200
        if resp.status != 200:
            return []

        html_content = resp.raw_response["content"]
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'lxml')
        if not soup or not soup.html: 
            # File is not valid html
            # Gets rid of dead 200 urls
            return []

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

        html_size = len(html_content)
        text = soup.get_text()
        text.lower()
        tokens = re.findall(r'[^\W_]{2,}', text)

        n_informational_tokens = 0
        term_frequency = defaultdict(int)

        for token in tokens:
            if token not in self._stopwords:
                term_frequency[token] += 1
                n_informational_tokens += 1

        # anchor texts are not high information most of the time
        anchors = soup.find_all('a', href=True)
        for anchor in anchors:
            for token in re.findall(r'[^\W_]{2,}', anchor.get_text()):
                term_frequency[token] -= 1
                n_informational_tokens -= 1


        MAX_HTML_SIZE = 500000
        MIN_TOKENS = 50
        html_too_large = html_size > MAX_HTML_SIZE
        not_enough_tokens = n_informational_tokens < MIN_TOKENS
        not_enough_tokens_for_large_html = html_size > MAX_HTML_SIZE - 200000 and (n_informational_tokens < MIN_TOKENS * 2)

        if html_too_large or not_enough_tokens or not_enough_tokens_for_large_html:
            return []

        if parsed_url.hostname not in self.subdomain_similarity:
            self.subdomain_similarity[parsed_url.hostname] = [0, defaultdict(int), []]
        # [0] is n_documents [1] is document_frequency [2] is fingerprints

        similarity = self.subdomain_similarity[parsed_url.hostname]
        # To compare similarity, 20 documents from this subdomain must be seen
        # This avoid the issue of initial innacurate fingerprints
        if similarity[0] < self.MAX_DOCUMENTS:
            # Update document frequency for this subdomain
            similarity[0] += 1
            for token in term_frequency.keys():
                similarity[1][token] += 1


            # Detect and avoid similarity after having calculated the document frequency for 20 documents in this subdomain
            # We freeze the document frequency after 20 document because this should capture the layouts and common general words while keeping the fingerprinting stable
        elif self.is_similar(term_frequency, similarity[1], similarity[2]):
            return []

        # Update the statistics for the length of the longest page and the token counts.
        self.max_page_len = max(self.max_page_len, len(tokens))
        for token in tokens:
            if token not in self._stopwords:
                self.token_counts[token] += 1


        # Extract all links in the webpage.
        links = set()
        for link in anchors:
            href = link.get('href')
            if href is not None:
                # Join relative links to the base URL.
                joined_url = urljoin(resp.url, href, allow_fragments=False)
                full_url = self._remove_query_params(joined_url, self._bad_params | self._order_params)


                # Only extract a link if the exact same URL has not been visited
                # and the URL is valid.
                if full_url not in self.visited_urls and is_valid(full_url):
                    links.add(full_url)

        return list(links)


    def is_similar(self, term_frequency, document_frequency, fingerprints):
        # SimHash algorithm, fixing binary hash width to 1st through 64 bits,
        # which avoids a preceding negative. Oftentimes the width of the
        # binary-formatted string of the hash will be slightly less than 64 so we need to adjust

        WIDTH = 64
        THRESHOLD = 0.80

        # Build 64-dimensional vector V to hold weighted components.
        vec_v = np.zeros(WIDTH, dtype=np.float64)

        for token, frequency in term_frequency.items():
            if frequency == 0:
                continue
            tf = frequency / len(term_frequency)
            idf = np.log10(self.MAX_DOCUMENTS / (1 + document_frequency[token]))

            # if it appeared in all 20 documents or just 19
            if idf <= 0:
                idf = 0.001

            # weight
            tf_idf = tf * idf

            # Convert the hash of the current token to a binary string
            hashed_token = hash(token)
            hashed_str = '{:b}'.format(hashed_token)

            # Sign extend manually
            if hashed_token < 0:
                hashed_str = hashed_str[1:]
                for _ in range(64 - len(hashed_str)):
                    hashed_str = '1' + hashed_str
            else:
                for _ in range(64 - len(hashed_str)):
                    hashed_str = '0' + hashed_str

            bits = np.array([1.0 if bit == '1' else -1.0 for bit in hashed_str], dtype=np.float64)
            vec_v += bits * tf_idf

        # Reduce V back to binary vased on whether V[i] is positive or negative.
        # V is now the fingerprint of this webpage.
        vec_v = np.where(vec_v >= 0, 1, 0)

        for fingerprint in fingerprints:
            same_bits = np.sum(vec_v == fingerprint)
            similarity = same_bits / WIDTH
            if similarity >= THRESHOLD: 
                return True

        # If the webpage is unique (not sufficiently similar to other webpages),
        # update the list of fingerprints for visited webpages.
        fingerprints.append(vec_v)


        return False


    def is_valid_domain(self, parsed_url):
        # Return True if the URL's domain is within the set of allowed domains/subdomains.
        domain = parsed_url.hostname
        path = parsed_url.path

        if not domain:
            return False

        for d in self._allowed_domains:
            if domain == d or domain.endswith("." + d):
                return True

        # Special case for "today.uci.edu/department/information_computer_sciences/*".
        if domain == "today.uci.edu" and path.startswith("/department/information_computer_sciences/"):
            return True

        return False


    def is_trap(self, parsed_url):
        # Return True if any of the URL's query parameters indicate a potential crawler trap.
        query_string = parsed_url.query
        queries = parse_qs(query_string)
        if any(query_param.split('[')[0] in self._bad_params for query_param in queries.keys()):
            return True

        # Some query parameters like the order_params or "page", "start", "offset", "limit", and "idx" do not
        # necessarily indicate crawler traps, but if the same URL (parameters notwithstanding)
        # has been visited more than 10 times, it is likely a trap.
        site = parsed_url.netloc + parsed_url.path
        if self.site_counts[site] > 10:
            return True

        return False

    def get_top_words(self):
        # Sort all tokens by highest count and take the first 50
        sorted_tokens = sorted(self.token_counts.items(), key=lambda item: item[1], reverse=True)
        return sorted_tokens[:50]

    def _remove_query_params(self, url, query_params_to_avoid):
        # Remove any query parameters that are known to cause traps.
        # Also removes any fragments.
        parsed_url = urlparse(url)
        queries = parse_qs(parsed_url.query)

        params_to_remove = []
        for param in queries.keys():
            if param.split('[')[0] in query_params_to_avoid:
                params_to_remove.append(param)
        for param in params_to_remove: # Can modify dictionary now
            del queries[param]

        query_string = urlencode(queries, doseq=True)

        full_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, query_string, ''))
        return full_url

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
                + r"|rm|smil|wmv|swf|wma|zip|rar|gz"
                + r"|ppsx|txt)$", parsed_url.path.lower()):
            return False

        if not s.is_valid_domain(parsed_url) or s.is_trap(parsed_url):
            return False

        return True
    except TypeError:
        print ("TypeError for ", parsed_url)
        raise

