import re
from urllib.parse import urlparse, urljoin, parse_qs
from bs4 import BeautifulSoup
from collections import defaultdict

def scraper(url, resp):
    links = extract_next_links(url, resp)

    return [link for link in links if is_valid(link)]

stopwords = {
"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours 	ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
}


def extract_next_links(url, resp):
    # Implementation required.
    # url: the URL that was used to get the page
    # resp.url: the actual url of the page
    # resp.status: the status code returned by the server. 200 is OK, you got the page. Other numbers mean that there was some kind of problem.
    # resp.error: when status is not 200, you can check the error here, if needed.
    # resp.raw_response: this is where the page actually is. More specifically, the raw_response has two parts:
    #         resp.raw_response.url: the url, again
    #         resp.raw_response.content: the content of the page!
    # Return a list with the hyperlinks (as strings) scrapped from resp.raw_response.content
    if resp.status != 200:
        return []

    
    links = []
    soup = BeautifulSoup(resp.raw_response.content, 'lxml')

    # check if should be crawled with tags and text analysis
    nofollow = soup.find("meta", {"name": "robots", "content": re.compile("nofollow")})
    if nofollow:
        return []

    text = soup.get_text()
    # words = text.split(',')
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    # count tokens now

    for link in soup.find_all('a', href=True):
        if link.get('href') is not None and link.get('href') != url:
            defragmented = link.get('href').split('#')[0]

            # For relative links they have to be joined with the base url
            full_url = urljoin(resp.url, defragmented, allow_fragments=False)
            links.append(full_url)

    return links


site_counts = defaultdict(int)
trap_params = { 
    "session", "sid", "token", "auth", "sort", "order", "dir", "filter", "utm_content", 
    "reply", "comment", "message", "print", "format", "output", "preview", "draft", "share", "invite",
    "debug", "test", "action", "do"
}
allowed_domains = { 
    "ics.uci.edu",
    "cs.uci.edu",
    "informatics.uci.edu",
    "stat.uci.edu" 
}

def is_valid(url):
    # Decide whether to crawl this url or not. 
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.

    try:
        parsed = urlparse(url)
        if parsed.scheme not in set(["http", "https"]):
            return False

        if re.match(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
                + r"|png|tiff?|mid|mp2|mp3|mp4"
                + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
                + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
                + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
                + r"|epub|dll|cnf|tgz|sha1"
                + r"|thmx|mso|arff|rtf|jar|csv"
                + r"|rm|smil|wmv|swf|wma|zip|rar|gz)$", parsed.path.lower()):
            return False

        if not is_valid_domain(parsed):
            return False

        if is_trap(parsed):
            return False

        return True

    except TypeError:
        print ("TypeError for ", parsed)
        raise

def is_valid_domain(parsed):
    # Check if the domain matches any of the allowed patterns, including subdomains
    domain = parsed.hostname
    path = parsed.path
    if not domain:
        return False

    if any(domain.endswith(d) for d in allowed_domains):
        return True
        # Special case for "today.uci.edu/department/information_computer_sciences/*"
    elif domain == "today.uci.edu" and path.startswith("/department/information_computer_sciences/"):
        return True
    
    return False

def is_trap(parsed):
    site = parsed.netloc + parsed.path
    # Return false if we receive some parameters that know are bad
    query_string = parsed.query
    params = parse_qs(query_string)
    if any(param in trap_params for param in params.keys()):
        return True

    # "page", "start", "offset", "limit", and "idx" are OK but we have to be careful to not get trapped in an infinite loop
    # Therefore if the same URL, (parameters nonwithstanding) has been crawled more than 10 times, its probably a trap 
    site_counts[site] += 1
    if site_counts[site] > 10:
        return True

    return False
