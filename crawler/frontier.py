import os
import shelve

from threading import Thread, RLock
from queue import Queue, Empty

from utils import get_logger, get_urlhash, normalize
from scraper import is_valid

import time

class Frontier(object):
    def __init__(self, config, restart):
        self.logger = get_logger("FRONTIER")
        self.config = config
        self.to_be_downloaded = Queue()

        self.lock = RLock()

        self.domains = [
            "ics.uci.edu",
            "cs.uci.edu",
            "informatics.uci.edu",
            "stat.uci.edu",
            "today.uci.edu"
        ]
        self.last_request_time = {domain: 0.0 for domain in self.domains} # Track the last request time for each domain 
        
        if not os.path.exists(self.config.save_file) and not restart:
            # Save file does not exist, but request to load save.
            self.logger.info(
                f"Did not find save file {self.config.save_file}, "
                f"starting from seed.")
        elif os.path.exists(self.config.save_file) and restart:
            # Save file does exists, but request to start from seed.
            self.logger.info(
                f"Found save file {self.config.save_file}, deleting it.")
            os.remove(self.config.save_file)
        # Load existing save file, or create one if it does not exist.
        self.save = shelve.open(self.config.save_file)
        if restart:
            for url in self.config.seed_urls:
                self.add_url(url)
        else:
            # Set the frontier state with contents of save file.
            self._parse_save_file()
            if not self.save:
                for url in self.config.seed_urls:
                    self.add_url(url)

    def _parse_save_file(self):
        ''' This function can be overridden for alternate saving techniques. '''
        with self.lock:
            total_count = len(self.save)
            tbd_count = 0
            for url, completed in self.save.values():
                if not completed and is_valid(url):
                    self.to_be_downloaded.put(url)
                    tbd_count += 1
            self.logger.info(
                f"Found {tbd_count} urls to be downloaded from {total_count} "
                f"total urls discovered.")

    def get_tbd_url(self):
        try:
            return self.to_be_downloaded.get()
        except Empty:
            return None

    def add_url(self, url):
        url = normalize(url)
        urlhash = get_urlhash(url)
        with self.lock:
            if urlhash not in self.save:
                self.save[urlhash] = (url, False)
                self.save.sync()
                self.to_be_downloaded.put(url)
    
    def mark_url_complete(self, url):
        urlhash = get_urlhash(url)

        if urlhash not in self.save:
            # This should not happen.
            self.logger.error(
                f"Completed url {url}, but have not seen it before.")

        self.save[urlhash] = (url, True)
        self.save.sync()

        domain = self._get_domain(url)
        if not domain:
            return

        with self.lock:
            self.last_request_time[domain] = time.time()


    def wait_for_request(self, url):
        domain = self._get_domain(url)
        current_time = time.time()

        if not domain:
            return

        with self.lock:
            elapsed_time = current_time - self.last_request_time[domain]
            # print(f"Requesting domain: {domain} at {elapsed_time}")
            if elapsed_time < 0.5:  # 500 ms
                time.sleep(0.5 - elapsed_time)  # Wait for the remaining time
            
            self.last_request_time[domain] = time.time()
            
    def _get_domain(self, url):
        from urllib.parse import urlparse
        subdomain = urlparse(url).hostname

        for domain in self.domains:
            if subdomain.endswith(domain):
                return domain
        

