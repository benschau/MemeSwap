# -*- coding: utf-8 -*-
"""
Module for querying reddit for 2018's top classy memes, for use in swapping faces.

Example Usage:

Todo:
    * For todos.

"""

import os, sys

import praw
import requests, urlparse
import bs4 as soup
from PIL import Image

class MemeGenerator:
    def __init__(self, reddit, subreddit, limit=25):
        """
        Args:
            reddit (praw.reddit): an instance of PRAW that has been successfully authenticated,
                                  or in read-only mode.
            subreddit (string): the name of the subreddit to browse
            limit (int): the number of hottest submissions to retain

        """
        if reddit == None:
            raise ValueError("reddit auth cannot be null")

        self.reddit = reddit
        self.subreddit = reddit.subreddit(subreddit)

        self.limit = limit
        self.hot_entries = None

    def get_meme(self):
        """
        Get the next top meme/image from /r/subreddit.

        Returns:
            str: the next image as a URL from /r/subreddit.
        """
        if not self.hot_entries: # NOTE: we populate hot_entries on demand
            self.load_memes()

        hot = self.hot_entries.pop(0)
        return hot.url

    def load_memes(self):
        """
        Load the hot_entries list with the newest set of entries, and stores it
        within the internal self.hot_entries list.

        NOTE: This means that some entries may end up being re-queried (depending
        on what images are new; essentially, refresh the hot_entries queue.)
        """

        self.hot_entries = []
        for submission in self.subreddit.hot(limit=self.limit):
            self.hot_entries.append(submission)

def get_secrets(cert_path):
    """
    Load the client_id, client_secret, and user_agent from a file
    specified by cert.

    Args:
        cert_path (str): certification file to read application specific oauth2 fields

    Returns:
        a read-only praw.Reddit object, or None on failure.
    """
    fields = { 'client_id': None, 'client_secret': None, 'user_agent': None }
    cert = open(cert_path, 'r')

    for line in cert:
        entry = line.split(':')

        field = entry[0]
        val = entry[1].replace('\n', '')

        if field in fields:
            fields[field] = val
        else:
            return None

    reddit = praw.Reddit(client_id=fields['client_id'],
                         client_secret=fields['client_secret'],
                         user_agent=fields['user_agent'])

    if not reddit.read_only:
        return None

    return reddit

def download_img(url, tgt=None, albums=False):
    """
    Download the image(s) at the URL.

    Args:
        url (str): a URL, formatted https://*.*/*.jpg
        tgt (str): a filename to save to. If None, we use the basename of url
        albums (bool): a boolean indicating whether to download an entire album at
                       the url (currently not functioning as intended)

    Returns:
        A handle to the downloaded image; None if there is no image.
    """
    url = urlparse.urlparse(url)
    if not bool(url.scheme):
        raise ValueError("url is invalid")

    if not tgt:
        tgt = os.path.basename(url.path)

    urllib.urlretrieve(url, tgt)

    try:
        img = Image.open(tgt)
    except IOError:
        print("couldn't get handle to {}".format(tgt))
        return None

    return img

if __name__ == "__main__":
    # test case:
    reddit = get_secrets('cert.txt')
    gen = MemeGenerator(reddit, 'dankmemes')

    meme = gen.get_meme()
    print(meme)

    meme = gen.get_meme()
    print(meme)
    download_img(meme)

