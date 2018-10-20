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

img_folder = "imgs/"
valid_img = ["png", "bmp", "jpg", "jpeg"]

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
        self.hot_entries = []

    def get_memes(self, num=1):
        """
        Get the next top meme/image from /r/subreddit.

        Args:
            num (int): number of memes to pop off

        Returns:
            str: the next image as a URL from /r/subreddit.
        """
        if not self.hot_entries: # NOTE: we populate hot_entries on demand
            self.load_memes()

        hot = []
        if len(self.hot_entries) < num:
            self.hot_entries = []
            return self.hot_entries

        hot = self.hot_entries[:num]
        self.hot_entries = self.hot_entires[num:]

        return hot

    def load_memes(self):
        """
        Load the hot_entries list with the newest set of entries, and stores it
        within the internal self.hot_entries list.

        NOTE: This means that some entries may end up being re-queried (depending
        on what images are new; essentially, refresh the hot_entries queue.)
        """

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

def download_img(url, tgt=None):
    """
    Download the image at the URL.

    Args:
        url (str): a URL, formatted https://*.*/*.jpg
        tgt (str): a filename to save to. If None, we use the basename of url

    Returns:
        The path to the downloaded image; None if there is no image.
    """
    url = urlparse.urlparse(url)
    if not bool(url.scheme):
        raise ValueError("url is invalid")

    if not tgt:
        tgt = img_folder + os.path.basename(url.path)

    os.makedirs(os.path.dirname(tgt), exist_ok=True)

    # validate image extension:
    if tgt[tgt.rfind(".") + 1:] not in valid_img:
        return None

    urllib.urlretrieve(url, tgt)

    # try:
    #     img = Image.open(tgt)
    # except IOError:
    #     print("couldn't get handle to {}".format(tgt))
    #     return None

    return tgt

if __name__ == "__main__":
    # test case:
    reddit = get_secrets('cert.txt')
    gen = MemeGenerator(reddit, 'dankmemes')
    memes = gen.get_memes(1)

    for meme in memes:
        download_img(meme.url)

