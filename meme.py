# -*- coding: utf-8 -*-
"""
Module for querying reddit for 2018's top classy memes, for use in swapping faces.

Example Usage:

Todo:
    * For todos.

"""

import praw
import requests

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

if __name__ == "__main__":
    # test case:
    reddit = get_secrets('cert.txt')
    gen = MemeGenerator(reddit, 'dankmemes')
    meme = gen.get_meme()

    print(meme)
