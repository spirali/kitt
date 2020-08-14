import contextlib
import getpass
import logging
import os
import socket

import requests


@contextlib.contextmanager
def announce_computation():
    url = _get_mattermost_url()
    if "NO_ANNOUNCE" in os.environ:
        url = None

    def send(message):
        if url:
            _send_mattermost_message(message, url)

    user = getpass.getuser()
    device = socket.gethostname()
    send(f"{user} started computing on {device} :cat:")

    def finished():
        send(f"{user} finished computing on {device} :tada:")

    try:
        yield
        finished()
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            finished()
        else:
            send(f"{user}'s computation crashed :scream_cat:")
        raise e


def _get_mattermost_url():
    return os.environ.get("MM_ANNOUNCE_URL")


def _send_mattermost_message(message, url):
    try:
        requests.post(url, json={"text": message})
    except BaseException as e:
        logging.warning(f"Error while announcing computation: {e}")
