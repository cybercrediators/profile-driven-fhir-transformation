"""Loading and storing (json-based) config files"""

import json
import pprint
import sys

import logging
logger = logging.getLogger(__name__)


def load_config(fname):
    """load the json model config"""
    try:
        with open(fname, encoding="UTF-8") as conf:
            config = json.load(conf)
            return config

    except FileNotFoundError:
        logger.error("File not found!")
        sys.exit(1)


def show_config(config):
    """pretty print given model config"""
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)


def save_config(fname, config):
    """Update/Save the given config/file"""
    with open(fname, "w", encoding="UTF-8") as f:
        json.dump(config, f, indent=4)
