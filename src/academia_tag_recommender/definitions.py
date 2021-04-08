"""This module holds definitions for folder locations.

Attributes:
    DATA_PATH: Path to get data from.
    MODELS_PATH: Path to store models to.
"""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(ROOT_DIR, 'data')

MODELS_PATH = os.path.join(ROOT_DIR, 'models')
