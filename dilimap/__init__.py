"""DILImap - Predicting DILI risk using Toxicogenomics"""

from ._version import __version__  # hidden file
from . import logging, s3, datasets, utils, models, clients, preprocessing as pp, plotting as pl
import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['pp', 'pl']})

__all__ = [
    '__version__',
    'logging',
    's3',
    'datasets',
    'utils',
    'pp',
    'pl',
    'models',
    'clients',
]
