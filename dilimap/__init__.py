"""DILImap - Predicting DILI risk using Toxicogenomics"""

import os
import sys
import warnings

# Set SCIPY_ARRAY_API for compatibility with Google colab
if 'google.colab' in sys.modules and os.environ.get('SCIPY_ARRAY_API') != '1':
    os.environ['SCIPY_ARRAY_API'] = '1'
    if 'scipy' in sys.modules:
        warnings.warn("'SCIPY_ARRAY_API' was set to '1' for compatibility in Colab.", stacklevel=2)

from ._version import __version__  # hidden file
from . import logging, s3, datasets, utils, models, clients, preprocessing as pp, plotting as pl

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
