"""DILImap - Predicting DILI risk using Toxicogenomics"""

import os
import sys
import warnings

# Set SCIPY_ARRAY_API for compatibility with Google colab
if os.environ.get('SCIPY_ARRAY_API') != '1':
    os.environ['SCIPY_ARRAY_API'] = '1'
    if 'scipy' in sys.modules:
        warnings.warn(
            '⚠️ To ensure full compatibility in Google Colab, please set\n'
            "`os.environ['SCIPY_ARRAY_API'] = '1'` at the very start of your notebook.\n",
            stacklevel=2,
        )

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
