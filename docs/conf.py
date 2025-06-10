import os
import sys
import zipfile
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv

# -- Path setup --------------------------------------------------------------

HERE = Path(__file__).parent
sys.path.insert(0, f'{HERE.parent.parent}')
sys.path.insert(0, os.path.abspath('_ext'))
logger = logging.getLogger(__name__)

# -- Project information -----------------------------------------------------

project = 'DILImap'
author = 'Volker Bergen'
master_doc = 'index'
default_role = 'literal'
html_theme = 'sphinx_rtd_theme'
github_repo = 'DILImap'
copyright = '2025 Cellarity'

# -- Options for HTML output ----------------------------------------------

html_theme_options = {'navigation_depth': 1, 'titles_only': True, 'logo_only': True}
html_static_path = ['_static']
html_css_files = ['custom.css']
html_logo = '_static/dilimap_cellarity_logo.png'
html_show_sourcelink = True
html_sourcelink_suffix = '.ipynb'
templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

extensions = [
    'edit_on_github',
    'clean_notebooks',
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
]

# -- Retrieve reproducibility notebooks ----------------------------------------

load_dotenv(dotenv_path=HERE / '../.env')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

if not GITHUB_TOKEN:
    logger.warning('No GitHub token found in .env file. Private notebook access may fail.')

HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'}
BASE_API_URL = 'https://api.github.com/repos/Cellarity/DILImap_reproducibility/contents'
TARGET_DIRS = ['tutorials', 'reproducibility']
suppress_warnings = ['misc.highlighting_failure']


def download_recursively(remote_path: str, local_base: str = '.'):
    """Recursively downloads files from DILImap_reproducibility and mirrors its structure."""
    api_url = f'{BASE_API_URL}/{remote_path}?ref=main'
    response = requests.get(api_url, headers=HEADERS)

    if response.status_code != 200:
        logger.warning(f'Failed to access {remote_path}: {response.status_code}')
        return

    for item in response.json():
        remote_file_path = item['path']

        if item['type'] == 'file':
            local_path = Path(local_base) / remote_file_path
            os.makedirs(local_path.parent, exist_ok=True)
            logger.info(f'Downloading: {remote_file_path}')
            file_content = requests.get(item['download_url'], headers=HEADERS).content
            with open(local_path, 'wb') as f:
                f.write(file_content)

        elif item['type'] == 'dir':
            download_recursively(remote_file_path, local_base)


for dir_name in TARGET_DIRS:
    download_recursively(dir_name)


# -- Add download button to all notebooks -------------------------------------------
nbsphinx_prolog = r"""
.. only:: html

    .. raw:: html

        <p><a href="../{{ env.docname.split('/')[0] }}.zip" download> 📥 Download all notebooks</a></p>
"""


def zip_notebook_dirs(app, exception):
    if app.builder.name != 'html' or exception is not None:
        return

    for folder in ['tutorials', 'reproducibility']:
        src_dir = os.path.join(app.srcdir, folder)
        zip_path = os.path.join(app.outdir, f'{folder}.zip')

        if os.path.exists(src_dir):
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(src_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, start=src_dir)
                        zipf.write(full_path, arcname=os.path.join(folder, rel_path))
            logger.info(f'Created zip archive: {zip_path}')
        else:
            logger.warning(f"Directory '{folder}' not found, skipping zip.")


def setup(app):
    app.connect('build-finished', zip_notebook_dirs)
