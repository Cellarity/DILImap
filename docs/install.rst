Installation
------------

DILImap requires Python 3.8 or later. We recommend to use Miniconda_.

PyPI
^^^^

Install DILImap from PyPI_ using::

    pip install -U dilimap

``-U`` is short for ``--upgrade``.
If you get a ``Permission denied`` error, use ``pip install -U dilimap --user`` instead.


Development Version
^^^^^^^^^^^^^^^^^^^

To work with the latest development version, install from GitHub_ using::

   pip install git+https://github.com/Cellarity/DILImap@main

or::

    git clone https://github.com/Cellarity/DILImap && cd DILImap
    pip install -e .

``-e`` is short for ``--editable`` and links the package to the original cloned
location such that pulled changes are also reflected in the environment.


Jupyter Notebook
^^^^^^^^^^^^^^^^

To run the tutorials in a notebook locally, please install::

   conda install notebook

and run ``jupyter notebook`` in the terminal.


If you run into issues, do not hesitate to approach us or raise a `GitHub issue`_.

.. _Rodia: https://python-packages-dev.cellarity.com/rodia/
.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _PyPI: https://pypi.org/project/dilimap
.. _Github: https://github.com/Cellarity/DILImap
.. _`Github issue`: https://github.com/Cellarity/DILImap/issues/new/choose
