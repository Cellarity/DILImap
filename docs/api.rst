.. automodule:: dilimap

API Docs
========

Import DILImap as::

   import dilimap as dmap


After reading the data from local, quilt (``dmap.quilt.read``) or loading an in-built dataset
(``dmap.datasets.*``), the workflow includes preprocessing (``dmap.pp.*``), model predictions
(``dmap.models.*``), plotting (``dmap.pl.*``) and several utility functions (``dmap.utils.*``).


Read/write from S3
------------------

.. autosummary::
   :toctree: .

   s3.read
   s3.write

Preprocessing (pp)
------------------

**QC**

.. autosummary::
   :toctree: .

   pp.qc_metrics
   pp.qc_cross_rep_correlation

**Preprocessing**

.. autosummary::
   :toctree: .

   pp.deseq2
   pp.pathway_signatures

Models
------

.. autosummary::
   :toctree: .

   models.ToxPredictor

Plotting (pl)
-------------

.. autosummary::
   :toctree: .

   pl.roc_curve
   pl.boxplot_with_swarm

Datasets
--------

.. autosummary::
   :toctree: .

   datasets.compound_Cmax_values
   datasets.compound_DILI_labels
   datasets.compound_cell_viability
   datasets.demo_data
   datasets.DILImap_training_data
   datasets.DILImap_validation_data
   datasets.DILImap_data

Clients
-------

.. autosummary::
   :toctree: .

   clients.chembl
   clients.drug_warnings

Utils
-----

.. autosummary::
   :toctree: .

   utils.platemap
   utils.groupby
   utils.crosstab
