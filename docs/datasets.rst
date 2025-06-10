Datasets
========
All datasets are accessible through the DILImap Python API. To get started, install the package::

    pip install dilimap

For example, to load DILI labels for all compounds, run::

    import dilimap as dmap
    df = dmap.datasets.compound_DILI_labels()

**Note**: Raw training data marked with an asterisk (*) are proprietary and available only upon request, subject to data-sharing agreements.

Compound-level data
-------------------

* :doc:`Consensus Cmax values <dilimap.datasets.compound_Cmax_values>` (Median Cmax per compound aggregated from 20+ studies)
* :doc:`Compound DILI labels <dilimap.datasets.compound_DILI_labels>` (DILI annotations derived from DILIrank and LiverTox)
* :doc:`Cell viability data <dilimap.datasets.compound_cell_viability>` (Raw viability data and IC₁₀ estimates)

DILImap training data
---------------------

* :doc:`Raw counts* <dilimap.datasets.DILImap_training_data>` (Raw gene expression counts)
* :doc:`Gene signatures* <dilimap.datasets.DILImap_training_data>` (Differential gene expression from DESeq2)
* :doc:`Pathway signatures <dilimap.datasets.DILImap_training_data>` (Pathway activation scores computed using WikiPathways)

DILImap validation data
-----------------------
* :doc:`Raw counts <dilimap.datasets.DILImap_validation_data>` (Raw gene expression counts)
* :doc:`Gene signatures <dilimap.datasets.DILImap_validation_data>` (Differential gene expression from DESeq2)
* :doc:`Pathway signatures <dilimap.datasets.DILImap_validation_data>` (Pathway activation scores computed using WikiPathways)

Models
------
* :doc:`ToxPredictor model <dilimap.models.ToxPredictor>` (ToxPredictor DILI prediction model)
