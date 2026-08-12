DILImap & ToxPredictor
======================

**DILImap** is a pioneering toxicogenomics atlas purpose-built for modeling Drug-Induced Liver Injury (DILI).
It features RNA-seq profiles from primary human hepatocytes exposed to 300 diverse compounds. It offers a powerful foundation for mechanistic insight and predictive modeling.

**ToxPredictor**, trained on DILImap, sets a new benchmark in DILI risk assessment. With 88% sensitivity at 100% specificity,
it outperforms traditional methods and reliably flags even the most elusive DILI-prone drugs, including withdrawn and idiosyncratic compounds.

**Publication**: `Bergen et al. (2025) <https://doi.org/10.1038/s41467-025-65690-3>`_

.. image:: https://github.com/user-attachments/assets/bbe2af70-c8ee-4fe1-8448-0858b9d1dd09
   :width: 724
   :alt: image

Usage & Support
---------------

- **Installation**: Install using :code:`pip install dilimap`

- **Documentation**: Visit `dilimap.org <https://dilimap.org/>`_ for detailed usage instructions.

- **Reproducibility**: Notebooks and results are at `github.com/Cellarity/DILImap_reproducibility <https://github.com/Cellarity/DILImap_reproducibility>`_.

- **Support**: Reach out at `DILImap@cellarity.com <mailto:DILImap@cellarity.com>`_ with any questions or feedback.

Citation
--------

If you use DILImap, ToxPredictor or associated resources, please cite::

    Bergen, V., Kodella, K., Srikrishnan, S. et al. A large-scale human toxicogenomics resource for drug-induced liver injury prediction. Nat Commun 16, 9860 (2025). https://doi.org/10.1038/s41467-025-65690-3

Data
----

The data have been deposited in GEO under a Creative Commons license (GSE308567). Processed training data (pathway-level signatures used as model input) and both raw and processed validation data are provided to enable full reproducibility of all model training and validation steps. The raw training data (gene expression count matrices) contain proprietary information and are not publicly available. Academic researchers may request access for internal, non-commercial use via DILImap@cellarity.com, with requests reviewed within 4–8 weeks. Approved data are available for 2 weeks and must be deleted within 6 months. Commercial access requires a data-sharing agreement.


.. |actions| image:: https://github.com/Cellarity/dilimap/actions/workflows/ci.yaml/badge.svg?branch=main
   :target: https://github.com/Cellarity/dilimap/actions/workflows/ci.yaml
   :alt: Continuous Integration Status: GitHub actions

.. |rtd| image:: https://readthedocs.com/projects/vl49-dilimap/badge/?version=latest&token=41c19a846e0be9f882d76528ec32ac7358d7234fd0cf158051bf81965d7d5359
   :target: https://docs.cellarity.com/dilimap/en/latest/?badge=latest
   :alt: Documentation Status
