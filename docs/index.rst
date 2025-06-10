Welcome to DILImap
==================

**DILImap** is the largest RNA-seq toxicogenomics library for Drug-Induced Liver Injury (DILI), built using primary human
hepatocytes. It includes transcriptomic profiles from 300 DILI-relevant compounds across multiple doses, setting a new
benchmark for predicting and understanding DILI mechanisms.

**ToxPredictor**, our machine learning model trained on DILImap, delivers state-of-the-art DILI risk assessments —
achieving 88% sensitivity at 100% specificity, outperforming existing methods.

💊 Core applications
^^^^^^^^^^^^^^^^^^^^
- Predict dose-resolved **DILI risk for new compounds**
- Estimate DILI **safety margins to prioritize candidates**
- Define tolerable **dose ranges to inform clinical guidance**
- Reveal predictive **DILI pathways linked to known mechanisms**

📚 Getting started
^^^^^^^^^^^^^^^^^^
Install DILImap using::

    pip install dilimap

Use the sidebar to navigate through resources for

- `Quickstart Guide <getting_started.html>`_
- `Tutorials <tutorials/1_Compute_Pathway_Signatures.html>`_
- `Paper results <reproducibility.html>`_


.. toctree::
   :maxdepth: 1
   :caption: Main
   :hidden:

   about


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started
   tutorials/1_Compute_Pathway_Signatures
   tutorials/2_Run_ToxPredictor_Model


.. toctree::
   :maxdepth: 1
   :caption: Resources
   :hidden:

   reproducibility
   datasets
   api
   GitHub <https://github.com/Cellarity/DILImap>