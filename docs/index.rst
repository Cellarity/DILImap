Welcome to DILImap
==================

**DILImap** is the largest RNA-seq toxicogenomics library for Drug-Induced Liver Injury (DILI), built using primary human
hepatocytes. It includes transcriptomic profiles from 300 DILI-relevant compounds across multiple doses, setting a new
benchmark for predicting and understanding DILI mechanisms.

**ToxPredictor**, our machine learning model trained on DILImap, delivers state-of-the-art DILI risk assessments —
achieving 88% sensitivity at 100% specificity, outperforming existing methods.

💊 DILImap core applications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Predict dose-resolved **DILI risk for new compounds**
- Estimate DILI **safety margins to prioritize candidates**
- Define tolerable **dose ranges to inform clinical guidance**
- Reveal predictive **DILI pathways linked to known mechanisms**

📚 Getting started
^^^^^^^^^^^^^^^^^^
Use the sidebar to navigate through resources for

- `Installation instructions <install.html>`_
- `Usage principles <getting_started.html>`_
- `Step-by-step tutorials <tutorials/1_Compute_Pathway_Signatures.html>`_
- `Reproducibility notebooks <reproducibility/index.html>`_.

.. toctree::
   :maxdepth: 1
   :caption: Main
   :hidden:

   about
   install
   api


.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   getting_started
   tutorials/1_Compute_Pathway_Signatures
   tutorials/2_Run_ToxPredictor_Model


.. toctree::
   :maxdepth: 1
   :caption: Reproducibility
   :hidden:

   reproducibility/index
