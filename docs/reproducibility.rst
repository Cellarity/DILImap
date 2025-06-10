Reproducibility
===============
This directory contains the full set of Jupyter notebooks used
to generate the results shown in the DILImap publication

Data Preparation
----------------

.. toctree::
   :hidden:
   :maxdepth: 1

   reproducibility/1.1_DataPrep_DILI_Labels
   reproducibility/1.2_DataPrep_Cmax_Values
   reproducibility/1.3_DataPrep_Viability_Assay

:doc:`reproducibility/1.1_DataPrep_DILI_Labels`
    Assigns consensus DILI labels by integrating clinical annotations with compound metadata.
:doc:`reproducibility/1.2_DataPrep_Cmax_Values`
    Aggregates Cmax values from 20+ studies to derive a consensus median Cmax per compound.
:doc:`reproducibility/1.3_DataPrep_Viability_Assay`
    Fits dose–response curves to raw viability data to estimate IC₁₀ values.

Model Training
--------------

.. toctree::
   :hidden:
   :maxdepth: 1

   reproducibility/2.1_Training_Gene_Signatures
   reproducibility/2.2_Training_Pathway_Signatures
   reproducibility/2.3_Training_ToxPredictor_Model

:doc:`reproducibility/2.1_Training_Gene_Signatures`
    *Generates compound-level gene signatures using DESeq2 after QC filtering.*
:doc:`reproducibility/2.2_Training_Pathway_Signatures`
    *Computes pathway-level signatures via enrichment of DE results using WikiPathways.*
:doc:`reproducibility/2.3_Training_ToxPredictor_Model`
    *Trains and tunes ensemble models; selects the final random forest classifier.*

Model Validation
----------------

.. toctree::
   :hidden:
   :maxdepth: 1

   reproducibility/3.1_Validation_Gene_Signatures
   reproducibility/3.2_Validation_Pathway_Signatures
   reproducibility/3.3_Validation_ToxPredictor_Model

:doc:`reproducibility/3.1_Validation_Gene_Signatures`
    *Generates gene signatures for held-out compounds using DESeq2 after QC filtering.*
:doc:`reproducibility/3.2_Validation_Pathway_Signatures`
    *Computes pathway-level signatures for validation compounds.*
:doc:`reproducibility/3.3_Validation_ToxPredictor_Model`
    *Applies ToxPredictor to unseen validation compounds to compute risk scores and safety margins.*

Results & Benchmarking
----------------------

.. toctree::
   :hidden:
   :maxdepth: 1

   reproducibility/4.1_Results_Main_Figures
   reproducibility/4.2_Benchmarking_Insilico_Models
   reproducibility/4.3_Benchmarking_Invitro_Models

:doc:`reproducibility/4.1_Results_Main_Figures`
    *Reproduces all main figures and tables from the manuscript that are not covered in validation or training notebook.*
:doc:`reproducibility/4.2_Benchmarking_Insilico_Models`
    *Benchmarks ToxPredictor against published in-silico DILI models.*
:doc:`reproducibility/4.3_Benchmarking_Invitro_Models`
    *Benchmarks ToxPredictor against published in-vitro DILI models.*
