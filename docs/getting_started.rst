Quickstart Guide
----------------

Get started with DILImap in just a few steps.

Install DILImap using::

    pip install dilimap

Or follow the doc:`detailed installation instructions <install>`


DILImap workflow
^^^^^^^^^^^^^^^^
Import the package::

    import dilimap as dmap

Read your data
''''''''''''''
Read your dataset (.h5ad or .csv format)::

    adata = dmap.read(filename)


Don’t have a dataset yet? Use a built-in demo::

    adata = dmap.datasets.demo_data()


WikiPathways signatures
'''''''''''''''''''''''
Apply quality control to filter out low-quality samples::

    dmap.pp.qc_metrics(adata, **params)
    dmap.pp.qc_cross_rep_correlation(adata, **params)

Run DESeq2 to compute differential gene expression::

    adata_deseq = dmap.pp.deseq2(adata, **params)

Compute pathway-level signatures from DESeq2 results::

    FDR = adata_deseq.to_df('FDR')
    adata_wp = dmap.pp.pathway_signature(FDR, **params)

ToxPredictor v1
'''''''''''''''
Apply the DILI prediction model::

    model = dmap.models.ToxPredictor('v1')
    df_results = model.predict(adata_wp, **params)

Estimate safety margins::

    df_margins = model.compute_safety_margins(adata_wp, **params)

Visualization
'''''''''''''

Generate dose-response plots per compound::

    model.plot_DILI_dose_regimes(cmpd, **params)
