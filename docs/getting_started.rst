Getting Started
---------------

This guide provides a quick overview of how to start using DILImap.

Install using ``pip install dilimap`` or follow the detailed installation instructions at
main/install


DILImap workflow
^^^^^^^^^^^^^^^^
Import DILImap as::

    import dilimap as dmap

Read your data
''''''''''''''
Read your data file using::

    adata = dmap.read(filename)

which stores the data matrix (``adata.X``), annotation of cells / observations (``adata.obs``)
and genes / variables (``adata.var``), unstructured annotation such as graphs (``adata.uns``) and
additional data layers such as p-values (``adata.layers``) .

If you don’t have a dataset yet, try one of the built-in options::

    adata = dmap.datasets.demo_counts()

The typical workflow consists of subsequent calls of preprocessing (``dmap.pp.*``), model
predictions (``dmap.models.*``) and plotting (``dmap.pl.*``).

WikiPathways signatures
'''''''''''''''''''''''
We start by applying QC metrics to filter out low-quality samples::

    dmap.pp.qc_metrics(adata, **params)
    dmap.pp.qc_cross_rep_correlation(adata, **params)

Next, we run DESeq2 to identify differentially expressed genes::

    adata_deseq = dmap.pp.deseq2(adata, **params)

Then, we create WikiPathway signatures based on adjusted p-values (FDR) of those differentially
expressed genes::

    FDR = adata_deseq.to_df('FDR')
    adata_wp = dmap.pp.pathway_signature(FDR, **params)

ToxPredictor v2
'''''''''''''''
We apply the ToxPredictor v2 DILI model to these WikiPathway signatures::

    model = dmap.models.ToxPredictor('v2')
    df_results = model.predict(adata_wp, **params)

and computing safety margins::

    df_margins = model.compute_safety_margins(adata_wp, **params)

Visualization
'''''''''''''

Finally, we can visualize the results::

    model.plot_DILI_dose_regimes(cmpd, **params)
