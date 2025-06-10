"""Preprocessing functions for quality control, DESeq2 analysis, and pathway signatures"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse
from dilimap.deseq2.deseq_pert import model_all_perts


def qc_metrics(adata):
    """
    Compute quality control (QC) metrics for gene expression data.

    Args:
        adata: An AnnData object containing gene expression data.

    Returns:
        Adds three QC metrics to `adata.obs`

        - `log_totalRNA`: log10-transformed total RNA count per cell.
        - `pct_mtRNA`: Percentage of mitochondrial RNA per cell.
        - `pct_rRNA`: Percentage of ribosomal RNA per cell.

    """
    var_mito = adata.var_names.str.startswith('MT-')
    var_ribo = adata.var_names.str.startswith(('RPS', 'RPL'))

    adata.obs['log_totalRNA'] = np.log10(np.ravel(adata.X.sum(1)))
    adata.obs['pct_mtRNA'] = adata[:, var_mito].X.sum(1) / adata.X.sum(1) * 100
    adata.obs['pct_rRNA'] = adata[:, var_ribo].X.sum(1) / adata.X.sum(1) * 100

    print("Added the following to `adata.obs`: ['log_totalRNA', 'pct_mtRNA', 'pct_rRNA']")


def qc_cross_rep_correlation(adata, group_key='CMPD_DOSE', plate_key='PLATE_NAME', min_corr=0.99):
    """
    Computes cross-replicate correlation quality control (QC) for each compound.

    Args:
        adata:  AnnData object containing single-cell data
        group_key (str): Column in `adata.obs` to define groups (default is 'CMPD_DOSE')
        plate_key (str): Column in `adata.obs` that defines the plate (default is 'PLATE_NAME')
        min_corr (float): Minimum correlation threshold for passing the QC (default is 0.99)

    Returns:
        Adds two columns to `adata.obs`

        - `cross_rep_correlation`: Stores cross-replicate correlations for each compound.
        - `rep_corr_qc_pass`: Boolean flag indicating whether the compound passes QC.
    """

    if plate_key is None:
        plate_key = 'tmp_plate_key'
        adata.obs['tmp_plate_key'] = ''

    group_plate_key = f'{group_key}_{plate_key}_tmp'
    adata.obs[group_plate_key] = (
        adata.obs[group_key].astype(str) + '_' + adata.obs[plate_key].astype(str)
    )

    write_key1 = 'cross_rep_correlation'
    write_key2 = 'rep_corr_qc_pass'

    adata.obs[write_key1] = np.full(adata.n_obs, np.nan, dtype=np.float32)
    adata.obs[write_key2] = np.full(adata.n_obs, np.nan, dtype=bool)

    # Loop through each unique plate in the dataset
    for plate in adata.obs[plate_key].unique():
        idx = adata.obs[plate_key] == plate
        data = adata[idx]

        df_corr = data.to_df().T.corr()

        df_corr.index = df_corr.columns = data.obs[group_plate_key]
        np.fill_diagonal(df_corr.values, np.nan)

        df_corr_mean = (
            df_corr.groupby(df_corr.index, sort=False)
            .mean()
            .T.groupby(df_corr.index, sort=False)
            .mean()
        )
        # Create a dataframe to store replicate correlations for each compound
        _, n_replicates = np.unique(df_corr.index, return_counts=True)

        rep_keys = [f'rep{i + 1}' for i in range(max(n_replicates))]
        df_corr_rep = pd.DataFrame(index=df_corr_mean.columns, columns=rep_keys)

        # Populate replicate correlation dataframe with the max correlated rep for each cmpd
        for cmpd in df_corr_mean.columns:
            corr_vals = df_corr.loc[cmpd, cmpd].max(1).values
            df_corr_rep.iloc[df_corr_rep.index.get_loc(cmpd), : len(corr_vals)] = corr_vals

        # Compute the standard deviation of the replicate correlations across all compounds
        std = np.nanstd(np.ravel(df_corr_rep.iloc[:, : min(n_replicates)]))

        # Handle the case where there are more than 4 replicates, which applies to DMSO
        for cmpd in df_corr_mean.columns:
            # If >4 replicates, compute the mean of the top-N correlations for better robustness
            df_corr_sub = df_corr.loc[cmpd, cmpd]
            if len(df_corr_sub) > 4:
                n_top = int(len(df_corr_sub) / 2) + 1

                corr_vals = np.nanmean(np.sort(df_corr_sub, axis=0)[::-1][:n_top], axis=0)
                df_corr_rep.iloc[df_corr_rep.index.get_loc(cmpd), : len(corr_vals)] = corr_vals

            # Store correlation values
            corr_vals = df_corr_rep.loc[cmpd].dropna().values
            data.obs.loc[data.obs[group_plate_key] == cmpd, write_key1] = corr_vals.astype(
                np.float32
            )

            # Flag entries as invalid if their correlation is below the threshold
            corr_thresh = max(np.median(corr_vals) - 3 * std, min_corr)
            data.obs.loc[data.obs[group_plate_key] == cmpd, write_key2] = (
                corr_vals > corr_thresh
            ).astype(bool)

        # Update the global adata.obs with the results from the current plate
        adata.obs.loc[idx, write_key1] = data.obs[write_key1]
        adata.obs.loc[idx, write_key2] = data.obs[write_key2]
    del adata.obs[group_plate_key]
    if plate_key == 'tmp_plate_key':
        del adata.obs['tmp_plate_key']

    print(f"Added the following to `adata.obs`: ['{write_key1}', '{write_key2}']")


def deseq2(
    adata,
    pert_name_col,
    other_pert_cols=(),
    condition_cols=(),
    dmso_pert_name='DMSO',
    dask_client=None,
    tempdir_prefix=None,
    **kwargs,
):
    """
    Runs DESeq2 for differential expression signatures across perturbations.

    This function executes the DESeq2 model to compare gene expression changes across different
    perturbations. It supports parallel processing using Dask.

    Args:
        adata (AnnData): An AnnData object containing gene expression count data.
        pert_name_col (str): Column name in `adata.obs` specifying the perturbation (e.g., drug).
        other_pert_cols (tuple of str): Additional columns to consider (e.g., dose).
        condition_cols (tuple of str): Columns specifying experimental conditions.
        dmso_pert_name (str): The name of the negative control perturbation (e.g., DMSO).
        dask_client (dask.distributed.Client): A Dask client for parallel processing.
        tempdir_prefix (str): Prefix for temporary directories used during processing.
        **kwargs: Additional arguments passed to the DESeq2 model function.

    Returns:
        A new AnnData object containing the DESeq2 results

        - `pvalue`, `FDR` and `LFC` (log2 fold change) in `layers` for each gene and perturbation.
        - `obs` with summarized perturbation metadata.
    """

    # Initialize Dask client if not provided
    if dask_client is None:
        from dask.distributed import LocalCluster, Client

        cluster = LocalCluster(n_workers=30, threads_per_worker=1, memory_limit='auto')
        dask_client = Client(cluster)

    # Convert sparse matrix to dense and ensure correct dtype
    data = adata.copy()

    if issparse(data.X):
        data.X = data.X.toarray()

    data.X = data.X.astype(int).astype(np.float32)
    data.obs_names_make_unique()

    # Run DESeq2 model
    df_results = model_all_perts(
        data,
        pert_name_col=pert_name_col,
        other_pert_cols=other_pert_cols,
        condition_cols=condition_cols,
        dmso_pert_name=dmso_pert_name,
        dask_client=dask_client,
        tempdir_prefix=tempdir_prefix,
        **kwargs,
    )
    # Group DESeq2 results by perturbation keys
    keys = [pert_name_col] + list(other_pert_cols)

    df_grouped = adata.obs.groupby(keys)
    df_obs = df_grouped.agg(lambda x: x[0] if len(set(x)) == 1 else np.nan)
    df_obs = df_obs.dropna(how='all', axis=1)

    for i, k in enumerate(keys):
        df_obs[k] = df_obs.index.get_level_values(i)

    df_obs = df_obs[~df_obs[pert_name_col].str.startswith(dmso_pert_name)].copy()

    # Create crosstabs for each DESeq2 metric
    index = [df_results[k] for k in keys]
    crosstab_args = {
        'index': index,
        'columns': df_results.gene,
        'aggfunc': np.mean,
        'dropna': False,
    }

    PVL = pd.crosstab(values=df_results.pvalue, **crosstab_args)
    FDR = pd.crosstab(values=df_results.padj, **crosstab_args)
    LFC = pd.crosstab(values=df_results.log2FoldChange, **crosstab_args)

    # Initialize a new AnnData to collect the DESeq2 results
    adata_deseq = ad.AnnData(X=None, obs=df_obs, var=adata.var)
    idx_var = adata_deseq.var_names

    idx_obs = adata_deseq.obs[keys].apply(tuple, axis=1)
    idx_obs = idx_obs[idx_obs.index.isin(PVL.index)]
    adata_deseq = adata_deseq[adata_deseq.obs_names.isin(idx_obs)].copy()

    adata_deseq.layers['pvalue'] = PVL.loc[idx_obs, idx_var]
    adata_deseq.layers['FDR'] = FDR.loc[idx_obs, idx_var]
    adata_deseq.layers['LFC'] = LFC.loc[idx_obs, idx_var]

    adata_deseq.obs_names = adata_deseq.obs_names.get_level_values(0).astype(str).values

    return adata_deseq


def pathway_signatures(df, pval_thresh=0.05, gene_sets='WikiPathways_2019_Human'):
    """Perform pathway enrichment analysis on DEGs for each compound.

    This function takes a DataFrame of p-values, identifies significant genes based on a p-value
    threshold, and runs pathway enrichment analysis using Enrichr via GSEApy.
    It supports both single-sample and multi-sample differential expression data, returning results
    as an AnnData object with pathway enrichment statistics.

    Args:
        df (pd.DataFrame or pd.Series): Input data containing p-values for each gene. Columns should
                                        represent genes, and rows observations (e.g., perturbations)
        pval_thresh (float): P-value threshold for selecting significant genes.
        gene_sets (str): The gene set library to use for pathway enrichment.

    Returns:
        An AnnData object containing pathway enrichment results, indexed by the
        observations in `df`, with `layers` for

        - `DES`: -log10(FDR) enrichment scores.
        - `pval`: Raw p-values for each pathway.
        - `FDR`: Adjusted p-values (FDR).
        - `combined_score`: Enrichr's combined score.
    """
    try:
        import gseapy as gp
    except ImportError as e:
        raise ImportError(
            'Please install gseapy first via `conda install -c bioconda gseapy`.'
        ) from e

    gseapy_kwargs = {'gene_sets': gene_sets, 'organism': 'human'}

    if len(df) == 1:
        gene_list = list(df.columns[np.ravel(df) < 0.05])
        return gp.enrichr(gene_list=gene_list, **gseapy_kwargs).results
    elif len(df.shape) == 1:
        gene_list = list(df.index[np.ravel(df) < 0.05])
        return gp.enrichr(gene_list=gene_list, **gseapy_kwargs).results

    else:
        import time

        res = pd.DataFrame()

        for i in range(len(df)):
            gene_list = list(df.columns[df.iloc[i] < pval_thresh])
            if len(gene_list) > 0:
                attempts = 0
                while attempts < 5:  # Setting a maximum of 5 retry attempts
                    try:
                        enr_res = gp.enrichr(gene_list=gene_list, **gseapy_kwargs).results
                        if len(enr_res) > 0:
                            enr_res['obs_index'] = i
                            res = pd.concat([res, enr_res])
                        break  # Break out of the while loop if the operation is successful
                    except Exception as e:
                        attempts += 1
                        print(f'Attempt {attempts} failed with error: {e}')
                        time.sleep(1)  # Optional: wait for 1 second before retrying
                        if attempts >= 5:
                            print(f'Failed after {attempts} attempts for index {i}.')
        df_crosstabs = {}

        kwargs = {'index': res['obs_index'], 'columns': res['Term'], 'aggfunc': 'first'}

        df_crosstabs['pval'] = pd.crosstab(values=res['P-value'], **kwargs)
        df_crosstabs['FDR'] = pd.crosstab(values=res['Adjusted P-value'], **kwargs)
        df_crosstabs['DES'] = -np.log10(df_crosstabs['FDR'])

        df_crosstabs['combined_score'] = pd.crosstab(values=res['Combined Score'], **kwargs)
        df_crosstabs['overlap'] = pd.crosstab(values=res['Overlap'], **kwargs)
        df_crosstabs['genes'] = pd.crosstab(values=res['Genes'], **kwargs)

        missing_idx = [i for i in range(len(df)) if i not in df_crosstabs['pval'].index]

        for k in df_crosstabs.keys():
            df_crosstabs[k] = pd.concat(
                [df_crosstabs[k], pd.DataFrame(index=missing_idx)]
            ).sort_index()

        data = ad.AnnData(df_crosstabs['DES'].set_index(df.index))

        for k in df_crosstabs.keys():
            if k not in ['overlap', 'genes']:
                data.layers[k] = df_crosstabs[k]

        return data
