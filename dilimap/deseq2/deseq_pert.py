from pathlib import Path
import os
import subprocess
import warnings
import dask
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse
from tempfile import TemporaryDirectory


def model_all_perts(
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
    Apply `model_pert` to all perturbations present in an AnnData, and concatenate dataframse

    Args:
        adata (AnnData): Input Anndata with transcriptional data and metadata in .obs
        pert_name_col (str): Column name of obs indicating perturbation name.
        other_pert_cols: (list[str]): Column names indicating perturbation conditions (e.g. dose)
        condition_cols (list[str]): Column names indicating technical conditions to be regressed out
        dmso_pert_name (str): Perturbation name corresponding to DMSO, default `'DMSO'`.
        dask_client (distributed.client.Client): A Dask client connecting to a local/remote cluster.

    Returns
        Dataframe with modeling results and perturbational covariates for all perturbations
    """
    # Initialize Dask client if not provided
    if dask_client is None:
        from dask.distributed import LocalCluster, Client

        cluster = LocalCluster(n_workers=30, threads_per_worker=1, memory_limit='auto')
        dask_client = Client(cluster)

    pert_cols = [pert_name_col, *other_pert_cols]

    # Set up temporary directory
    if tempdir_prefix is None:
        tempdir_prefix = os.path.expanduser('~') + '/.tmp/'
        os.makedirs(tempdir_prefix, exist_ok=True)

    # Extract all unique perturbations excluding DMSO
    perts = [
        x
        for x in {tuple(x[1:]) for x in adata.obs.loc[:, pert_cols].itertuples()}
        if x[0] != dmso_pert_name
    ]

    @dask.delayed
    def model_pert_parallel(
        adata,
        pert_name_modeled,
        pert_conditions_modeled,
        pert_name_col,
        other_pert_cols,
        **kwargs,
    ):
        # Use a context manager to ensure temp directory cleanup
        with TemporaryDirectory(prefix=tempdir_prefix) as tempdir:
            kwargs['data_dir'] = tempdir
            res = model_pert(
                adata,
                pert_name_modeled=pert_name_modeled,
                pert_conditions_modeled=pert_conditions_modeled,
                pert_name_col=pert_name_col,
                other_pert_cols=other_pert_cols,
                **kwargs,
            )
            res[pert_name_col] = pert_name_modeled

            # add perturbation info columns
            for i, col in enumerate(other_pert_cols):
                res[col] = pert_conditions_modeled[i]

        return res

    # Create parallel computation tasks
    outputs = [
        model_pert_parallel(
            adata,
            pert[0],
            list(pert[1:]),
            pert_name_col,
            other_pert_cols,
            dmso_pert_name=dmso_pert_name,
            condition_cols=condition_cols,
            **kwargs,
        )
        for pert in perts
    ]

    # Execute the tasks in parallel
    futures = dask_client.compute(outputs, sync=False)
    comp_results = [x.result() for x in futures]

    return pd.concat(comp_results)


def model_pert(
    adata,
    pert_name_modeled,
    pert_conditions_modeled,
    pert_name_col,
    other_pert_cols,
    condition_cols=(),
    dmso_pert_name='DMSO',
    **kwargs,
):
    X = _make_design_matrix(
        adata.obs,
        pert_name_modeled=pert_name_modeled,
        pert_conditions_modeled=pert_conditions_modeled,
        pert_name_col=pert_name_col,
        other_pert_cols=other_pert_cols,
        condition_cols=condition_cols,
        dmso_pert_name=dmso_pert_name,
    )
    Y = adata[X.index, :].X
    modelres = _deseq_model(X, Y, **kwargs)

    modelres['gene'] = [adata.var_names[idx] for idx in modelres.index]

    return modelres


def _deseq_model(
    X,
    Y,
    data_dir=None,
    is_treat_col='is_treat',
    thresh=1e-5,
    use_adj=True,
    **kwargs,
):
    """

    Parameters
    ----------
    X: pandas.DataFrame, shape (n_samples x n_covariates)
        Design matrix with column indicating treatment vs. control and other covariates
    Y: arraylike, shape (n_samples x n_genes)
        Gene expression observed for those covariates
    data_dir: str or None
        Directory where DESeq stores temporary files. If None, will make a temporary directory under ~/.tmp/
    is_treat_col: str
        Column name of X with a binary/boolean indicator whether the well is treatment or control.
    thresh: float
        p-value threshold for calling a gene differentially-expressed.
    use_adj: bool
        if True (default), use shrinkage-adjusted p-values to call DE genes.
    kwargs:
        Other arguments passed to `DeSeq2`, e.g. exec_path

    Returns
    -------
    A pandas dataframe with n_genes rows and columns for p-value, adjusted p-value, logFC, and differential expr
    calls for each gene.
    """
    adata = ad.AnnData(Y)
    adata.var_names = np.arange(Y.shape[1]).astype('str')  # preserve original idxs

    adata.obs = X
    adata.obs[is_treat_col] = adata.obs[is_treat_col].astype('str')

    assert is_treat_col in X.columns, 'X has no column {}'.format(is_treat_col)

    design = [x for x in X.columns if x != is_treat_col]
    design.append(is_treat_col)

    if data_dir is None:  # make a temporary directory in a context manager and destroy it after.
        prefix = os.path.expanduser('~') + '/.tmp/'
        os.makedirs(prefix, exist_ok=True)

        with TemporaryDirectory(prefix=prefix) as tempdirname:
            data_dir = tempdirname
            deres = _run_deseq2(adata, design, data_dir, pert_col=is_treat_col, **kwargs)
    else:
        deres = _run_deseq2(adata, design, data_dir, pert_col=is_treat_col, **kwargs)

    result = deres.var
    result.index = result.index.astype('int')  # back to int
    if use_adj:
        result['is_de'] = result['padj'] < thresh
    else:
        result['is_de'] = result['pvalue'] < thresh

    return result


def _run_deseq2(
    adata: ad.AnnData,
    design: list = ('plate', 'treatment'),
    data_dir: str = '/home/jovyan/tmp/deseq',
    input_name: str = 'input.h5ad',
    output_name: str = 'output.h5ad',
    pert_col: str = None,
    pert_name: str = 'True',
    control_name: str = 'False',
) -> ad.AnnData:
    """
    Contrast two conditions using DESeq
    ----------
    adata: anndata.AnnData
        AnnData object with raw expression data in .X and relevant design parameters as .obs columns
    design: list
        List of terms in the formula. Formula is '~'+'+'.join(design)
    data_dir:
        Directory where DESeq will store intermediate values. Can be a temporary directory.
    input_name: str
        Name of the file where adata will be stored on disk to be read by DESeq (no need to change this)
    output_name: str
        Name of the file where the result of DESeq will be stored. No need to modify.
    pert_col: str
        Column of adata.obs indicating perturbation (or level to be contrasted).
    pert_name: str
        Perturbation to be modeled
    control_name: str
        Level of adata.obs[pert_col] to use as background signal , e.g. 'DMSO'
    Returns
    -------
    anndata.AnnData
        anndata object with gene expression information in adata.var.
    """
    exec_path = Path(__file__).parent / 'deseq_exec'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    input_path = os.path.join(data_dir, input_name)
    output_path = os.path.join(data_dir, output_name)

    # coerce sparse matrix to dense for DESeq input
    if issparse(adata.X):
        adata.X = adata.X.todense()

    # Write to disk
    adata.write_h5ad(input_path)

    # Run component
    cmd = [
        'bash',
        str(exec_path),
        '--input',
        input_path,
        '--output',
        output_path,
        '--design',
        '{}'.format(' + '.join(design)),
    ]
    if pert_col is not None:
        cmd += ['--pert_col', pert_col]
    if pert_name is not None:
        cmd += ['--pert_name', pert_name]
    if control_name is not None:
        cmd += ['--control_name', control_name]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception as e:
        raise RuntimeError(
            'Error in DESeq \n Error message: {} \n \n stdout: {}'.format(e.stderr, e.stdout)
        ) from e
    # Get output and return
    return ad.read_h5ad(output_path)


def _make_design_matrix(
    obs,
    pert_name_modeled,
    pert_conditions_modeled,
    pert_name_col,
    other_pert_cols=(),
    condition_cols=(),
    dmso_pert_name='DMSO',
):
    """
    From an IL experiment, set up a design matrix for DE gene modeling

    Parameters
    ----------
    obs: pandas.DataFrame
        Metadata used to make design matrix
    pert_name_modeled: str
        Perturbation whose effect is being modeled.
    pert_conditions_modeled: list[str]
        List of Perturbation conditions (dose, time, etc.) whose effect is being modeled,
        corresponding to other_pert_cols
    pert_name_col: str
        Column name of obs indicating perturbation name.
    other_pert_cols: list[str]
        Column names of obs indicating perturbation conditions to be modeled separately (e.g. dose, time)
    condition_cols: list[str]
        Column names of obs indicating technical conditions to be regressed out
    dmso_pert_name: str
        Perturbation name corresponding to DMSO, default `'DMSO'`.
    Returns
    -------
    pandas.DataFrame
        Design matrix whose rows are those of the chosen perturbation, and paired DMSOs.
        Has an 'is_treat' column indicating whether the well is DMSO or the chosen treatment,
        and additional columns for each technical covariate to be regressed out. Indices
        are subsets of obs.index.
    """

    pert_cols = [pert_name_col, *other_pert_cols]
    pert_names = [pert_name_modeled, *pert_conditions_modeled]
    df = obs.loc[:, np.unique([*pert_cols, *condition_cols])]

    pert_vals = list(zip(*[df[col].values.astype('str') for col in pert_cols]))
    pert_rows = np.where([np.all(np.array(pert_names) == np.array(x)) for x in pert_vals])[0]

    if len(pert_rows) == 0:
        warnings.warn(
            'No perturbations with name {} and conditions:\n {}'.format(
                pert_name_modeled,
                '\n'.join(
                    [
                        str(x) + ':' + str(y)
                        for x, y in zip(other_pert_cols, pert_conditions_modeled)
                    ]
                ),
            ),
            stacklevel=1,
        )

    condition_vals = list(zip(*[df[col].values.astype('str') for col in condition_cols]))

    kept_rows = pert_rows.tolist()

    for row in pert_rows:
        if len(condition_vals) > 0:
            # find and add DMSO rows with conditions matching the treatment rows
            condition = condition_vals[row]

            # find wells with same conditions
            same_condition = [
                np.all(np.array(condition_vals[i]) == np.array(condition))
                for i in range(len(condition_vals))
            ]

        else:  # no additional conditions to meet
            same_condition = True

        is_dmso = [x[0] == dmso_pert_name for x in pert_vals]

        new_rows = np.where(np.logical_and(same_condition, is_dmso))[0]
        kept_rows += new_rows.tolist()

    kept_rows = np.unique(kept_rows)

    if len(kept_rows) == 0:
        raise ValueError(
            'No rows matching conditions; check that pert/experimental conditions are properly set'
        )

    X = df.iloc[kept_rows, :].loc[:, condition_cols]

    X['is_treat'] = [x[0] != dmso_pert_name for x in np.array(pert_vals)[kept_rows]]

    if np.sum(X['is_treat'].values) == X.shape[0]:
        warnings.warn('No control wells present; modeling may not be possible', stacklevel=1)

    # remove constant covariates
    to_remove = X.columns[X.nunique() == 1]
    for col in to_remove:
        del X[col]

    return X
