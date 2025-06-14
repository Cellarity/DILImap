import numpy as np
import pandas as pd
from natsort import natsorted
import anndata as ad


def platemap(data, value_key, batch=None):
    """
    Generates a plate map for visualization.

    Args:
        data (AnnData): Input data containing observations with plate well information.
        value_key (str): Key in `obs` specifying the values to aggregate (e.g., compound_name).
        batch (str, optional): Column in `obs` specifying batch labels for stratifying the data.

    Returns:
        A DataFrame where each row corresponds to a well row, each column to a well
        column, and values are aggregated based on the specified `value_key`.
    """

    batches = [''] if batch is None else np.unique(data.obs[batch])

    df_batches = pd.DataFrame()
    for batch_i in batches:
        adata_sub = data if batch_i == '' else data[data.obs[batch] == batch_i]

        aggfunc = (
            ','.join if all(isinstance(item, str) for item in adata_sub.obs[value_key]) else 'mean'
        )
        index, columns = adata_sub.obs['WELL_ROW'], adata_sub.obs['WELL_COL']
        df_batch = pd.crosstab(index, columns, values=adata_sub.obs[value_key], aggfunc=aggfunc)
        df_batch = df_batch[natsorted(pd.unique(adata_sub.obs['WELL_COL']))]
        df_batch.index.name = None
        df_batch.columns.name = None

        df_batch.index = ('' if batch_i == '' else str(batch_i) + '_') + df_batch.index.astype(str)
        df_batches = pd.concat([df_batches, df_batch])

    return df_batches


def groupby(data, key, aggfunc='mean'):
    """
    Groups a dataset by a specified key, preserving one-to-one categorical mappings.

    - Columns with a single unique value per group are retained as-is.
    - Columns with multiple unique values are aggregated using `aggfunc`.
    - Non-numeric categorical columns are retained only if they have a one-to-one mapping with `key`.

    Args:
        data (pandas.DataFrame or anndata.AnnData): Input dataset. If AnnData, uses `obs`.
        key (str): The column name to group the data by.
        aggfunc (str or function): Aggregation function to apply to numerical columns.

    Returns:
        pandas.DataFrame: Grouped data.
    """

    df = data.obs.copy() if isinstance(data, ad.AnnData) else data.copy()

    # Identify categorical columns
    all_cols = df.columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Compute unique counts per group
    nunique_per_group = df.groupby(key, observed=False).nunique()

    # Identify columns with a one-to-one mapping to `key`
    one_to_one_cols = [
        k for k in all_cols if k in nunique_per_group and nunique_per_group[k].max() == 1
    ]

    # Separate numerical columns that need aggregation
    multi_val_cols = [
        k for k in numerical_cols if k in nunique_per_group and nunique_per_group[k].max() > 1
    ]

    # Group categorical one-to-one columns using `first()`
    grouped_cat_df = df.groupby(key, observed=False)[one_to_one_cols].first()

    # Aggregate numerical columns using `aggfunc`
    grouped_num_df = df.groupby(key, observed=False)[multi_val_cols].agg(aggfunc)

    # Rename aggregated columns with function name
    aggfunc_name = aggfunc if isinstance(aggfunc, str) else aggfunc.__name__
    grouped_num_df.columns = [f'{col}_{aggfunc_name}' for col in grouped_num_df.columns]

    # Concatenate results
    return pd.concat([grouped_cat_df, grouped_num_df], axis=1)


def crosstab(data, keys, aggfunc=None):
    """
    Creates a crosstab for the specified keys, applying an optional aggregation function.

    Args:
        data (pandas.DataFrame or anndata.AnnData): Input dataset. If AnnData, uses `obs`.
        keys (list of str): Column names for rows, columns and values for pandas.crosstab.
        aggfunc (str or function): Aggregation function for values.

    Returns:
        DataFrame with crosstab of the input data with the specified aggregation.
    """

    df = data.obs if isinstance(data, ad.AnnData) else data
    if len(keys) == 2:
        return pd.crosstab(df[keys[0]], df[keys[1]])

    if aggfunc is None:
        aggfunc = 'first' if isinstance(df[keys[-1]].iloc[0], str) else 'mean'

    return pd.crosstab(df[keys[0]], df[keys[1]], df[keys[2]], aggfunc=aggfunc)


def map_dili_labels_and_cmax(data, obs_key=None, labels=None, insert_at_front=None):
    """
    Annotate an AnnData or pandas.DataFrame with selected DILI labels and Cmax values.

    Parameters
    ----------
    data : anndata.AnnData or pandas.DataFrame
        The data to annotate. Uses obs_names (AnnData) or index/column (DataFrame)
        to map compound names.

    obs_key : str, optional
        If provided, use data.obs[obs_key] or data[obs_key] to map compound names instead of index.

    labels : list of str, optional
        List of columns to map. Options include:
        - 'DILI_label', 'DILIrank' (from DILI dataset)
        - 'Cmax_uM', 'free_Cmax_uM' (from Cmax dataset)
        - 'LDH_IC10_uM' (from cytotoxicity dataset)
        If None, all available columns will be added.

    Returns
    -------
    None
        Modifies `data.obs` (AnnData) or `data` (DataFrame) in-place.
    """
    from . import datasets

    df_DILI = datasets.compound_DILI_labels()
    df_DILI.index = df_DILI.index.astype(str).str.lower()

    # Auto-detect all possible labels if none are specified
    if labels is None:
        labels = set(df_DILI.columns)
        load_cmax = True
        load_ldh = True
        labels.update(['Cmax_uM', 'free_Cmax_uM', 'LDH_IC10_uM'])
    else:
        labels = set(labels)
        load_cmax = any(label in labels for label in ['Cmax_uM', 'free_Cmax_uM'])
        load_ldh = 'LDH_IC10_uM' in labels

    if load_cmax:
        df_CMAX = datasets.compound_Cmax_values()
        df_CMAX.index = df_CMAX.index.astype(str).str.lower()

    if load_ldh:
        df_LDH = datasets.compound_cell_viability()
        df_LDH.index = df_LDH.index.astype(str).str.lower()

    # Determine data frame to annotate
    if isinstance(data, ad.AnnData):
        df = data.obs
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError('Input must be an AnnData or pandas DataFrame')

    # Extract compound names
    obs_names = df.index if obs_key is None else df[obs_key]
    obs_names = obs_names.astype(str).str.lower()

    # Prepare new columns
    new_cols = {}

    # Map DILI labels
    for col in df_DILI.columns:
        if col in labels:
            new_cols[col] = obs_names.map(df_DILI[col])

    # Map Cmax
    if 'Cmax_uM' in labels:
        new_cols['Cmax_uM'] = obs_names.map(df_CMAX['Cmax_median'])

    if 'free_Cmax_uM' in labels:
        new_cols['free_Cmax_uM'] = obs_names.map(df_CMAX['free_Cmax_median'])

    # Map viability data
    if 'LDH_IC10_uM' in labels:
        new_cols['LDH_IC10_uM'] = obs_names.map(df_LDH['IC10_uM'])

    # Insert columns in-place, overwrite if exists
    for i, (col, values) in enumerate(new_cols.items()):
        if insert_at_front:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
            df.insert(loc=i, column=col, value=values)
        else:
            df[col] = values


def map_dili_labels_and_cmax_(adata, obs_key=None):
    """
    Annotate an AnnData object with DILI labels and Cmax values.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to annotate. Assumes `obs_names` or a column in `adata.obs` contains compound names.

    obs_key : str, optional
        If provided, use `adata.obs[obs_key]` instead of `adata.obs_names` to map compound names.

    Returns
    -------
    None
        The function modifies `adata.obs` in-place by adding the following columns:
        - DILI-related labels (e.g., 'DILI_label', 'DILI_confidence') from `compound_DILI_labels.csv`
        - 'Cmax_uM': total Cmax values from `compound_Cmax_values.csv`
        - 'free_Cmax_uM': free Cmax values from `compound_Cmax_values.csv`
    """
    from . import datasets

    # Map DILI annotations
    df_DILI = datasets.compound_DILI_labels()
    df_DILI.index = df_DILI.index.str.lower()

    obs_names = adata.obs_names.copy() if obs_key is None else adata.obs[obs_key].copy()
    obs_names = obs_names.str.lower()

    for col in df_DILI.columns:
        adata.obs[col] = obs_names.map(df_DILI[col])

    # Map Cmax values
    df_CMAX = datasets.compound_Cmax_values()
    df_CMAX.index = df_CMAX.index.str.lower()
    adata.obs['Cmax_uM'] = obs_names.map(df_CMAX['Cmax_median'])
    adata.obs['free_Cmax_uM'] = obs_names.map(df_CMAX['free_Cmax_median'])

    # Map IC10s from LDH cell viability screen
    df_LDH = datasets.compound_cell_viability()
    df_LDH.index = df_LDH.index.str.lower()
    adata.obs['LDH_IC10_uM'] = obs_names.map(df_LDH['IC10_uM'])
