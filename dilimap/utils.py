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
