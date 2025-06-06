from .s3 import read
import warnings
import anndata as ad


def compound_Cmax_values():
    """Plasma Cmax annotations data for compounds in DILImap (csv file)"""
    return read('compound_Cmax_values.csv', package_name='public/data')


def compound_DILI_labels():
    """DILI annotations from DILIrank and LiverTox databases (csv file)"""
    return read('compound_DILI_labels.csv', package_name='public/data')


def compound_cell_viability(level='raw'):
    """
    Cell viability assay results for all compounds in DILImap (csv file)

    Args:
    level (str): Specifies the data processing level. Options are

        - `raw`: Raw assay results
        - `IC10`: IC10 estimates

    """
    if level not in ['raw', 'IC10']:
        raise ValueError(f"must be one of ['raw', 'IC10'], got {level}")

    return read(f'compound_cell_viability_{level}.csv', package_name='public/data')


def demo_data(level='counts'):
    """
    Example gene expression data (anndata file)

    Args:
    level (str): Specifies the data processing level. Options are

        - `counts`: Raw count data
        - `deseq2`: DESeq2 normalized data
        - `pathways`: WikiPathways-level signatures

    """
    if level not in ['counts', 'deseq2', 'pathways']:
        raise ValueError(f"must be one of ['counts', 'deseq2', 'pathways'], got {level}")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        return read(f'demo_data_{level}.h5ad', package_name='public/data')


def DILImap_training_data(level='pathways'):
    """
    DILImap gene expression training data (anndata file)

    Args:
    level (str): Specifies the data processing level. Options are

        - `counts`: Raw count data
        - `deseq2`: DESeq2 normalized data
        - `pathways`: WikiPathways-level signatures

    """
    if level not in ['counts', 'deseq2', 'pathways']:
        raise ValueError(f"must be one of ['counts', 'deseq2', 'pathways'], got {level}")
    name = 'public/data' if level == 'pathways' else 'proprietary/data'
    return read(f'training_data_{level}.h5ad', package_name=name)


def DILImap_validation_data(level='pathways'):
    """
    DILImap gene expression validation data (anndata file)

    Args:
    level (str): Specifies the data processing level. Options are

        - `counts`: Raw count data
        - `deseq2`: DESeq2 normalized data
        - `pathways`: WikiPathways-level signatures

    """
    if level not in ['counts', 'deseq2', 'pathways']:
        raise ValueError(f"must be one of ['counts', 'deseq2', 'pathways'], got {level}")
    return read(f'validation_data_{level}.h5ad', package_name='public/data')


def DILImap_data(level='pathways'):
    """
    DILImap gene expression data (anndata file)

    Args:
    level (str): Specifies the data processing level. Options are

        - `counts`: Raw count data
        - `deseq2`: DESeq2 normalized data
        - `pathways`: WikiPathways-level signatures

    """
    if level not in ['counts', 'deseq2', 'pathways']:
        raise ValueError(f"must be one of ['counts', 'deseq2', 'pathways'], got {level}")

    adatas = {
        'training': read(f'training_data_{level}.h5ad', package_name='public/data'),
        'validation': read(f'validation_data_{level}.h5ad', package_name='public/data'),
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        return ad.concat(adatas, label='dataset', index_unique=None)
