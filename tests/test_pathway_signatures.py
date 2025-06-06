import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import dilimap


# Dummy result to mock enrichr response
def mock_enrichr_result(index=0):
    return pd.DataFrame(
        {
            'Term': ['Pathway A', 'Pathway B'],
            'P-value': [0.01, 0.02],
            'Adjusted P-value': [0.015, 0.025],
            'Combined Score': [10.0, 8.5],
            'Overlap': ['5/100', '4/80'],
            'Genes': ['GENE1;GENE2', 'GENE3;GENE4'],
            'obs_index': index,
        }
    )


@patch('dilimap.preprocessing.gp.enrichr')
def test_single_sample_dataframe(mock_enrichr):
    mock_enrichr.return_value.results = mock_enrichr_result()

    df = pd.DataFrame([[0.01, 0.10, 0.03]], columns=['TP53', 'TNF', 'EGFR'])

    result = dilimap.pp.pathway_signatures(df)

    assert isinstance(result, pd.DataFrame)
    assert 'Term' in result.columns
    assert 'Adjusted P-value' in result.columns


@patch('dilimap.preprocessing.gp.enrichr')
def test_multi_sample_dataframe(mock_enrichr):
    # Mock returns slightly different result per call
    def enrichr_side_effect(*args, **kwargs):
        idx = enrichr_side_effect.call_count
        enrichr_side_effect.call_count += 1
        mock = MagicMock()
        mock.results = mock_enrichr_result(index=idx)
        return mock

    enrichr_side_effect.call_count = 0

    mock_enrichr.side_effect = enrichr_side_effect

    df = pd.DataFrame(
        [
            [0.01, 0.20, 0.03],
            [0.06, 0.02, 0.04],
        ],
        index=['0', '1'],
        columns=['TP53', 'TNF', 'EGFR'],
    )

    adata = dilimap.pp.pathway_signatures(df)

    assert adata.n_obs == 2
    assert 'FDR' in adata.layers
    assert 'DES' in adata.layers
    assert 'combined_score' in adata.layers
    assert np.isfinite(adata.layers['FDR']).all()
