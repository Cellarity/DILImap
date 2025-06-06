import pandas as pd
import numpy as np
import anndata as ad
from unittest.mock import patch, MagicMock
from dilimap.models import ToxPredictor


@patch('dilimap.s3.read')  # patch the S3 read function
def test_toxpredictor_predict(mock_read):
    # Create a mock model with minimal behavior
    features = pd.Index(
        [
            'Amino Acid metabolism WP3925',
            'Ferroptosis WP4313',
            'Nuclear Receptors Meta-Pathway WP2882',
        ]
    )

    mock_model = MagicMock(spec=ToxPredictor)
    mock_model.features = features
    mock_model.estimators = [MagicMock()]
    mock_model.estimators[0].predict_proba.return_value = np.array([[0.383354, 0.383354]]).T
    mock_model.predict_proba.return_value = pd.DataFrame(
        np.array([[0.383354, 0.383354]]).T, columns=['DILI_probability'], index=['drug1', 'drug2']
    )

    # Patch read() to return the mock model
    mock_read.return_value = mock_model

    # Create dummy AnnData input
    adata = ad.AnnData(
        X=np.array([[1, 0, 1], [2, 1, 2]]),
        obs=pd.DataFrame(index=['drug1', 'drug2']),
        var=pd.DataFrame(index=features),
    )

    # Instantiate ToxPredictor (will trigger mock_read)
    model = ToxPredictor()
    # model.cv_models['estimator'] = mock_model.estimators  # plug in mocked estimator

    # Run prediction
    result = model.predict(adata)

    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert 'DILI_probability' in result.columns
    assert np.isclose(result.iloc[0, 0], 0.383354)
