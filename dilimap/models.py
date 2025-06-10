# Standard library imports
import joblib
from warnings import warn
import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin

# Local imports
from .s3 import read, write
from .utils import groupby, crosstab

# Dictionary of all available classifiers in scikit-learn
CLF = {
    name: estimator
    for name, estimator in all_estimators()
    if issubclass(estimator, ClassifierMixin)
}


class ToxPredictor:
    """
    ToxPredictor pre-trained v1 or initialize new model for re-training.

    It supports multi-output classification and custom hyperparameters for model training.

    Args:
        version (str): Model version; for TGP use 'v0.1', for DILImap use 'v1'.
        model (str): The model type to be used, for instance 'RandomForestClassifier'.
        multioutput (bool): Whether to use a multi-output classifier for the model.
        model_params (dict): Dictionary of parameters to pass to the model.

    Usage:
    ------
    .. code:: python

        from dilimap.models import ToxPredictor

        model = ToxPredictor('v1')
        results = model.predict(data)
        safety_margins = model.compute_safety_margin(data, 'compound_name', 'dose_uM', 'Cmax_uM')

    """

    def __init__(
        self,
        version='v1',
        model=None,
        multioutput=None,
        model_params=None,
        local_filepath=None,
    ):
        """Initializes the ToxPredictor model."""
        # Initialize general classifier attributes
        self.model_name = model
        self.multioutput = multioutput
        self.model_params = model_params or {}
        self.cv_models = None

        # Initialize ToxPredictor attributes
        self.version = version
        self.DILI_pred_cutoff = None
        self.safety_margin_results = None
        self.DILImap_margins = None

        if local_filepath:
            self._load_from_local(local_filepath)
        elif model:
            self._initialize_model()
        elif version:
            self._load_from_s3()
        else:
            raise ValueError(
                'Please provide one of the following: version, model or local_filepath'
            )

    def _load_from_s3(self):
        """Loads the pre-trained model from S3."""
        try:
            model = read(f'ToxPredictor_{self.version}.joblib', package_name='public/models')
            self._validate_and_update(model)
        except (FileNotFoundError, RuntimeError) as e:
            raise RuntimeError('Failed to load pretrained model.') from e

    def _load_from_local(self, filepath):
        """Loads the pre-trained model from a local file."""
        if not filepath:
            raise ValueError('No local filepath provided.')
        model = joblib.load(filepath)
        self._validate_and_update(model)

    def _validate_and_update(self, model):
        """Validates and updates the current instance with the loaded model."""
        if not isinstance(model, ToxPredictor):
            raise ValueError('The file does not contain a ToxPredictor object.')
        self.__dict__.update(model.__dict__)

    def _initialize_model(self):
        """Initialize the model based on the provided parameters."""
        # Set default model if not provided
        model_map = {
            'rf': 'RandomForestClassifier',
            'logreg': 'LogisticRegression',
            'svc': 'SVC',
            'mlp': 'MLPClassifier',
            'gb': 'GradientBoostingClassifier',
        }

        if self.model_name in model_map:
            self.model_name = model_map[self.model_name]

        # Validate model
        if self.model_name not in CLF:
            raise ValueError(
                f'Invalid model name. Please select from the following: {list(CLF.keys())}.'
            )

        # Set default hyperparameters if none are provided

        if self.model_name == 'RandomForestClassifier' and not self.model_params:
            self.model_params = {
                'n_estimators': 100,
                'max_depth': 2,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
            }

        # Initialize the model with the specified parameters
        self.model = CLF[self.model_name](**self.model_params)

        # Wrap the model in MultiOutputClassifier if multi-output is specified
        if self.multioutput:
            self.model = MultiOutputClassifier(self.model, n_jobs=-1)

        # Initialize additional attributes
        self.cv_models = None

    def _prepare_data(self, X, y):
        """Prepares the input data by converting it to a DataFrame and handling missing values."""
        self._X = pd.DataFrame(X)
        self._y = y

        # Store observation and feature names
        self.obs_names = self._X.index
        self.features = self._X.columns

        # Fill NaN values
        if self._X.isna().any().any():
            self._X.fillna(0, inplace=True)
            warn(
                'Your data contains missing values (NaNs), which have been filled with zeros.',
                stacklevel=2,
            )

    @property
    def estimators(self):
        """Returns the trained estimators from cross-validation."""
        if self.cv_models is None:
            raise ValueError('Training or cross-validation has not been performed yet.')
        return self.cv_models['estimator']

    @property
    def indices(self):
        """Returns the training/test indices from cross-validation."""
        if self.cv_models is None:
            raise ValueError('Training or cross-validation has not been performed yet.')
        return self.cv_models['indices']

    @property
    def classes_(self):
        """Returns the classes of the classifier estimators."""
        return self.estimators[0].classes_

    @property
    def multioutput_estimators(self):
        """Returns the estimators for each output dimension in a multi-output model."""
        if isinstance(self.estimators[0], MultiOutputClassifier):
            # For multi-output models, split estimators by output dimension
            n_est = len(self.estimators[0].estimators_)
            return [[est.estimators_[i] for est in self.estimators] for i in range(n_est)]
        else:
            return self.estimators

    def cross_validate(self, X, y, n_splits=5, groups=None, scoring=None):
        """
        Performs cross-validation on the model using GroupKFold.

        Parameters:
        ----------
        X : array-like
            Feature matrix for training.
        y : array-like
            Target vector for training.
        n_splits : int, optional (default=5)
            Number of splits for GroupKFold.
        groups : array-like, optional
            Group labels for the samples used while splitting the dataset into training/test sets.
        """
        # Prepare the data
        self._prepare_data(X, y)

        # Set random seed for reproducibility
        np.random.seed(0)

        # Initialize GroupKFold with specified number of splits
        group_kfold = GroupKFold(n_splits=n_splits)

        # Perform cross-validation and store the models and indices
        self.cv_models = cross_validate(
            self.model,
            self._X,
            self._y,
            cv=group_kfold.split(self._X, self._y, groups),
            scoring=scoring,
            return_estimator=True,
            return_indices=True,
        )

    def predict_proba_cv(self, estimators=None, rescale=True):
        """
        Predicts class probabilities for cross-validated models.

        Parameters:
        ----------
        estimators : list or dict, optional (default=None)
            A list of trained estimators from cross-validation. If None, it will use the estimators
            from the cross-validation process stored in `self.estimators`.

        rescale : bool, optional (default=True)
            Whether to rescale the predicted probabilities.
            Rescaling is performed based on the predictions when all features are zeroed out.

        Returns:
        -------
        y_pred : pandas.DataFrame
            A DataFrame containing the predicted probabilities for each observation and class.
        """
        # Use the stored cross-validation estimators if none are provided
        if estimators is None:
            estimators = self.estimators

        # If the estimators are provided as a dictionary, extract the 'estimator' key
        if isinstance(estimators, dict) and 'estimator' in estimators.keys():
            estimators = estimators['estimator']

        # Handle multi-output models (e.g., MultiOutputClassifier/Regressor)
        if isinstance(estimators[0], MultiOutputClassifier):
            # Recursively call predict_proba_cv for each output dimension
            return [self.predict_proba_cv(s, rescale=rescale) for s in self.multioutput_estimators]

        # Initialize a DataFrame to store predicted probabilities
        y_pred = pd.DataFrame(index=self.obs_names, columns=estimators[0].classes_)

        # Predict probabilities for each test set (from cross-validation)
        for vl_idx, model in zip(self.indices['test'], estimators):
            y_pred.iloc[vl_idx] = model.predict_proba(self._X.iloc[vl_idx])

        # Optionally rescale predicted probabilities
        if rescale:
            null_pred = y_pred.copy()
            for vl_idx, model in zip(self.indices['test'], estimators):
                null_pred.iloc[vl_idx] = model.predict_proba(self._X.iloc[vl_idx] * 0.0)

            # Rescale based on mean and standard deviation of null predictions
            null_pred = null_pred.iloc[:, 1]  # Extract probabilities for class 1
            ymin, ymax = null_pred.mean() + 2 * null_pred.std(), y_pred.iloc[:, 1].max()
            y1 = y_pred.iloc[:, 1]
            y_pred.iloc[:, 1] = np.clip((y1 - ymin) / (y1.max() - ymin), 0, 1) * ymax

        return y_pred.astype(float)

    def _map_input(self, X):
        """Maps the input data to the model features and handles missing values."""
        # Convert AnnData to DataFrame if needed
        X = X.to_df() if isinstance(X, ad.AnnData) else X.copy()

        if hasattr(X, 'columns'):
            # Subset the test data to only the columns available in the training data
            cols = X.columns[X.columns.isin(self.features)]
            X_new = X[cols].copy()

            # Identify missing and additional columns
            missing_cols = self.features.difference(X.columns)
            additional_cols = X.columns.difference(self.features)

            # Add missing columns in a single concat operation to avoid fragmentation
            if not missing_cols.empty:
                X_missing = pd.DataFrame(0, index=X_new.index, columns=missing_cols)
                X_new = pd.concat([X_new, X_missing], axis=1)

            # Reorder columns to match training data
            X_new = X_new[self.features]

            if len(missing_cols) > 0:
                print(
                    f'{len(missing_cols)} out of {len(self.features)} features are missing from '
                    'your data and have been filled with zeros. You can access the features '
                    'available in the training data via `model.features`.'
                )

            if len(additional_cols) > 0:
                print(
                    f'{len(additional_cols)} out of {len(X.columns)} features in your data are not '
                    'present in the training data. These features will not impact predictions. You '
                    'can access the features available in the training data via `model.features`.'
                )

            X = X_new.copy()

        else:
            if len(X.T) != len(self.features):
                raise ValueError('X must have the same number of features as the training data.')

        # Ensure a DataFrame and fill NaNs
        X = pd.DataFrame(X)
        if X.isna().any().any():
            X.fillna(0, inplace=True)
            print('Your data contains missing values (NaNs) which have been filled with zeros.')

        return X

    def predict_proba(self, data, estimators=None, rescale=True):
        """
        Predicts class probabilities using trained estimators.

        Parameters:
        ----------
        data : pandas.DataFrame or anndata.AnnData
            The input data for prediction.

        estimators : list or dict, optional (default=None)
            A list of trained estimators. If None, it uses the pre-trained `self.estimators`.

        rescale : bool, optional (default=True)
            Whether to rescale the predicted probabilities.
            Rescaling is performed based on the predictions when all features are zeroed out.

        Returns:
        -------
        y_pred : pandas.DataFrame
            A DataFrame containing the predicted probabilities for each observation and class.
        """
        X = self._map_input(data)  # Validate input data

        # Use the stored cross-validation estimators if none are provided
        if estimators is None:
            estimators = self.estimators

        # If the estimators are provided as a dictionary, extract the 'estimator' key
        if isinstance(estimators, dict) and 'estimator' in estimators.keys():
            estimators = estimators['estimator']

        # Handle multi-output models (e.g., MultiOutputClassifier/Regressor)
        if isinstance(estimators[0], MultiOutputClassifier):
            # Recursively call predict_proba_cv for each output dimension
            return [self.predict_proba(s, rescale=rescale) for s in self.multioutput_estimators]

        # Prediction probabilities for the input data
        y_pred = np.mean([m.predict_proba(X) for m in estimators], axis=0)

        # Optionally rescale predicted probabilities to adjust for class imbalance
        if rescale:
            # Predict probabilities when all features are zeroed out
            null_pred = np.vstack([m.predict_proba(X * 0) for m in estimators])[:, 1]
            ymin, ymax = null_pred.mean() + 2 * null_pred.std(), y_pred[:, 1].max()
            # Rescale probabilities for class 1
            y_pred[:, 1] = np.clip((y_pred[:, 1] - ymin) / (y_pred[:, 1].max() - ymin), 0, 1) * ymax

        # Convert the prediction results into a pandas DataFrame
        y_pred = pd.DataFrame(y_pred, index=X.index, columns=estimators[0].classes_)

        return y_pred.astype(float)

    def predict_proba_across_estimators(self, data, estimators=None, rescale=True):
        """
        Predict and collects class probabilities across all trained estimators.

        Parameters:
        ----------
        data : pandas.DataFrame or anndata.AnnData
            The input data for prediction.

        estimators : list or dict, optional (default=None)
            A list of trained estimators. If None, it uses the pre-trained `self.estimators`.

        rescale : bool, optional (default=True)
            Whether to rescale the predicted probabilities.
            Rescaling is performed based on the predictions when all features are zeroed out.

        Returns:
        -------
        y_pred : pandas.DataFrame
            A DataFrame containing the predicted probabilities for each observation and class.
        """
        X = self._map_input(data)  # Validate input data

        # Use the stored cross-validation estimators if none are provided
        if estimators is None:
            estimators = self.estimators

        # If the estimators are provided as a dictionary, extract the 'estimator' key
        if isinstance(estimators, dict) and 'estimator' in estimators.keys():
            estimators = estimators['estimator']

        # Handle multi-output models (e.g., MultiOutputClassifier/Regressor)
        if isinstance(estimators[0], MultiOutputClassifier):
            # Recursively call predict_proba_cv for each output dimension
            return [self.predict_proba(s, rescale=rescale) for s in self.multioutput_estimators]

        # Prediction probabilities for the input data
        y_pred = np.stack([m.predict_proba(X)[:, -1] for m in estimators]).T

        # Optionally rescale predicted probabilities to adjust for class imbalance
        if rescale:
            # Predict probabilities when all features are zeroed out
            null_pred = np.vstack([m.predict_proba(X * 0) for m in estimators])[:, 1]
            ymin, ymax = null_pred.mean() + 2 * null_pred.std(), y_pred[:, 1].max()
            # Rescale probabilities for class 1
            y_pred = np.clip((y_pred - ymin) / (y_pred.max() - ymin), 0, 1) * ymax

        # Convert the prediction results into a pandas DataFrame
        y_pred = pd.DataFrame(y_pred, index=X.index)

        return y_pred.astype(float)

    def predict_cv(self):
        """
        Predicts the DILI risk from pathways signatures.

        Parameters:
        ----------
        data : pandas.DataFrame or anndata.AnnData
            The input data for prediction.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame containing the predicted DILI risk for each observation.
        """
        # Map input data to model features and predict probabilities
        y_pred = self.predict_proba_cv()
        y_pred['DILI_probability'] = y_pred.iloc[:, -1]

        return y_pred[['DILI_probability']]

    def predict(self, data):
        """
        Predicts the DILI risk from pathways signatures.

        Parameters:
        ----------
        data : pandas.DataFrame or anndata.AnnData
            The input data for prediction.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame containing the predicted DILI risk for each observation.
        """
        # Map input data to model features and predict probabilities
        y_pred = self.predict_proba(data)
        y_pred['DILI_probability'] = y_pred.iloc[:, -1]

        return y_pred[['DILI_probability']]

    def compute_safety_margin(
        self,
        data,
        pert_col,
        dose_col,
        cmax_col,
        y_pred_col=None,
        y_thresh=None,
        retain_all_cols=None,
    ):
        """
        Computes the margin of safety (MOS) based on predicted toxicity.

        Parameters:
        ----------
        data : anndata.AnnData or pandas.DataFrame
            Input data containing observations and predictions.
            Must include columns like 'compound_name', 'dose_uM', and 'Cmax_uM'.
        pert_col : str
            Column indicating perturbations (e.g., compound names).
        dose_col : str
            Column with dose values (e.g., dose_uM).
        cmax_col : str
            Column with Cmax values (e.g., Cmax_uM).
        y_pred_col : str
            Column representing predicted probabilities of toxicity.
        y_thresh : float, optional
            Threshold for toxicity predictions. Default uses `self.DILI_pred_cutoff` or 0.7.

        Returns:
        -------
        pandas.DataFrame
            DataFrame grouped by compound, with computed MOS and DILI flags.
        """
        y_thresh = y_thresh or self.DILI_pred_cutoff or 0.7
        self.DILI_pred_cutoff = y_thresh

        # Copy the observation data to avoid modifying the original
        df_obs = data.obs.copy() if isinstance(data, ad.AnnData) else data.copy()

        y_pred_col = y_pred_col or 'DILI_prob'
        if y_pred_col not in df_obs.columns:
            df_obs[y_pred_col] = self.predict(data)['DILI_probability']

        # Create a table with compound x dose and y_prediction as values
        df_pred = crosstab(df_obs, [pert_col, dose_col, y_pred_col], aggfunc='mean') > y_thresh

        # Compute RNA_IC50_uM: the minimum toxic dose per compound
        df_obs['RNA_IC50_uM'] = df_obs[pert_col].map(
            (df_pred * df_pred.columns).replace(0, np.nan).min(axis=1)
        )

        # Calculate MOS (Margin of Safety) for different indicators, with clamping between 1 and 300
        df_obs['MOS_Cmax'] = np.clip(1e4 / df_obs[cmax_col], 1, 300)

        df_obs['MOS_LDH'] = (
            np.clip(np.nan_to_num(df_obs['LDH_IC10_uM'] / df_obs[cmax_col], nan=300), 1, 300)
            if 'LDH_IC10_uM' in df_obs.columns
            else 300
        )

        df_obs['MOS_RNA'] = (
            np.clip(np.nan_to_num(df_obs['RNA_IC50_uM'] / df_obs[cmax_col], nan=300), 1, 300)
            if 'RNA_IC50_uM' in df_obs.columns
            else 300
        )

        # Calculate the final MOS by taking the minimum across MOS_Cmax, MOS_LDH, and MOS_RNA
        cols = [k for k in ['MOS_Cmax', 'MOS_LDH', 'MOS_RNA'] if k in df_obs.columns]
        df_obs['MOS'] = df_obs[cols].min(axis=1)

        # Determine DILI flag based on MOS values
        df_obs['DILI_flag'] = np.select(
            [df_obs['MOS_RNA'] < 80, df_obs['MOS_LDH'] < 80, df_obs['MOS_Cmax'] < 80],
            ['RNA', 'LDH', 'Cmax'],
            default='none',
        )

        # Group the results by compound_name
        df_res = groupby(df_obs, 'compound_name')

        if not retain_all_cols:
            all_cols = [cmax_col, 'RNA_IC50_uM'] + cols + ['MOS', 'DILI_flag', 'DILI_label']
            all_cols = [c for c in all_cols if c in df_res.columns]
            df_res = df_res[all_cols]

        self.safety_margin_results = df_res.copy()
        return df_res

    def _compute_DILImap_margins(self, **kwargs):
        self.DILImap_margins = self.compute_safety_margin(**kwargs, retain_all_cols=True)

    def compute_empirical_DILI_risk(self, IC50_uM=None, dose_range=None, logspace=True, num=500):
        """
        Calculates the empirical DILI risk likelihoods based on safety margin results.

        Parameters:
        ----------
        IC50 : float, optional
            The IC50 value to compare against the Margin of Safety (MOS). Default is 1e5.

        num : int, optional
            Number of dose points to evaluate. Default is 500.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame with dose values and corresponding empirical DILI likelihood.
        """
        if self.DILImap_margins is None:
            raise ValueError('Please run `_compute_DILImap_margins` first.')

        IC50_uM = np.nan_to_num(IC50_uM, 1e4) or 1e4
        df_res = self.DILImap_margins
        df = df_res[df_res['DILI_label'].isin(['DILI (withdrawn)', 'DILI (known)', 'No DILI'])]

        tot_pos = df['DILI_label'].str.startswith('DILI').sum()
        tot_neg = df['DILI_label'].str.startswith('No').sum()

        if dose_range is None:
            dose_range = [1e-2, 30]

        if logspace:
            log_dose_range = np.log10(dose_range)
            doses = np.logspace(log_dose_range[0], log_dose_range[1], num)
        else:
            doses = np.linspace(dose_range[0], dose_range[1], num)

        DILI_risk_avg = 0
        IC50s = [IC50_uM * -np.log10(0.5), IC50_uM, IC50_uM / -np.log10(0.5)]
        for ic50 in IC50s:
            DILI_risk = [
                (df[df['MOS'] > ic50 / d]['DILI_label'].str.startswith('DILI').sum() / tot_pos)
                - (df[df['MOS'] < ic50 / d]['DILI_label'].str.startswith('No').sum() / tot_neg)
                for d in doses
            ]

            # Clip values, and smooth the risk curve
            DILI_risk = np.clip(DILI_risk, 0, 1)
            DILI_risk_avg += pd.Series(np.nan_to_num(DILI_risk, nan=0)).rolling(window=3).mean()

        DILI_risk_avg = DILI_risk_avg / len(IC50s)

        # Create results DataFrame
        df_results = pd.DataFrame({'dose': doses, 'DILI_risk': DILI_risk_avg})
        df_results = df_results.dropna().reset_index(drop=True)

        return df_results

    def compute_DILI_dose_regimes(model, IC50_uM=None, dose_range=None):
        """
        Determines DILI dose regimes based on specified IC50.

        Parameters:
        ----------
        IC50_uM : float, optional
            The IC50 value (first dose showing DILI signal) for the compound in uM units.

        Returns:
        -------
        dict
            A dictionary categorizing the dose regimes into high, mid-high, medium, low DILI risk.
        """
        df = model.compute_empirical_DILI_risk(IC50_uM=IC50_uM, dose_range=dose_range)
        dose, DILI_risk = df['dose'], df['DILI_risk']

        cutoffs_uM = []
        for k in [0.2, 0.5, 0.8]:
            if np.any(DILI_risk > k) and np.any(DILI_risk < k):
                cutoffs_uM.append(np.round(dose[np.argmax(DILI_risk > k)], 3))
            else:
                cutoffs_uM.append(np.nan)  # Use NaN if no dose exceeds the threshold

        res = {
            'High risk': f'>{cutoffs_uM[2]} μM' if not np.isnan(cutoffs_uM[2]) else 'NA',
            'Mid-High risk': f'{cutoffs_uM[1]} - {cutoffs_uM[2]} μM'
            if not np.isnan(cutoffs_uM[1])
            else 'NA',
            'Medium risk': f'{cutoffs_uM[0]} - {cutoffs_uM[1]} μM'
            if not np.isnan(cutoffs_uM[0])
            else 'NA',
            'Low risk': f'<{cutoffs_uM[0]} μM' if not np.isnan(cutoffs_uM[0]) else '<30uM',
        }
        return res

    def plot_DILI_dose_regimes(
        self,
        compound,
        dose_range=None,
        xmax=None,
        fontsize=14,
        show_cmax=True,
        show_legend=True,
        ax=None,
    ):
        """
        Plots the DILI likelihood dose-response curve for a given compound.

        Parameters:
        ----------
        compound : str
            The name of the compound for which the DILI dose regimes will be plotted.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        if self.safety_margin_results is None:
            raise ValueError('Please run `compute_safety_margin` first.')

        # Estimate empirical likelihoods
        IC50_uM = self.safety_margin_results['RNA_IC50_uM'][compound]
        Cmax_uM = self.safety_margin_results['Cmax_uM'][compound]

        DILI_likelihoods = self.compute_empirical_DILI_risk(IC50_uM=IC50_uM, dose_range=dose_range)
        dose_regimes = self.compute_DILI_dose_regimes(IC50_uM=IC50_uM, dose_range=dose_range)

        dose = DILI_likelihoods['dose']
        DILI_likelihood = DILI_likelihoods['DILI_risk']

        # Plot
        ax.scatter(dose, DILI_likelihood, color='k', s=10)
        ax.plot(dose, DILI_likelihood, color='k')

        for d, (n, regime) in zip([0.8, 0.5, 0.2, 0], dose_regimes.items()):
            ax.axhline(d, label=f'{n} ({regime})', c='red', linestyle='--', linewidth=max(d, 0.1))

        ax.set_yticks(
            [0, 0.2, 0.5, 0.8, 1], ['0%', '20%', '50%', '80%', '100%'], fontsize=fontsize - 2
        )

        if show_cmax:
            ax.axvline(Cmax_uM, color='g', linestyle='-', linewidth=1, label=f'Cmax ({Cmax_uM} μM)')
        ax.set_xlabel('Plasma Cmax (μM)', fontsize=fontsize)
        ax.set_ylabel('Likelihood of DILI risk', fontsize=fontsize)
        ax.set_title(compound)
        if xmax is not None:
            ax.set_xlim(-0.5, xmax)
        ax.set_ylim(-0.05, 1.05)

        ax.set_xscale('log')

        if show_legend:
            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))

    @property
    def DILI_pathways(self):
        """Returns the groups of pathways associated with DILI predictions."""
        pathways_dict = {
            # 1. Direct Hepatotoxicity
            # (pathways that directly contribute to liver cell injury through the formation of
            # reactive metabolites, which can cause cellular damage, necrosis, or apoptosis)
            'Direct Hepatotoxicity': [
                'Aflatoxin B1 metabolism WP699',
                'Arylamine metabolism WP694',
                'Benzo(a)pyrene metabolism WP696',
                'Codeine and Morphine Metabolism WP1604',
            ],
            # 2. Oxidative and cellular stress
            # (pathways involved in the generation oxidative stress. Excessive oxidative stress can
            # lead to cellular damage, lipid peroxidation, and apoptosis.)
            'Oxidative stress': [
                'Oxidative Stress WP408',
                'Oxidative Damage WP3941',
                'Ferroptosis WP4313',
                'NRF2 pathway WP2884',
                'NRF2-ARE regulation WP4357',
            ],
            # 3. Drug Metabolism and Detoxification (pathways that metabolize drugs and other
            # xenobiotics. Dysregulation or overload of these pathways can lead to the accumulation
            # of toxic substances and reactive intermediates, causing liver injury.)
            'Metabolism': [
                'Glucuronidation WP698',
                'Oxidation by Cytochrome P450 WP43',
                'Metapathway biotransformation Phase I and II WP702',
                'Phase I biotransformations, non P450 WP136',
                'Sulindac Metabolic Pathway WP2542',
                'Tamoxifen metabolism WP691',
                'Drug Induction of Bile Acid Pathway WP2289',
                'Aryl Hydrocarbon Receptor Pathway WP2873',
                'Aryl Hydrocarbon Receptor WP2586',
            ],
            # 4. Endocrine and Receptor Signaling (pathways involve hormonal and receptor-mediated
            # signaling that regulates liver metabolism. Dysregulation can affect liver function and
            # its ability to handle drugs and toxins, indirectly contributing to DILI.)
            'Receptor signaling': [
                'Estrogen Receptor Pathway WP2881',
                'Constitutive Androstane Receptor Pathway WP2875',
                'Farnesoid X Receptor  Pathway WP2879',
                'Liver X Receptor Pathway WP2874',
                'Nuclear Receptors in Lipid Metabolism and Toxicity WP299',
                'Nuclear Receptors Meta-Pathway WP2882',
                'Pregnane X Receptor pathway WP2876',
            ],
            # 5. Lipid Metabolism and Cholesterol Homeostasis (pathways regulating lipid metabolism
            # and cholesterol synthesis. Imbalances can lead to steatosis and stress on liver
            # function, potentially increasing the risk of DILI.)
            'Lipid metabolism': [
                'Cholesterol Biosynthesis Pathway WP197',
                'Fatty Acid Beta Oxidation WP143',
                'Fatty Acid Biosynthesis WP357',
                'Fatty Acid Omega Oxidation WP206',
                'SREBF and miR33 in cholesterol and lipid homeostasis WP2011',
                'Sterol Regulatory Element-Binding Proteins (SREBP) signalling WP1982',
                'Metabolic pathway of LDL, HDL and TG, including diseases WP4522',
                'PPAR Alpha Pathway WP2878',
                'PPAR signaling pathway WP3942',
                'Mevalonate pathway WP3963',
            ],
            # 6. Metabolic Pathways (pathways essential for various metabolic functions such as
            # amino acid and protein metabolism. Disturbances in these pathways can indirectly
            # stress liver cells and alter susceptibility to DILI.)
            'Metabolic pathways': [
                'Alanine and aspartate metabolism WP106',
                'Amino Acid metabolism WP3925',
                'Cysteine and methionine catabolism WP4504',
                'One carbon metabolism and related pathways WP3940',
                'Tryptophan metabolism WP465',
                'Mitochondrial LC-Fatty Acid Beta-Oxidation WP368',
            ],
            # 7. Miscellaneous (pathways with complex or less direct links to DILI. They may
            # influence liver health through indirect mechanisms like inflammation, energy
            # metabolism, or bile acid regulation.)
            'Miscellaneous': [
                'Electron Transport Chain (OXPHOS system in mitochondria) WP111',
                'Oxidative phosphorylation WP623',
                'Statin Pathway WP430',
                'Valproic acid pathway WP3871',
            ],
        }

        return {key: [p for p in pws if p in self.features] for key, pws in pathways_dict.items()}

    def feature_importances(self, dili_pathways=None):
        from sklearn.metrics import roc_auc_score

        df = pd.DataFrame(columns=['AUC', 'MDI'])
        for k in self.features:
            df.loc[k, 'AUC'] = roc_auc_score(self._y, np.ravel(self._X[k]))

        df['MDI'] = np.mean([m.feature_importances_ for m in self.estimators], axis=0)

        if dili_pathways:
            df = df[[k for k in np.hstack(list(self.DILI_pathways.values())) if k in df.index]]
        return df

    def pull_data(self):
        """Loads the training data from the model."""
        from .datasets import DILImap_training_data

        return DILImap_training_data()

    def save_model(self, filepath, push_to_s3=False):
        """Save the model to S3 registry."""
        if self.DILImap_margins is None and self.safety_margin_results is not None:
            self.DILImap_margins = self.safety_margin_results

        import copy

        model_copy = copy.deepcopy(self)
        for attr in self.__dict__.keys():
            if attr.startswith('_'):
                del model_copy.__dict__[attr]
        if push_to_s3:
            write(model_copy, filename=filepath, package_name='public/models')
        else:
            with open(filepath, 'wb') as f:
                joblib.dump(model_copy, f)
