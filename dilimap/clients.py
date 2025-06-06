import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress deprecation warning caused by chembl_webresource_client's use of pkg_resources,
# which is scheduled for removal from setuptools in late 2025.
warnings.filterwarnings('ignore', category=UserWarning, module='chembl_webresource_client')


def chembl(molecule_name=None, smiles=None):
    """
    Fetch compound information from ChEMBL.

    Args:
    molecule_name (str or list of str, optional): Compound name, synonym or CHEMBL ID to look up.
    smiles (str or list of str, optional): Canonical SMILES string of the compound.

    Returns:
    pd.DataFrame: Compound metadata indexed by ChEMBL ID.
    """
    if molecule_name is None and smiles is None:
        raise ValueError("You must provide at least one of: 'molecule_name' or 'smiles'.")

    if isinstance(molecule_name, str) or isinstance(smiles, str):
        results = [_fetch_chembl_entry(molecule_name, smiles)]
    else:
        results = []

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(_fetch_chembl_entry, name): name for name in list(molecule_name)
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

    df_res = pd.DataFrame(results)
    if 'molecule_chembl_id' in df_res.columns:
        df_res = df_res.set_index('molecule_chembl_id')

    return df_res


def drug_warnings(chembl_id, aggregate=None):
    """
    Fetch drug warnings from ChEMBL for given compound(s).

    Args:
    chembl_id (str or list of str): One or more ChEMBL IDs.
    aggregate (bool, optional): Aggregate warnings per compound (default: True if list of chembl_ids provided).

    Returns:
    pd.DataFrame: Drug warning information indexed by ChEMBL ID.
    """
    from chembl_webresource_client.new_client import new_client

    res = new_client.drug_warning.filter()

    if aggregate is None:
        aggregate = not isinstance(chembl_id, str)
    if isinstance(chembl_id, str):
        chembl_id = [chembl_id]

    results = []
    for r in res:
        if r['molecule_chembl_id'] in list(chembl_id):
            results.append(r)

    df = pd.DataFrame(results).set_index('molecule_chembl_id')

    if aggregate:
        for k in ['warning_id', 'warning_refs', 'warning_year']:
            if k in df.columns:
                df.pop(k)

        withdrawn = df[df.warning_type == 'Withdrawn']
        black_box = df[df.warning_type == 'Black Box Warning']
        black_box = black_box[~black_box.index.isin(withdrawn.index)]

        df = pd.concat([withdrawn, black_box])

        def unique_or_scalar(series):
            unique_vals = list(set(series.dropna()))
            return unique_vals[0] if len(unique_vals) == 1 else ', '.join(unique_vals)

        df = df.groupby('molecule_chembl_id').agg(unique_or_scalar)

    return df


def _fetch_chembl_entry(molecule_name=None, smiles=None):
    from chembl_webresource_client.new_client import new_client

    try:
        if molecule_name and molecule_name.upper().startswith('CHEMBL'):
            res = new_client.molecule.filter(molecule_chembl_id=molecule_name)
        elif molecule_name:
            res = new_client.molecule.filter(
                molecule_synonyms__molecule_synonym__iexact=molecule_name
            )
        elif smiles:
            res = new_client.molecule.filter(molecule_structures__canonical_smiles=smiles)
        else:
            raise ValueError("You must provide at least one of: 'molecule_name' or 'smiles'.")

        if not res:
            return None

        entry = res[0].copy()  # Keep all top-level fields

        # Add compound name
        if molecule_name:
            entry['molecule_name'] = molecule_name

        # Flatten nested fields safely
        entry['smiles'] = (entry.get('molecule_structures') or {}).get('canonical_smiles', np.nan)
        entry['mw'] = (entry.get('molecule_properties') or {}).get('full_mwt', np.nan)
        entry['alogp'] = (entry.get('molecule_properties') or {}).get('alogp', np.nan)

        # Join all list fields into a string
        for key, value in entry.items():
            if isinstance(value, list):
                entry[key] = '; '.join(str(v) for v in value if v is not None)

        # Remove large or nested fields you don't need
        for key in [
            'molecule_structures',
            'molecule_properties',
            'molecule_synonyms',
            'molecule_hierarchy',
            'biotherapeutic',
            'cross_references',
        ]:
            entry.pop(key, None)

        return entry

    except Exception as e:
        print(f'Error processing {molecule_name}: {e}')
        return None
