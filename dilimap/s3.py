"""Reading and writing from S3 registry."""

import warnings
import contextlib
import json
import os
import pathlib
import pickle
import joblib
import pandas as pd
import anndata as ad
import quilt3
import boto3
import shutil
from pathlib import Path
from dotenv import load_dotenv
from getpass import getpass
from typing import Any, Callable, Dict, Optional


def login():
    """
    Load AWS credentials from a .env file or prompt the user interactively if not found.

    This function sets the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the environment.
    """
    # Try to load from .env
    if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
        load_dotenv()

    # Prompt if not set
    if not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'):
        print('AWS credentials not found in environment or .env file.')
        os.environ['AWS_ACCESS_KEY_ID'] = getpass('Enter AWS Access Key ID: ')
        os.environ['AWS_SECRET_ACCESS_KEY'] = getpass('Enter AWS Secret Access Key: ')

    # Try validating credentials
    try:
        list_files()
    except Exception as e:
        print(f"That didn't work. {e}. Try again.")
        os.environ.pop('AWS_ACCESS_KEY_ID', None)
        os.environ.pop('AWS_SECRET_ACCESS_KEY', None)


def list_files(registry='s3://dilimap'):
    """
    List all files/packages available in a given S3 bucket.

    Args:
        registry (str): The S3 registry URI in the format 's3://bucket-name'.

    Returns:
        list: A list of filenames (keys) in the bucket.
    """
    bucket_name = registry.replace('s3://', '').strip('/')

    session = boto3.Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    )
    s3 = session.client('s3')

    response = s3.list_objects_v2(Bucket=bucket_name)
    files = []

    if 'Contents' in response:
        for obj in response['Contents']:
            files.append(obj['Key'])
    else:
        print(f"No files found in registry '{registry}'.")

    files = [f for f in files if not f.startswith('.')]

    return files


def read(filename, package_name='public/data', registry='s3://dilimap', **kwargs):
    """
    Read a file from an S3 registry.

    Args:
        filename (str): The name of the file within the package.
        package_name (str, optional): The name of the package, e.g., 'public/data', 'public/model' or 'propietary/data'
        registry (str, optional): The S3 registry where the package is stored. Defaults to 's3://dilimap'.
        **kwargs: Additional arguments passed to the `load_file` method.

    Returns:
        File contents loaded from the specified filename.
    """
    login()

    lqp = QuiltPackage(name=package_name, registry=registry)
    return lqp.load_file(filename, **kwargs)


def write(obj, filename, package_name='public/data', registry='s3://dilimap', **kwargs):
    """
    Write an object to an S3 registry.

    Args:
        obj: The object to write to the package.
        filename (str): The name of the file within the package.
        package_name (str, optional): The name of the package, e.g., 'public/data', 'public/model' or 'propietary/data'
        registry (str, optional): The S3 registry where the package is stored. Defaults to 's3://dilimap'.
        **kwargs: Additional arguments passed to the `add_obj` method.
    """
    login()

    lqp = QuiltPackage(name=package_name, registry=registry)
    lqp.add_obj(obj, filename, **kwargs)
    lqp.push()


def _write_h5ad(obj: Any, filepath: str):
    """Write an object to disk as an .h5ad file by calling `.write_h5ad()`."""  # noqa: D402
    obj.write_h5ad(filepath)


def _write_parquet(obj: Any, filepath: str):
    """Write an object to disk as an .parquet file by calling `.to_parquet()`."""
    obj.to_parquet(filepath)


def _write_csv(obj: Any, filepath: str):
    """Write an object to disk as an .csv file by calling `.to_csv()`."""
    obj.to_csv(filepath)


def _write_tsv(obj: Any, filepath: str):
    """Write an object to disk as an .tsv file by calling `.to_csv()` with tab as the separator."""
    obj.to_csv(filepath, sep='\t')


def _write_buffer(buffer, path):
    """Write an object to disk as an .csv file by calling `.to_csv()`."""
    with open(path, 'wb') as f:
        f.write(buffer.getbuffer())


def _write_joblib(obj: Any, filepath: str):
    """Write a `joblib` object to disk to `filepath` by calling joblib.dump()."""
    return joblib.dump(obj, filepath)


def _write_json(obj: dict, filepath: str):
    """Write dictionary to disk as a .json file by calling json.dump()."""
    with open(filepath, 'w') as f:
        json.dump(obj, f)


def _write_torch_pt(obj: dict, filepath: str):
    """Write object to .pt file at `filepath`."""
    import torch

    return torch.save(obj, filepath)


def _write_pickle(obj: Any, filepath: str):
    """Write object to .pickle file at `filepath`."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def _read_anndata_from_h5ad(filepath, **kwargs):
    """Load an `AnnData` object from the .h5ad file at `filepath`."""
    return ad.read_h5ad(filepath, **kwargs)


def _read_df_from_parquet(filepath, **kwargs):
    """Load a `pandas.DataFrame` object from the .parquet file at `filepath`."""
    return pd.read_parquet(filepath, **kwargs)


def _read_df_from_csv(filepath, **kwargs):
    """Load a `pandas.DataFrame` object from the .csv file at `filepath`."""
    return pd.read_csv(filepath, **kwargs)


def _read_df_from_tsv(filepath, **kwargs):
    """Load a `pandas.DataFrame` object from the .tsv file at `filepath`."""
    return pd.read_csv(filepath, sep='\t', **kwargs)


def _read_joblib(filepath: str, **kwargs):
    """Load a `joblib` object from .joblib file at `filepath`."""
    return joblib.load(filepath, **kwargs)


def _read_json(filepath: str, **kwargs):
    """Load a dictionary from .json file at `filepath`."""
    with open(filepath) as json_file:
        return json.load(json_file, **kwargs)


def _read_torch_pt(filepath: str, **kwargs):
    """Load an object from .pt file at `filepath`."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "The 'torch' package is required to load .pt files. "
            'Please install it using: pip install torch'
        ) from e

    return torch.load(filepath, map_location=torch.device('cpu'), **kwargs)


def _read_pickle(filepath: str, **kwargs):
    """Load an object from .pickle file at `filepath`."""
    with open(filepath, 'rb') as f:
        return pickle.load(f, **kwargs)


def _read_excel(filepath: str, **kwargs):
    """Load an object from .xlsx file at `filepath`."""
    try:
        import openpyxl as _  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The 'openpyxl' package is required to read Excel files. "
            'Please install it using: pip install openpyxl'
        ) from e

    return pd.read_excel(filepath, **kwargs)


def _print_blue(text):
    print(f'\033[1;34m{text}\033[m')


file_extension_to_write_func = {
    '.h5ad': _write_h5ad,
    '.parquet': _write_parquet,
    '.csv': _write_csv,
    '.tsv': _write_tsv,
    '.joblib': _write_joblib,
    '.json': _write_json,
    '.pt': _write_torch_pt,
    '.pkl': _write_pickle,
}

file_extension_to_read_func = {
    '.h5ad': _read_anndata_from_h5ad,
    '.parquet': _read_df_from_parquet,
    '.csv': _read_df_from_csv,
    '.tsv': _read_df_from_tsv,
    '.joblib': _read_joblib,
    '.json': _read_json,
    '.pt': _read_torch_pt,
    '.pkl': _read_pickle,
    '.xlsx': _read_excel,
}


class QuiltPackage:
    """Simple client for working with Quilt Packages.

    :tutorial:`LocalQuiltPackage`

    Params
    ------
    name: str
        Name of Quilt package. Must be in the form `{namespace}/{packagename}`.
    registry: str
        Name of S3 bucket where the package currently resides or will reside after pushing.
        Must include `"s3://"` prefix.
    top_hash: str (default: None)
        Top hash of the package. If left as None, the latest will be used by default.
    local_base_dir : str (default: None)
        Directory where package will be installed. Set to `"~/data"` if left as None.
    metadata: dict (default: None)
        Package-level metadata that will be stored alongside the package on S3.
    include_bucket_name_in_path: bool (default: True)
        If `True`, the package is stored in the directory `{local_base_dir}/{bucket_name}/`,
        where `bucket_name` is made by removing the `s3://` prefix from registry. If
        `False`, it is stored in `{local_base_dir}` directly.
    """

    def __init__(
        self,
        name: str = 'models/DILImap',
        registry: str = 's3://dilimap',
        top_hash: Optional[str] = None,
        local_base_dir: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        include_bucket_name_in_path: bool = True,
    ):
        self.name = name
        self.registry = registry
        self.s3_bucket_name = registry.split('//')[-1]
        self.top_hash = top_hash

        if not local_base_dir:
            local_base_dir = os.path.join(os.path.expanduser('~'), 'data')

        if include_bucket_name_in_path:
            self.local_dir = os.path.join(local_base_dir, self.s3_bucket_name, name)
        else:
            self.local_dir = os.path.join(local_base_dir, name)

        # if the local directory for our package does not exist, make it
        if not os.path.exists(self.local_dir):
            pathlib.Path(self.local_dir).mkdir(parents=True)

        self.package_exists_remotely = False
        self.quilt_package = quilt3.Package()
        try:
            if self.name in set(quilt3.list_packages(self.registry)):
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        self.quilt_package = quilt3.Package.browse(
                            self.name, registry=self.registry, top_hash=self.top_hash
                        )
                        self.package_exists_remotely = True
            else:
                raise ValueError('Could not find Package')
        except ValueError as e:
            raise ValueError(
                'S3NoValidClientError: Likely an invalid client or networking issues'
            ) from e

        if metadata is not None:
            self.quilt_package.set_meta(metadata)

        # always print first 10 digits of top hash for reproducibility
        _print_blue(
            f'Package: {self.registry}/{self.name}. Top hash: {self.quilt_package.top_hash[:10]}'
        )

    def add_obj(
        self,
        obj: Any,
        filepath: str,
        write_func: Optional[Callable[[Any, str], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add an object or existing file to the S3 package.

        If `obj` is a file path, it is copied into the local package directory
        under the given `filepath`. Otherwise, `write_func` is used to write the object.

        Calling `.push()` is required to upload the files to the remote package.

        Params
        ------
        obj: Object
            If str and path exists, will be treated as a file to include directly.
        filepath: str
            Local path that the file will be written to. Path should be relative to local
            package directory.
        write_func: callable
            Callable that takes as input an object and a filepath and writes out the object to
            that filepath.
        metadata: dict
            Object-level metadata that will be uploaded to S3 alongside the file.
        """
        dst_path = os.path.join(self.local_dir, filepath)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if isinstance(obj, (str, Path)) and os.path.isfile(obj):
            # copy file to local package directory
            shutil.copy2(obj, dst_path)

        else:
            # write file to local package directory
            if write_func is None:
                _, file_extension = os.path.splitext(filepath)
                if file_extension not in file_extension_to_write_func:
                    recognized_filetypes_str = ', '.join(file_extension_to_write_func.keys())
                    raise ValueError(
                        f'Extension {file_extension} not recognized. Please provide '
                        f'one of the following: {recognized_filetypes_str}.'
                    )

                write_func = file_extension_to_write_func[file_extension]
            write_func(obj, dst_path)

        if metadata is not None:
            self.quilt_package.set(filepath, dst_path, meta=metadata)
        else:
            self.quilt_package.set(filepath, dst_path)

    def push(
        self, output_format: str = 'quilt-raw', message: Optional[str] = None, force: bool = False
    ):
        """Push local changes to the remote package.

        Params
        ------
        output_format: str
            Controls status output messages. If `"quilt-raw"`, the output from `quilt-raw`
            will be displayed. If "`silent`", no status messages will be displayed.
        message: str
            Commit message for the package update.
        force: bool
            Force overwrite package name, note: top_hash will always save previous package version.
        """

        def _push():
            self.quilt_package.push(
                name=self.name, registry=self.registry, force=force, message=message
            )

        if output_format == 'silent':
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    _push()
        elif output_format == 'quilt-raw':
            _push()
        else:
            raise ValueError('`output` must be either `"silent"` or `"quilt-raw"`.')

    def install(self):
        """Download all files from the remote Quilt package."""
        quilt3.Package().install(
            name=self.name, registry=self.registry, dest=self.local_dir, top_hash=self.top_hash
        )

    def install_file(self, filepath: str, output_format: str = 'per-entry'):
        """Download an individual file from the remote Quilt package.

        Params
        ------
        filepath: str
            Path of the file to be installed. The path should be relative to the local package
            directory.
        output_format: str (default `"per-entry"`)
            If `"per-entry"`, a status method is printed that is specific to the file being
            installed. If `"quilt-raw"`, raw output from the Quilt Package method .install() is
            printed. If `"silent"`, nothing is printed.
        """
        if output_format not in ['per-entry', 'quilt-raw', 'silent']:
            raise ValueError("output must be 'per-entry', 'quilt-raw', 'silent'.")

        full_path = os.path.normpath(filepath).split(os.sep)
        if len(full_path) == 1:
            subdir = ''
        else:
            subdir = os.path.join(*full_path[:-1])
        local_subdir = os.path.join(self.local_dir, subdir)
        if not os.path.exists(local_subdir):
            pathlib.Path(local_subdir).mkdir(parents=True)

        def run_install():
            self.quilt_package.install(
                name=self.name,
                registry=self.registry,
                dest=local_subdir,
                path=filepath,
                top_hash=self.top_hash,
            )

        if output_format == 'quilt-raw':
            run_install()
        elif output_format in ['silent', 'per-entry']:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    run_install()

    def load_file(
        self,
        filepath: str,
        read_func: Optional[Callable[[str], Any]] = None,
        output_format: str = 'per-entry',
        **kwargs,
    ):
        """Install a file into the local package from the remote package.

        Params
        ------
        filepath: list
            File path of the file to be installed. The path should be relative to the local
            package directory.
        read_func: callable
            Callable that takes as input a filepath and returns a Python object.
        output_format: str (default `"per-entry"`)
            If `"tqdm"`, a `tqdm` status bar is displayed showing the progress in installing
            the packages in the list. If `"per-entry"`, status messages are printed for each
            file as it is installed. If `"quilt-raw"`, raw output from the Quilt Package
            method .install() is printed. If `"silent"`, nothing is printed.
        """
        if read_func is None:
            _, file_extension = os.path.splitext(filepath)
            if file_extension not in file_extension_to_read_func:
                recognized_filetypes_str = ', '.join(file_extension_to_write_func.keys())
                raise ValueError(
                    f'Extension {file_extension} not recognized. Please provide a custom '
                    '`read_func` or use one of the following recognize filetypes: '
                    f'{recognized_filetypes_str}.'
                )

            read_func = file_extension_to_read_func[file_extension]

        self.install_file(filepath, output_format=output_format)

        if file_extension == '.csv' and 'index_col' not in kwargs:
            kwargs['index_col'] = 0

        full_path = os.path.join(self.local_dir, filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            return read_func(full_path, **kwargs)

    def browse_package(self):
        """Print the directory structure of the Quilt package"""
        p = quilt3.Package.browse(self.name, self.registry, self.top_hash)
        print(p)
