import nbclean
import os
from pathlib import Path
from datetime import datetime


def setup(app):
    # read time of previous build
    try:
        text = (Path.cwd() / '.last_sphinx_build').read_text()
        last_build_time = float(text)
    except FileNotFoundError:
        last_build_time = 0
    # iterate through all source files and check time stemps
    cwd = Path.cwd()
    paths = sorted(
        [
            f
            for f in cwd.parent.glob('**/*')
            if f.suffix in {'.ipynb', '.rst'}
            and 'ipynb_checkpoints' not in f.as_posix()
            and cwd not in f.parents  # do not consider files in cwd
        ]
    )

    print('cleaning outdated notebooks:')
    for f in paths:
        time = f.stat().st_mtime
        if time > last_build_time:
            print('   ', f)
            # new file is same file but in 'docs/' sub-directory
            newfile = Path(f.as_posix().replace(cwd.parent.as_posix(), cwd.as_posix()))
            os.makedirs(newfile.parent, exist_ok=True)
            if f.suffix == '.rst':  # copy file
                Path(newfile).write_text(f.read_text())

            elif f.suffix == '.ipynb':  # copy and strip file
                ntbk = nbclean.NotebookCleaner(f.as_posix())
                ntbk.clear('stderr')
                ntbk.save(newfile.as_posix())
    # write time of this build
    (Path.cwd() / '.last_sphinx_build').write_text(str(datetime.now().timestamp()))
