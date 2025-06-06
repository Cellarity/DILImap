import os
import warnings


__licence__ = 'BSD (3 clause)'


def get_github_repo(app, path):
    return app.config.github_repo, '/docs/'


def html_page_context(app, pagename, templatename, context, doctree):
    if templatename != 'page.html':
        return

    if not app.config.github_repo:
        warnings.warn('`github_repo `not specified', stacklevel=2)
        return

    if not app.config.github_nb_repo:
        app.config.github_nb_repo = app.config.github_repo

    path = os.path.relpath(doctree.get('source'), app.builder.srcdir)
    repo, conf_py_path = get_github_repo(app, path)

    # For sphinx_rtd_theme.
    context['display_github'] = True
    context['github_user'] = 'cellarity'
    context['github_version'] = 'main'
    context['github_repo'] = repo
    context['conf_py_path'] = conf_py_path


def setup(app):
    app.add_config_value('github_nb_repo', '', True)
    app.add_config_value('github_repo', '', True)
    app.connect('html-page-context', html_page_context)
