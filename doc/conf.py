# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey
project = 'nna_methods'
copyright = '2024, Barron H. Henderson'
author = 'Barron H. Henderson'
with open('../nna_methods/__init__.py', 'r') as initf:
    for l in initf.readline():
        if l.strip().startswith('__version__ ='):
            release = l.split('=')[1].strip()
    else:
        release = '9.9.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_nb',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']

# -- Options for Gallery --
class MyOrder:
    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.order = [
            'plot_californiahousing.py',
            'plot_correctedozone.py',
            'plot_equatesevna.py',
        ]
 
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __call__(self, filename):
        if not filename in self.order:
            self.order.append(filename)

        return (self.order.index(filename), filename)

sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'within_subsection_order': MyOrder
}

