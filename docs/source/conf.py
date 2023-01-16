import os
import sys

sys.path.insert(0, os.path.abspath('../../gcpds'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'gcpds.image_segmentation'
copyright = '2023, Juan Carlos Aguirre Arango'
author = 'Juan Carlos Aguirre Arango'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'nbsphinx',
        'dunderlab.docs'
        ]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


autodoc_mock_imports = [
    'IPython',
    'numpy',
    'scipy',
    'mne',
    'matplotlib',
    'matplotlib.rcParams',
    'google',
    'colorama',
    'tqdm',
    'pandas',
    'tables',
    'pyedflib',
    'netifaces',
    'nmap',
    'rawutil',
    'kafka',
    'rpyc',
    'serial',
    'openbci_stream',
    'gcpds',
    'figurestream',
    'qt_material',
    'browser',
    'seaborn',
    'simple_pid',
    'pacman',
    'points',
    'tensorflow',
    'skimage',
    'cv2',
    'sklearn',
    'keras',
    'gdown',
    'tensorflow_datasets',
]