import re
import fnmatch
import os

from setuptools import setup, find_packages


def get_package_version(path):
    '''Extracts the version'''
    with open(VERSION_FILE, "rt") as f:
        verstrline = f.read()

    VERSION = r"^version = ['\"]([^'\"]*)['\"]"
    results = re.search(VERSION, verstrline, re.M)

    if results:
        version = results.group(1)
    else:
        raise RuntimeError("Unable to find version string in {}.".format(path))

    return version


def get_fcs_files():
    matches = []
    for root, dirnames, filenames in os.walk('fcsparser'):
        for filename in fnmatch.filter(filenames, '*.fcs'):
            matches.append(os.path.join(root, filename))
    return matches


VERSION_FILE = "fcsparser/_version.py"

version = get_package_version(VERSION_FILE)

with open('README.rst', 'r') as f:
    README_content = f.read()

setup(
    name='fcsparser',
    packages=find_packages(),
    version=version,
    description='A python package for reading raw fcs files',
    author='Eugene Yurtsev',
    author_email='eyurtsev@gmail.com',
    url='https://github.com/eyurtsev/fcsparser',
    download_url='https://github.com/eyurtsev/fcsparser/archive/v{0}.zip'.format(version),
    keywords=['flow cytometry', 'data analysis', 'cytometry', 'parser', 'data'],
    license='MIT',

    install_requires=[
        "setuptools",
        'six',
        "numpy",
        "pandas"
    ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],

    long_description=README_content,
    include_package_data=True,

    package_data={
        'fcsparser': get_fcs_files(),
    },
)
