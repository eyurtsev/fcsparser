import fnmatch
import os

from setuptools import setup, find_packages

import versioneer


def get_fcs_files():
    matches = []
    for root, dirnames, filenames in os.walk('fcsparser'):
        for filename in fnmatch.filter(filenames, '*.fcs'):
            matches.append(os.path.join(root, filename))
    return matches


with open('README.rst', 'r') as f:
    README_content = f.read()

setup(
    name='fcsparser',
    packages=find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='A python package for reading raw fcs files',
    author='Eugene Yurtsev',
    author_email='eyurtsev@gmail.com',
    url='https://github.com/eyurtsev/fcsparser',
    keywords=['flow cytometry', 'data analysis', 'cytometry', 'parser', 'data'],
    license='MIT',

    install_requires=[
        "setuptools",
        "six",
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
