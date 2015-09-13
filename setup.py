from setuptools import setup, find_packages
import re

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
    ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],

    long_description=README_content,
)
