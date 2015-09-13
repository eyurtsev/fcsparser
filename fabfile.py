"""Fabric file."""
import base64
import os
import json
import urllib2

from contextlib import contextmanager

from fabric.api import local, lcd, abort
from fabric.decorators import task

BUILD_DIRS = (
    "dist",
    "doc/build",
    "build",
    "fcsparser.egg-info",
)

SDIST_RST_FILES = (
    "README.rst",
)

SDIST_TXT_FILES = [os.path.splitext(x)[0] + ".txt" for x in SDIST_RST_FILES]

@task
def clean():
    """Clean build files."""
    for build_dir in list(BUILD_DIRS):
        local("rm -rf %s" % build_dir)


###############################################################################
# PyPI
###############################################################################
@contextmanager
def _dist_wrapper():
    """Add temporary distribution build files (and then clean up)."""
    try:
        # Copy select *.rst files to *.txt for build.
        for rst_file, txt_file in zip(SDIST_RST_FILES, SDIST_TXT_FILES):
            local("cp %s %s" % (rst_file, txt_file))

        # Perform action.
        yield
    finally:
        # Clean up temp *.txt files.
        for rst_file in SDIST_TXT_FILES:
            local("rm -f %s" % rst_file, capture=False)


@task
def sdist():
    """Package into distribution."""
    with _dist_wrapper():
        local("python setup.py sdist", capture=False)


@task
def pypi_register():
    """Register and prep user for PyPi upload.

    .. note:: May need to weak ~/.pypirc file per issue:
        http://stackoverflow.com/questions/1569315
    """
    with _dist_wrapper():
        local("python setup.py register", capture=False)


@task
def pypi_upload():
    """Upload package."""
    with _dist_wrapper():
        local("python setup.py sdist upload", capture=False)
