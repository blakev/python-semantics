#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# https://github.com/blakev/python-semantics
#
# >>
#
#   python-semantics, 2019
#
# <<

import re
import setuptools

with open("README.md", "r") as fp:
    description = fp.read()

with open("semantics.py", "r") as fp:
    version = re.search(r'^__version__ = "([\w\d.]+)"', fp.read(), re.M + re.I)
    version = version.groups()[0]

# setup
setuptools.setup(
    name='version-semantic',
    version=version,
    author='Blake VandeMerwe',
    author_email='blakev@null.net',
    description="Semantic version parsing and comparison library.",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/blakev/python-semantics",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            'semver = semantics'
        ]
    },
    classifiers=[
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only"
    ]
)
