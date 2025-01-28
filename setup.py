# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from setuptools import setup, find_packages



with open("README.md", "r") as f:
    long_description = f.read()


init_str = Path("mbrl/__init__.py").read_text()
version = init_str.split("__version__ = ")[1].rstrip().strip('"')

setup(
    name="mbrl",
    version=version,
    author="Bernd Frauenknecht, and Devdutt Subhasish",
    description="Submission code for Infoprop Paper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    zip_safe=False,
)