from distutils.core import setup
from setuptools import find_packages


setup(
    name="shelter",
    version="0.0.1",
    license="--",
    packages=find_packages(),
    package_data={
        '': ['*.svg', '*.yaml', '*.zip', '*.ico', '*.bat']
    }
)