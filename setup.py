import os
import setuptools

# loading requirements from textfile
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# loading long description from readme
with open("README.md", "r") as f:
    long_description = f.read()

# loading version number from path
VERSION_PATH = os.path.join(os.path.dirname(__file__), "qlkit", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    version = version_file.read().strip()

setuptools.setup(
    name="qlkit",
    version=version,
    author="Massimiliano Pronesti, "
           "Federico Tiblias, "
           "Giulio Corallo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mspronesti/qlkit",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.7'
)
