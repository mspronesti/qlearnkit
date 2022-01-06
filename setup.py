import os
import setuptools

# loading requirements from textfile
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# loading long description from readme
with open("README.md", "r") as f:
    long_description = f.read()


# loading version number from path
def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), encoding="utf-8") as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


version = get_version('qlkit/version.py')

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
