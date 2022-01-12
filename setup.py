import setuptools

# loading requirements from textfile
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# loading long description from readme
with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="qlearnkit",
    author="Massimiliano Pronesti, "
           "Federico Tiblias, "
           "Giulio Corallo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mspronesti/qlearnkit",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.7'
)
