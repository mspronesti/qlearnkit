import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qlkit",
    version="0.1",
    author=["Massimiliano Pronesti",
            "Federico Tiblias",
            "Giulio Corallo"
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mspronesti/qlkit",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "qiskit"
    ],
    python_requires='>=3.6'
)
