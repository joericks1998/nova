from setuptools import setup, find_packages

setup(
    name = "nova_py",
    version = "0.1.1",
    packages = find_packages(),
    install_requires = [
        "numpy",
        "tensorflow",
        "Levenshtein"
    ]
)
