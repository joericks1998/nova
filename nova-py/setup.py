from setuptools import setup, find_packages

setup(
    name = "nova_py",
    version = "0.2.1",
    packages = find_packages(where='src'),
    package_dir = {'': 'src'},
    install_requires = [
        "numpy",
        "tensorflow",
        "Levenshtein"
    ]
)
