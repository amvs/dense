from setuptools import setup, find_packages

setup(
    name="dense",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # list your dependencies here, e.g.:
        # "numpy>=1.20",
    ],
)
