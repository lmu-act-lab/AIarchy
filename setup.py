from setuptools import setup, find_packages

setup(
    name="AIarchy",
    version="0.1",
    packages=find_packages(include=['src', 'src.*']),
    package_dir={'': '.'},
    python_requires=">=3.7",
)