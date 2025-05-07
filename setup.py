from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="HOTEL-RES-PROJECT",
    version="0.1",
    author="Shahil Shaikh",
    packages=find_packages(),
    install_requires=requirements,
)