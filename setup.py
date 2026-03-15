from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="mlops-smartphone-addiction-detection",
    version="1.0",
    author="Sanket Lawande",
    packages=find_packages(),
    install_requires = requirements,
)