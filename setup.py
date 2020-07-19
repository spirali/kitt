import os

from setuptools import find_packages, setup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def read(fname):
    return open(os.path.join(ROOT_DIR, fname)).read()


with open("requirements.txt") as reqs:
    requirements = [line.strip() for line in reqs.readlines()]

setup(
    name="kitt",
    version="0.0.1",
    author="KITT team",
    description="Machine learning toolkit",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,
    long_description=read('README.md'),
)
