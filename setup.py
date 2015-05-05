import os
from glob import glob

from setuptools import setup

from light_compositing import __version__


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="light-compositing",
    version=__version__,
    author="Tjaart van der Walt",
    author_email="pypi@tjaart.co.za",
    description=("An implementation of a Light Compositing article (see the README for mor details)"),
    scripts=glob("bin/*"),
    packages=["light_compositing"],
    license="MIT",
    keywords="light compositing",
    url="https://github.com/tjaartvdwalt/light-compositing",
    long_description=read("README"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: MIT License",
    ],
)
