import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "light-compositing",
    version = "0.0.1",
    author = "Tjaart van der Walt",
    author_email = "python@tjaart.co.za",
    description = ("An implementation of a Light Compositing article (see Readme)"),
    license = "MIT",
    keywords = "light compositing",
    url = "https://github.com/tjaartvdwalt/light-compositing",
    packages=['light-compositing'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: MIT License",
    ],
)
