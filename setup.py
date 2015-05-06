from glob import glob

from setuptools import setup

from light_compositing import __version__

setup(
    name="light-compositing",
    version=__version__,
    author="Tjaart van der Walt",
    author_email="pypi@tjaart.co.za",
    description=("An implementation based on the paper: Image Compositing for \
    Photographic Lighting."),
    scripts=glob("bin/*"),
    packages=["light_compositing"],
    license="MIT",
    keywords="light compositing",
    url="https://github.com/tjaartvdwalt/light-compositing",
    # long_description=read("This is a imp"),
    install_requires=["numpy", "scipy"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
    ],
)
