import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="pymol-residue-selector",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="PyMOL plugin to automatically create selections for each unique residue type",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/residue-type-selector",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pymol',
    ],
    python_requires='>=3.6',
    entry_points={
        'pymol.plugins': [
            'residue_selector = residue_type_selector.core:__init_plugin__'
        ]
    }
)
