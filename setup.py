# -*- coding: utf-8 -*-
"""Setup module."""
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def get_requires():
    """Read requirements.txt."""
    requirements = open("requirements.txt", "r").read()
    return list(filter(lambda x: x != "", requirements.split()))


def read_description():
    """Read README.md and CHANGELOG.md."""
    try:
        with open("README.md") as r:
            description = "\n"
            description += r.read()
        with open("CHANGELOG.md") as c:
            description += "\n"
            description += c.read()
        return description
    except Exception:
        return '''ParsiPy: NLP Toolkit for Historical Persian Texts in Python'''


setup(
    name='parsipy',
    packages=[
        'parsipy', ],
    version='0.1',
    description='ParsiPy: NLP Toolkit for Historical Persian Texts in Python',
    long_description=read_description(),
    long_description_content_type='text/markdown',
    author='ParsiPy Development Team',
    author_email='parsipy@openscilab.com',
    url='https://github.com/openscilab/parsipy',
    download_url='https://github.com/openscilab/parsipy/tarball/v0.1',
    keywords="nlp persian text",
    project_urls={
            'Source': 'https://github.com/openscilab/parsipy',
    },
    install_requires=get_requires(),
    include_package_data=True,
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Manufacturing',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    license='MIT',
)
