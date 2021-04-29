========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-snnnet/badge/?style=flat
    :target: https://readthedocs.org/projects/python-snnnet
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/ehthiede/python-snnnet.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/ehthiede/python-snnnet

.. |codecov| image:: https://codecov.io/github/ehthiede/python-snnnet/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/ehthiede/python-snnnet

.. |version| image:: https://img.shields.io/pypi/v/snnnet.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/snnnet

.. |wheel| image:: https://img.shields.io/pypi/wheel/snnnet.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/snnnet

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/snnnet.svg
    :alt: Supported versions
    :target: https://pypi.org/project/snnnet

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/snnnet.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/snnnet

.. |commits-since| image:: https://img.shields.io/github/commits-since/ehthiede/python-snnnet/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/ehthiede/python-snnnet/compare/v0.0.0...master



.. end-badges

Codes for neural net computation that are equivariant to permutation

* Free software: MIT license

Installation
============

::

    pip install -e .

You can also install the in-development version with::

    pip install https://github.com/ehthiede/python-snnnet/archive/master.zip


Documentation
=============


https://python-snnnet.readthedocs.io/


Training Models
===============

To train the models, make sure to acquire the required datasets (qm9, zinc).
After installing the package, the training scripts can be invoked in the following fashion.

::

    python -um snnnet.molecule.train --directory TRAIN_DIR --train_dataset TRAIN_DATASET --valid_dataset VALID_DATASET


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

Examples
========

Link-prediction on citation graph:
::
    cd examples/
    sh citation_link_prediction.sh

Link-prediction on molecules:
::
    cd examples/
    sh molecule_link_prediction.sh