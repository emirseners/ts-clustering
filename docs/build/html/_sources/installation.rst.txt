Installation
============

Installing via pip
------------------

To install the clusterer package from the Python package index use the pip command:

.. code-block:: bash

   pip install simgon

Installing with conda
---------------------

To install clusterer with conda, using the conda-forge channel, use the following command:

.. code-block:: bash

   conda install -c conda-forge simgon

Required Dependencies
--------------------

Clusterer requires **Python X.X** or above.

Clusterer builds on the core Python data analytics stack, and the following third party libraries:

* Numpy >= 1.24.0
* Scipy >= 1.16.0
* Pandas >= 2.0.0
* Numba >= 0.57.0

These modules should build automatically if you are installing via pip. If you are building from source, or if pip fails to load them, they can be loaded with the same pip syntax as above.

Optional Dependencies
---------------------

For enhanced plotting capabilities:

* Matplotlib >= 3.7.0

For using `PySD <https://pysd.readthedocs.io/en/master/index.html>`_ for simulation:

* PySD >= 3.0.0