Installation
============

Prerequisites
-------------

Before installing wetting-angle-kit, ensure you have **Python 3.10 or higher**
installed on your system.

Optional Dependencies Strategy
------------------------------

OVITO and ASE are only imported inside the respective parser classes. Installing the package
without extras keeps dependencies minimal. Calling an OVITO/ASE parser without installing
raises a clear ImportError with installation instructions.

Installation Options
--------------------

Core (only for xyz file analysis)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install wetting-angle-kit

With OVITO
^^^^^^^^^^

.. code-block:: bash

   pip install wetting-angle-kit[ovito]

With ASE
^^^^^^^^

.. code-block:: bash

   pip install wetting-angle-kit[ase]

All optional dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install wetting-angle-kit[all]

Install OVITO
^^^^^^^^^^^^^

OVITO must be installed using pip:

.. code-block:: bash

   pip install ovito==3.11.3
