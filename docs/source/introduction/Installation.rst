Installation
============

Prerequisites
-------------

Before installing wetting_angle_kit, ensure you have the following prerequisites:

1. **Python 3.9 or higher**: Make sure you have Python 3.9 or higher installed on your system.
2. **Conda**: Ensure you have Conda installed. If not, you can install it from `here <https://docs.conda.io/en/latest/miniconda.html>`_.

## Optional Dependencies Strategy
OVITO and ASE are only imported inside the respective parser classes. Installing the package
without extras keeps dependencies minimal. Calling an OVITO/ASE parser without installing
raises a clear ImportError with installation instructions.

Installation Options
--------------------

Core (only for xyz file analysis)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install wetting_angle_kit

With OVITO
^^^^^^^^^^

.. code-block:: bash

   pip install wetting_angle_kit[ovito]

With ASE
^^^^^^^^

.. code-block:: bash

   pip install wetting_angle_kit[ase]

All optional dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install wetting_angle_kit[all]

Install OVITO
^^^^^^^^^^^^^

OVITO must be installed using the following Conda command:

.. code-block:: bash

   conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.11.3
