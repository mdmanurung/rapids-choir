API reference
=============

The public API imports without initializing CUDA. CUDA libraries load only
when a GPU entry point runs.

R-compatible entry points
-------------------------

.. autofunction:: rapids_choir.CHOIR

.. autofunction:: rapids_choir.compareClusters

.. autofunction:: rapids_choir.getRecords

RNA preparation
---------------

.. autofunction:: rapids_choir.pp.prepare_rna

Results
-------

.. autoclass:: rapids_choir.ChoirResult
   :members:

.. autoclass:: rapids_choir.ComparisonResult
   :members:

Development profile
-------------------

.. autoclass:: rapids_choir.RFProfile
