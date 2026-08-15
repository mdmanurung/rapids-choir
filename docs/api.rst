API reference
=============

The public API imports without initializing CUDA. GPU libraries load when a
GPU entry point runs.

Configuration
-------------

.. autoclass:: rapids_choir.RFProfile
   :members:

.. autoclass:: rapids_choir.ComparisonConfig
   :members:

.. autoclass:: rapids_choir.RNAConfig
   :members:

.. autoclass:: rapids_choir.ChoirConfig
   :members:

Preprocessing
-------------

.. autofunction:: rapids_choir.pp.prepare_rna

.. autofunction:: rapids_choir.pp.validate_prepared_rna

Analysis
--------

.. autofunction:: rapids_choir.tl.choir

.. autofunction:: rapids_choir.tl.compare_clusters

.. autoclass:: rapids_choir.tl.ChoirComparator
   :members:

Results
-------

.. autoclass:: rapids_choir.ChoirResult
   :members:

.. autoclass:: rapids_choir.ComparisonResult
   :members:

.. autoclass:: rapids_choir.HierarchyTree
   :members:

.. autoclass:: rapids_choir.HierarchyNode
   :members:

.. autofunction:: rapids_choir.get.records

Enums
-----

.. autoclass:: rapids_choir.RuleSet
   :members:

.. autoclass:: rapids_choir.Multiplicity
   :members:

.. autoclass:: rapids_choir.InputPolicy
   :members:

.. autoclass:: rapids_choir.Decision
   :members:

.. autoclass:: rapids_choir.ReasonCode
   :members:
