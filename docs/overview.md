# Overview

`rapids-choir` implements the unimodal RNA workflow from CHOIR v0.3.0 for
`AnnData`. It prepares RNA inputs, builds a strict hierarchy, prunes that
hierarchy with GPU comparisons, and returns labels with provenance. Its scope
follows.

## Scope

Version 0.1 runs one `AnnData` object on one NVIDIA L40S. The workflow starts
with `rapids_choir.pp.prepare_rna`, continues with
`rapids_choir.tl.choir`, and has no CPU analysis backend.

Within this workflow, corrected rules repair captured reference quirks.
Compatibility rules preserve frozen R operation order for parity.
Neither ruleset claims numerical identity between ranger and cuML. That
limitation bounds the evidence.

## Evidence boundary

Accordingly, the implementation and local tests establish only software
contracts. A scientific-parity claim requires one unchanged candidate, the
frozen runtime, a real L40S, and the complete confirmation workflow. The
current candidate has not passed that workflow.
