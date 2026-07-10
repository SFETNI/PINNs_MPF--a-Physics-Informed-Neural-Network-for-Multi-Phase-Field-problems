# B5 Triple-Junction 128 × 128 Reference

The accepted finite-difference reference dataset used for the B5 validation is packaged under:

`validated_benchmarks/B5_triple_junction/reference/`

The reference was generated during the development and validation campaign using the multiphase finite-difference implementation provided in:

`reference/pf_solver_multiphase.py`

This public release preserves the accepted reference fields and their validation evidence. It does not currently claim a one-command, bitwise regeneration path for the exact accepted reference campaign.

The packaged reference is used for benchmark evaluation, field comparison, topology analysis, junction-angle tracking, and reader-facing figures. The accepted PINNs-MPF protocol and its handoff assumptions are documented in the B5 benchmark README and in `TECHNICAL_REPORT.md`.