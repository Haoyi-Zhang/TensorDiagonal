# Diagonal completion and corruption certificates

This standalone repository accompanies **Diagonal Completion of Orthogonal Tensors: Certificates and Corruption Distances**. It contains exact rational implementations, complete mathematical arguments, all constructed inputs and certificate records, and deterministic reproduction commands. It does not depend on a neighboring paper directory or a network service.

## Mathematical scope

The clean object is a real symmetric third-order orthogonally decomposable tensor, allowing ranks zero through the ambient dimension. All mixed entries are fixed in the exact model; only coordinate-diagonal entries can suffer arbitrary additive corruption. The complete fiber is affine. Its gain-graph kernel yields exact support-family and cardinality correction criteria. There is a sharp dimension boundary: no consistent mixed fiber is rigid for dimensions one or two, while continuous random orthogonal factors and weights are rigid almost surely from dimension three onward. Consistency forces exceptional Cauchy-clique free components, and the constructions realize every distance at full tensor rank. Pointwise lists and a conditional noisy posterior bound are included.

Commuting-slice equivalence, support-induced linearization in related recovery problems, and general minimum-distance reasoning have prior literature; they are not asserted as new universal techniques. The project manuscript contains a 64-work relevance-first bibliography; the standalone proof packet retains a focused 17-work map, with every retained entry cited in its text. The tensor basis is fixed. Arbitrary whitening, nonorthogonal CP factors, rational eigenvectors, production infrastructure, and statistical coverage are not certified. The noisy verifier assumes that a compatible truth exists and that the supplied noise bounds are valid. It cannot establish these hypotheses from the observations alone.

## Run from this directory

A POSIX Python 3 installation with the standard library is sufficient for the scientific code. There are no packages to install, external APIs, data downloads, GPUs, or worker pools. The campaign rejects Python optimized mode (`-O` or `-OO`) at startup because it deliberately uses assertions as fail-closed scientific invariants; a subprocess regression checks this behavior.

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python run.py --suite all --out results
PYTHONDONTWRITEBYTECODE=1 python check.py --results results
PYTHONDONTWRITEBYTECODE=1 python export_figures.py --results results --out results/plot-data
```

Each of `exhaustive`, `structured`, `decoding`, `noise`, and `mutations` can replace `all`. The `all` route runs them sequentially in fresh interpreters, one at a time; child interpreters use Python's `-S` flag so the declared standard-library-only calculation cannot inherit host-specific site hooks. This is process isolation for deterministic resource release, not parallelism or a worker pool. A suite writes in a temporary directory and installs its files only after successful completion. An interrupted nonempty working directory is deliberately not erased automatically; inspect its contents before removing that one directory and retrying. The input design is fixed in `inputs/design.json`, not selected through the command line. The implementation caps dimension at 16 where tensor inputs are decoded and enumerated support families at 2,048. The distance and Cauchy series reach dimension eight, the separate distance-spectrum regression reaches dimension nine, deterministic generic-rigidity witnesses reach dimension twelve, and noisy certificates reach dimension seven.

`check.py` independently rechecks saved exact and noisy certificates and their reported truths. It reconstructs every Cartesian case from its family/index identifier and rejects missing or duplicate indices. For all 94 structured cases it reconstructs the declared tensor, binds every truth mixed orbit to the certificate input, checks dimension, parameters, tensor rank and fiber dimension, and computes distance from independently recovered disjoint free components rather than from an arbitrary saved basis. It also requires the noise certificates, CSV and declared $9\times4$ design to have the same unique keys and labels. The producer in `run.py` already compares the decoder with a finite generating oracle; the saved-record checker separately rebuilds the two fixed fibers and replays all 270 observations and 1,566 budgets without importing the production decoding routine. These are finite checks, not a machine-checked proof of arbitrary-dimensional claims.

The saved-control path requires the exact ten unique mutation names, literal Boolean rejection flags, and matching summary counts. It separately reconstructs valid rational fixtures and exercises each named rejection operation; the saved mutation format contains names and flags, not the altered certificates themselves. Both negative-control records must retain their refusal status and reason, and their singular-restriction or excessive-noise premise is checked from independent equations.

## Expected scientific results

| Suite | Fixed evidence |
|---|---|
| Mixed-input enumeration | 83,348 records: 741 unique, 772 ambiguous, 81,835 inconsistent; includes every binary mixed tensor in dimension four |
| Structured tensors | 94: 36 distance realizations, 42 Cauchy cases, five rigid and one ambiguous Householder case, and ten exact all-distinct-entry rigidity witnesses |
| Pointwise decoding | 270 words, 1,566 budget comparisons, one normalization counterexample |
| Noisy certification | 36 accepted conditional bounds, two informative refusals; identical candidates and radii across four gross amplitudes within each fixed noise configuration |
| Certificate mutations | Ten rejected alterations |
| Boundary and adversarial unit tests | Twenty-nine current tests, including three portable support-reconstruction tests plus proof-critical regressions and saved-result rejection tests for false component distance, mismatched truth/input orbits, duplicate or missing noise keys, mislabeled noise configurations, altered observations, swapped case labels, altered budget counts, missing or duplicated mutation names, non-Boolean rejection flags, mismatched mutation summaries, and altered negative-control status or reasons |


The saved-result regression subset can be run directly:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_saved_results -v
```

The current subset contains five test methods. They accept the unmodified 94 structured records, 36 noise records, 270 decoding records, ten mutation records and two negative controls, then reject the specified alterations. The original generation oracle in `run.py` remains part of the campaign; this regression subset checks the separately implemented saved-record path.

The modal decoder reconstructs its estimate only on each free component's listed support, preserving the full dense directions and certificate schema. Disjoint supports make this the same exact completion as the dense coordinate sum; it avoids arithmetic on zero coordinates but does not change elimination, consistency, ratio normalization, modal ties or list formulas. No timing gain is claimed. The three tests in `tests/test_support_reconstruction.py` enumerate the two fixed fibers' 270 observations and 1,566 budgets, check direct commutators/dense reconstruction, and cover singleton, rigid, inconsistent and signed/rescaled cases. They are discovered by existing scientific CI. A portable standard-library-only subset is `python -B -m unittest discover -s tests -p test_support_reconstruction.py -v`; it requires no saved results or private paths. Retained historical counts/resources and scientific outputs are unchanged.

The two three-dimensional enumeration families overlap; the dimension-four family is separate. The total is a number of tested records, not random samples or an empirical prevalence estimate. Raw tensor entries and certificates are retained in `results/`, including negative cases. Historical POSIX command measurements are recorded in `results/reproduction.json`; they predate the two additional saved-control regression methods. Wall/CPU times and peak resident memory are environment-specific observations, not stable outputs or speed claims. Scientific JSON and rational CSV values are deterministic under the documented design. Byte-identical PDFs are not promised because typesetting metadata can change.

The scientific workflow in `.github/workflows/scientific-checks.yml` is configured for a flat artifact-repository root on Ubuntu 24.04. It runs the tests, regenerates all five finite suites, independently checks the new results, exports the plots, and compares deterministic outputs with the retained records. It bounds the complete calculation and uploads raw output even after a failed gate. A configured workflow is not evidence that a hosted run has completed.

## File roles and input format

`src/exact.py` produces the affine system, exact witnesses, gain graph and structured tensors. `src/verify.py` does not import it: the verifier forms slice products and obtains coefficient rows by exact finite differences. `src/decoding.py` calculates modal completions and finite/infinite list sizes. `src/stability.py` generates rational inverse witnesses and posterior radii. `check.py` reads saved evidence; `export_figures.py` derives every numerical table/plot value from it.
`claim_evidence_ledger.csv` maps material scientific claims to proofs and finite checks. `external_resources.csv` records scholarly and official resources and their acquisition/usage modes. `reference_audit.csv` is the one-row-per-citation bibliography audit: it records the canonical source, verification depth, manuscript role, and 2026-09-16 recheck status, while explicitly distinguishing full-text/theorem inspection from metadata-and-abstract verification.

An input stores an integer `n` and a list of `orbits`, in lexicographic order of triples `i <= j <= k` with **zero-based** indices. Entries are integer values or rational strings, not floating-point numbers. The completion operation discards supplied pure-diagonal entries when forming the mixed tensor. Positive certificates include a particular diagonal, null basis, and rank minor; negative certificates include a sparse left-null inconsistency witness. No tolerance decides these identities. The mathematics uses real factors even when the certificate is rational.

## Standalone proof supplement

The general proofs and worked examples are under `proofs/`; all their inputs and references are local. They contain the same scientific arguments as the article, with independent section numbering. To typeset the supplement, use a standard LaTeX installation with the supplied JMLR style and its ordinary dependencies, TikZ and PGFPlots:

```sh
cd proofs
pdflatex -interaction=nonstopmode -halt-on-error theory.tex
pdflatex -interaction=nonstopmode -halt-on-error theory.tex
```

`references.bib` is the focused 17-work bibliography source record; `references.tex` is the corresponding explicit natbib bibliography used for the offline build, and every retained entry is cited. BibTeX is not required. The supplied publisher style is unchanged. Compiling this supplement proves only that the text typesets; the mathematical proofs remain human-readable arguments, not a proof-assistant development.

## Trust, licensing, and external use

The accepted identities rely on Python integer/fraction arithmetic and the verifier implementation. The verifier is separately implemented within the same research effort, not an independent human or blind review. It does not promise resistance to all malicious input or resource exhaustion. The finite checks do not establish practical data relevance or general scientific priority.

The MIT text applies to original implementation material as a proposed project license. Third-party style material is excluded and keeps its upstream notices; see `THIRD_PARTY_NOTICES.md`. Scholarly sources were consulted, not copied into implementation internals. All synthetic scientific inputs were generated by the included formulas.
