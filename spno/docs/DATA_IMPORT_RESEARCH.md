# External PDE data research and import plan

Date: 2026-08-31

## Bottom line

Do not import PDEBench wholesale. It is large, it has no nonlinear Schrodinger or
wave-equation dataset, and several of its datasets do not test the structural claims
made by SPNO.

The smallest scientifically coherent first tranche is:

1. `Heat_4096.h5` + `Heat_256.h5` from the Temporal PDEs release (~435 MB total).
2. `KdV_train_4096.h5` + `KdV_valid_256.h5` from Hamiltonian Neural PDE Solvers
   (~3.6 GB total), initially downloading only the 211 MB validation file for an audit.
3. The processed MDBench archive (476 MB) only for its small external NLS trajectory,
   as a numerical sanity check rather than as training data.
4. One PDEBench advection shard and one Burgers shard only after the loader and native
   conservation audit exist. Each shard is 7.7 GB.

This creates a useful progression:

`NLS (Hamiltonian) -> KdV (Hamiltonian) -> Heat (dissipative) -> KS (mixed)`.

Wave data should be a separate second tranche. First generate a controlled 1D periodic
wave dataset with the complete canonical state `(q, p)`. Then, after adding 2D and
multi-channel support, use PDEGym `Wave-Gauss -> Wave-Layer` as the external transfer
test. The public PDEGym files contain displacement-like `solution` snapshots but not an
explicit velocity channel, so they are not suitable for a one-frame Markov model or a
clean energy/symplectic claim without reconstructing the state.

## What the repository can consume today

The current code is not a generic PDE benchmark runner:

- `StepOperator` consumes a complex field, a real potential, and the NLS-specific
  scalar parameters `alpha` and `beta`.
- `TrajectoryShard` stores complex NLS trajectories in one in-memory PyTorch object.
- The learned FNO and split-step models are currently one-dimensional.
- The evaluation code assumes NLS notions of mass, energy, reversibility, dispersion,
  and potential.

Relevant local contracts are
[`src/spno/models/base.py`](../src/spno/models/base.py) and
[`src/spno/data/datasets.py`](../src/spno/data/datasets.py).

Therefore, merely converting every external HDF5 file to `.pt` would be misleading.
KdV, heat, and wave require equation-specific dynamics and structural laws, while the
large files require lazy HDF5/NetCDF access rather than full-memory deserialization.

## Recommended datasets

| Priority | Source | PDE / role | Size and shape | Why import it | Main caveat |
|---|---|---|---|---|---|
| P0 | MDBench / PDE-FIND | NLS external diagnostic | processed archive 475.9 MB; NLS itself is a small real/imag trajectory | Checks that conclusions are not artifacts of the local generator | Model-discovery data, not enough independent trajectories for neural-operator training |
| P1 | Temporal PDEs | 1D heat | train 409.65 MB + valid 25.61 MB; 4096/256 trajectories, 250 x 100, `nu in [0.1,0.8]` | Clean, cheap dissipative counterpart with parameter variation | Only train/valid are published; derive validation IDs from train and reserve the source `valid` split as test |
| P1 | Hamiltonian Neural PDE Solvers | 1D KdV | train 3.38 GB + valid 211 MB; 4096/256 trajectories, 200 x 256 | Closest external Hamiltonian benchmark and a direct structure-preserving competitor | Source solver has measurable Hamiltonian drift; the release supplies a low-drift trajectory list |
| P2 | PDEBench | 1D advection | 7.7 GB per parameter file, 47 GB for the family | Recognizable conservative linear control and FNO benchmark | The official numerical generator is dissipative; native drift must be measured first |
| P2 | PDEBench | viscous Burgers | 7.7 GB per viscosity file, 93 GB for the family | Controlled negative case for an incorrectly imposed norm constraint | Dissipative; NLS mass projection is deliberately wrong here |
| P2 | PDEArena | 1D parameterized KS | 3.92 GB; 256 grid points; 4096 train and 512 test trajectories | Strong spectral and misspecification test with variable dissipation | A different structural model is needed; not a direct SPNO transfer |
| P3 | PDEGym / Poseidon | 2D Wave-Gauss and Wave-Layer | 11.7 GB and 15.2 GB; 10512 x 15 x 128 x 128 each | Excellent coefficient-field topology shift: smooth Gaussian medium to layers | Scalar snapshots only, sparse time axis, 2D, CC-BY-NC-4.0 |
| Defer | The Well | acoustic scattering and broad physical systems | 15 TB collection; relevant acoustic subsets start at 157 GB | Full pressure, velocity and material fields in a shared schema | Far beyond the present model and compute/storage scope |

Primary sources:

- PDEBench [official repository](https://github.com/pdebench/PDEBench),
  [download table](https://github.com/pdebench/PDEBench/blob/main/pdebench/data_download/README.md),
  and [DaRUS dataset v8 / DOI](https://doi.org/10.18419/DARUS-2986).
- Temporal PDEs [dataset card](https://huggingface.co/datasets/ayz2/temporal_pdes),
  [official implementation](https://github.com/anthonyzhou-1/temporal_pdes), and its
  source [Masked Autoencoders are PDE Learners](https://github.com/anthonyzhou-1/mae-pdes).
- Hamiltonian Neural PDE Solvers [dataset card](https://huggingface.co/datasets/ayz2/hamiltonian_pdes),
  [official code](https://github.com/anthonyzhou-1/hamiltonian_pdes), and
  [NeurIPS 2025 paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/2cb47172fab526fddec0cb12155cb2b1-Paper-Conference.pdf).
- PDEArena [official repository](https://github.com/pdearena/pdearena) and
  [KS dataset](https://huggingface.co/datasets/phlippe/Kuramoto-Sivashinsky-1D).
- PDEGym [official overview](https://camlab-ethz.github.io/poseidon/),
  [Wave-Gauss](https://huggingface.co/datasets/camlab-ethz/Wave-Gauss), and
  [Wave-Layer](https://huggingface.co/datasets/camlab-ethz/Wave-Layer).
- The Well [collection overview](https://polymathic-ai.org/the_well/datasets_overview/)
  and [data API](https://polymathic-ai.org/the_well/api/).
- MDBench [official repository](https://github.com/gryaklab/mdbench) and
  [dataset DOI](https://doi.org/10.5281/zenodo.17611099).

## PDEBench: import and skip decisions

The current DaRUS release is version 8, CC BY 4.0, with 336 HDF5 files. The canonical
layout is `[batch, time, x1, ..., xd, variables]`. The repository recommends direct
file downloads, not downloading the complete Dataverse release.

### Import later

**1D advection, one or three parameter files.** Start with `beta=0.4`. If the loader and
audit pass, add two files that create an interpolation/extrapolation split rather than
downloading all 47 GB. This is useful for a standard benchmark comparison, but it does
not validate an NLS-specific phase architecture.

**1D Burgers, one or three viscosity files.** Start with `nu=0.01`, then add a lower and
higher viscosity only if the research question is the transition from near-conservative
to dissipative dynamics. It is a negative control for hard L2 preservation.

**1D periodic compressible Navier-Stokes, only as a stretch goal.** It has the full
conserved fluid state and can support mass/momentum/energy diagnostics, but individual
files are roughly 12--25 GB and the model must become multi-channel.

### Skip now

- **Diffusion-sorption (4 GB):** cheaper than most PDEBench tasks but less clean than a
  parameterized heat equation for studying contractive dynamics.
- **1D reaction-diffusion (62 GB) / 2D reaction-diffusion (13 GB):** reaction terms
  confound the pure diffusion law, and the 1D release is unnecessarily large.
- **Shallow water (6.2 GB):** the PDEBench release stores only height for the radial dam
  break task, so a one-frame state is non-Markov and momentum/energy cannot be audited.
- **Darcy (6.2 GB):** steady operator, not an autoregressive evolution problem.
- **2D incompressible Navier-Stokes (2.3 TB):** out of scope for the current hardware and
  research question.
- **2D/3D CFD (551/285 GB):** architecture and storage mismatch.

The official download table reports family sizes of 47 GB (advection), 93 GB (Burgers),
88 GB (1D CFD), 4 GB (diffusion-sorption), 62 GB (1D reaction-diffusion), 13 GB (2D
reaction-diffusion), 551 GB (2D CFD), 285 GB (3D CFD), 6.2 GB (Darcy), 2.3 TB
(incompressible NS), and 6.2 GB (shallow water).

## Heat-equation track

The recommended external heat dataset is the `Heat` portion of Temporal PDEs:

- `train/Heat_4096.h5`: 409,654,344 bytes.
- `valid/Heat_256.h5`: 25,608,264 bytes.
- State shape per trajectory: 250 time points by 100 spatial points.
- Diffusivity: `nu ~ Uniform(0.1, 0.8)`.
- Periodic 1D domain in the upstream MAE-PDE generator.
- HDF5 keys follow `u`, `x`, `t`, and coefficient arrays.

This is better than treating PDEBench reaction-diffusion as "heat". A pure heat equation
provides a precise structural target:

- the spatial mean is conserved for periodic, unforced dynamics;
- every nonzero Fourier mode decays as `exp(-nu * k^2 * dt)`;
- L2 norm and Dirichlet energy are non-increasing;
- the solution map is contractive and forms a semigroup;
- backward stepping is ill-conditioned, not a symmetry to enforce.

Consequently, the existing NLS L2-mass projection is a deliberately misspecified heat
model. A good heat experiment compares:

1. plain FNO;
2. FNO plus the wrong fixed-L2 projection;
3. a spectral contractive model whose multipliers are constrained to `[0, 1]`;
4. a soft monotonicity penalty.

Report one-step error, long rollout error, monotonic-law violations, empirical spectral
decay versus `nu*k^2`, and generalization to unseen `nu`, `dt`, bandwidth, and grid.

## Wave-equation track

For a structure-preserving wave model the state must include both displacement and
velocity, or an equivalent canonical pair. A single displacement frame is insufficient
because the wave equation is second order in time.

### Stage W0: controlled 1D data

Generate a small periodic 1D benchmark using the public MIT-licensed MAE-PDE wave
generator or an independently validated spectral reference. Store `(q, p)` explicitly,
sample the wave speed, use trajectory-level train/validation/test splits, and retain a
substepped/high-resolution reference. This is the fastest route to a valid energy and
time-reversal experiment.

Required metrics are total energy, symplectic/reversibility error, phase speed and
dispersion, rollout error, and resolution transfer. For a spatially varying speed,
record the exact energy convention and boundary conditions in the manifest.

### Stage W1: external 2D transfer

Use PDEGym `Wave-Gauss` for training and `Wave-Layer` for zero-shot or few-shot testing.
Both have 10,512 trajectories, 15 uniform snapshots, 128x128 grids, and a per-trajectory
wave-speed field `c(x,y)`. Their official split is 10,212/60/240.

This tests a meaningful distribution shift in the *topology of the coefficient field*,
not just a scalar coefficient. However, it should only be run after:

- a 2D, multi-channel operator exists;
- the state uses two frames or a reconstructed velocity channel;
- native energy drift from the source data is measured;
- the non-commercial data license is acceptable for the intended output.

The Well acoustic-scattering data is a cleaner full-state physical benchmark because it
contains pressure, velocity, density, and sound-speed fields. It is deferred because the
smallest relevant subsets are 157--311 GB and use discontinuous media and non-periodic
boundaries.

## Canonical external-data contract

Add a generic contract alongside, not inside, the NLS-specific `TrajectoryShard`:

```text
state              [trajectory, time, channel, *space]
coordinates        one array per spatial axis
times              [time] or [trajectory, time]
static_fields       named tensors, e.g. potential or wave speed
parameters          named scalar/vector coefficients per trajectory
equation_id         canonical equation and convention
boundary_condition named BC plus parameters
split               source train/valid/test
trajectory_id       immutable source ID
source              URL, DOI, version/revision, license, citation
checksum            SHA-256/MD5 for every raw file
native_format       dtype, shape, units, channel names
structure_contract invariants, monotone quantities, reversibility expectation
```

Implementation principles:

- Stream HDF5/NetCDF lazily; do not convert multi-GB sources into monolithic `.pt` files.
- Keep raw files immutable and gitignored; commit only manifests, adapters, and audit
  summaries.
- Preserve source trajectory splits. If a source has no test split, split by trajectory
  ID once, persist the assignment, and never split frames.
- Keep a native-resolution view and a budget view (64 or 128 points). Create the budget
  view with anti-aliased Fourier truncation for periodic data, not naive decimation.
- Normalize using training trajectories only and persist the statistics.
- Make `h5py` and `netCDF4` optional data-import dependencies, not core runtime
  requirements.

## Mandatory audit before training

Every imported source gets a machine-readable audit report:

1. Verify checksum, source revision, license, shapes, fields, coordinates, dtype, and
   monotone time axis.
2. Verify that train/validation/test trajectory IDs are disjoint.
3. Measure native one-step and rollout statistics before resampling.
4. Measure every claimed invariant or monotone law on the source data itself.
5. Estimate the source solver's numerical floor using convergence information when
   available; otherwise label it unknown.
6. Compare native and budget-resolution trajectories for aliasing and loss of invariant
   fidelity.
7. Refuse a structural claim when the stored state is non-Markov or omits variables
   required by the conservation law.

This matters in two known cases:

- PDEBench advection is conservative as a continuous PDE, but the released numerical
  generator introduces dissipation.
- The Hamiltonian KdV release explicitly notes that its solver does not perfectly
  conserve the nonlinear Hamiltonian and supplies a list of 2,048 lower-drift samples.
  Report results on the full official validation set as well as any filtered subset; do
  not present the filtered result alone.

## Download and experiment sequence

### Phase D0 -- adapter and metadata only

- Add optional HDF5 support, generic schema, manifest validator, and lazy trajectory
  loader.
- Add equation-specific structure contracts for NLS, KdV, heat, and wave.
- Unit-test with tiny synthetic HDF5 fixtures; download no large data yet.

### Phase D1 -- cheap audits (under 1 GB)

- Download `Heat_256.h5` (25.6 MB).
- Download `KdV_valid_256.h5` (211 MB) and the supplied KdV Hamiltonian-drift index.
- Download MDBench `processed.zip` (475.9 MB), extract only the clean NLS file if the
  archive layout permits, and retain archive checksum/provenance.
- Run schema, split, spectrum, invariant, and temporal-resolution audits.

### Phase D2 -- first training tranche (~4.0 GB additional)

- Download `Heat_4096.h5` (409.7 MB).
- Download `KdV_train_4096.h5` (3.38 GB).
- Train heat and KdV only after equation-specific models and matched baselines exist.
- Use three seeds and equalized parameter/compute budgets.

### Phase D3 -- misspecification boundary

- Add PDEArena KS (3.92 GB).
- Bin results by viscosity and test where a conservative inductive bias changes from
  helpful to harmful.
- Alternatively, procedurally generate matched advection/KdV/Burgers scenarios with
  [APEBench](https://github.com/tum-pbs/apebench), which provides 46 periodic PDE
  scenarios and a pseudo-spectral ETDRK reference without a large static download.

### Phase D4 -- community anchor

- Download one PDEBench advection file and one Burgers file (~15.4 GB total), not the
  full families.
- Only add more parameter files if the first audit passes and the experiment requires a
  coefficient interpolation/extrapolation split.

### Phase D5 -- wave transfer

- Generate and validate 1D `(q,p)` wave data first.
- Add PDEGym Wave-Gauss only after 2D support.
- Add Wave-Layer only as the held-out medium-topology shift.

## Research positioning

A stronger paper is not "SPNO on many random PDE datasets." The coherent question is:

> When does an architectural structural bias help, how does it transfer between
> conservative PDEs, and how does performance fail as the true dynamics become
> dissipative or the assumed state is incomplete?

The four equation families expose different structural laws:

| PDE | Correct structural target | Useful failure mode |
|---|---|---|
| NLS | complex phase geometry, L2 mass, reversibility, symplectic split | coefficient/potential misspecification |
| KdV | Hamiltonian/integral invariants with nonlinear transport and dispersion | source-solver Hamiltonian drift |
| Heat | contractive semigroup, monotone norm/energy, spectral decay | hard norm conservation is wrong |
| Wave | canonical `(q,p)`, energy, reversibility, symplecticity | displacement-only data is non-Markov |

For the current thesis, the minimum convincing extension is **NLS + heat**, because it
tests the positive and negative sides of structure with little new data. The strongest
single-paper extension is **NLS + KdV + heat/KS**. Wave should become a separate chapter
or follow-up unless a shared reversible/contractive operator family is designed first.
