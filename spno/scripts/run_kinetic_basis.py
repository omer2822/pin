"""Exploratory K0 basis surgery: low-band distillation, optional head-only training.

The physical rate is used only by evaluation, never as a fit target. Source checkpoints
and the learned local operator are preserved, including the source kinetic zero mode.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from spno.artifacts import file_digest
from spno.checkpoints import load_checkpoint_payload
from spno.config import DataConfig, config_hash
from spno.data.datasets import TrajectoryShard, generate_shard, shard_paths
from spno.data.generate import sample_initial_conditions, energy_fraction_above
from spno.domain import l2_mass
from spno.evaluation.phase7_probes import global_phase
from spno.evaluation.spectral import cascade_series
from spno.losses.relative_l2 import relative_l2_per_sample
from spno.precision import widen_to_double
from spno.solvers.split_step import SubsteppedReference
from spno.train import TrainConfig, train_one_step
from scripts.run_phase6 import _model_from_checkpoint

DEFAULTS = dict(seeds=[0], probe_seeds=[1000], probe_batch=8, degrees=[1, 3],
                fit_k=8, parameter_points=9, head_epochs=0, head_pairs=8192,
                head_lr=1e-4, device='cpu', rollout_steps=200, stride=25,
                bandwidths=[8, 12, 16], reference_substeps=32,
                reference_tolerance=1e-4, tail_threshold=1e-6,
                cascade_floor=1e-8, dispersion_tolerance=.1,
                g4_tolerance=.1, g9_tolerance=.25, g1_ratio_guard=1.1,
                g1_absolute_floor=1e-3, smoke=False)


class PolynomialKinetic(nn.Module):
    """c0_source(alpha,beta) + sum_j c_j(alpha,beta) (k^2 / fit_k^2)^j.

    The coefficient head sees ONLY alpha,beta, with a learned affine map. There is
    no supplied alpha*k^2 feature or prescribed physical coefficient. The fixed scale
    improves conditioning and remains fixed when extrapolating in k.
    """
    def __init__(self, source, degree, fit_k):
        super().__init__()
        self.domain = source.domain
        self.degree = degree
        self.mode = f'poly{degree}'
        self.offset_source = copy.deepcopy(source)
        self.offset_source.requires_grad_(False)
        self.register_buffer('k_squared', source.k_squared.detach().clone())
        self.register_buffer('basis_scale', torch.tensor(float(fit_k ** 2), dtype=torch.float64))
        self.coefficients = nn.Linear(2, degree).double()

    def forward(self, parameters):
        q = self.k_squared / self.basis_scale
        basis = torch.stack([q ** j for j in range(1, self.degree + 1)], dim=-1)
        offset = self.offset_source(parameters)[:, :1]
        return offset + self.coefficients(parameters) @ basis.T


@torch.no_grad()
def fit_polynomial(source, cfg, *, degree=1, fit_k=8, parameter_points=9):
    if degree not in (1, 2, 3) or not degree <= fit_k <= cfg.initial_bandwidth:
        raise ValueError('degree must be 1–3; degree <= fit_k <= training bandwidth')
    if parameter_points < 2:
        raise ValueError('need at least two parameter points per axis')
    source = widen_to_double(source).eval()
    repaired = copy.deepcopy(source)
    repaired.requires_grad_(False)
    head = PolynomialKinetic(source.kinetic, degree, fit_k)
    a = torch.linspace(*cfg.alpha_range, parameter_points, dtype=torch.float64)
    b = torch.linspace(*cfg.beta_range, parameter_points, dtype=torch.float64)
    parameters = torch.cartesian_prod(a, b)
    # Evaluate the teacher only on the training-support band. No exact-PDE labels.
    q = source.kinetic.k_squared[:fit_k + 1] / float(fit_k ** 2)
    basis = torch.stack([q ** j for j in range(1, degree + 1)], dim=-1)
    p = torch.cat([parameters, torch.ones(len(parameters), 1, dtype=torch.float64)], dim=1)
    design = torch.einsum('pi,kj->pkji', p, basis).reshape(-1, degree * 3)
    teacher = source.kinetic(parameters)
    target = (teacher[:, :fit_k + 1] - teacher[:, :1]).reshape(-1)
    solution = torch.linalg.lstsq(design, target, driver='gelsd').solution.reshape(degree, 3)
    head.coefficients.weight.copy_(solution[:, :2])
    head.coefficients.bias.copy_(solution[:, 2])
    repaired.kinetic = head
    rmse = float((design @ solution.reshape(-1) - target).square().mean().sqrt())
    return repaired.eval(), {'degree': degree, 'fit_k': fit_k, 'fit_rmse': rmse,
                             'coefficient_weight': solution[:, :2].tolist(),
                             'coefficient_bias': solution[:, 2].tolist(),
                             'labels': 'centered frozen K0 rates, low band only'}


def load_sources(source_root, cfg, seeds):
    result, provenance = {}, []
    candidates = sorted((Path(source_root) / 'checkpoints/phase6').rglob('C1-seed*.pt'))
    for seed in seeds:
        matches = [(p, load_checkpoint_payload(p)) for p in candidates
                   if p.name == f'C1-seed{seed}.pt']
        matches = [(p, v) for p, v in matches if v.metadata.model_name == 'C1'
                   and v.metadata.data_hash == config_hash(cfg)]
        if len(matches) != 1:
            raise ValueError(f'Need one base C1 seed {seed} for data {config_hash(cfg)}; found {len(matches)}')
        path, payload = matches[0]
        if payload.metadata.architecture.get('kinetic_mode', 'K0') != 'K0':
            raise ValueError('This experiment requires a K0 source')
        model = _model_from_checkpoint(payload, cfg, expected_name='C1')
        model.load_state_dict(payload.state_dict, strict=True)
        result[seed] = widen_to_double(model).eval()
        provenance.append({'seed': seed, 'path': str(path), 'sha256': file_digest(path),
                           'metadata': asdict(payload.metadata)})
    return result, provenance


def fine_tune_head(model, source_root, cfg, options, seed):
    paths = shard_paths(Path(source_root) / 'data', config_hash(cfg))
    if not all(paths[s].exists() for s in ('train', 'val')):
        raise FileNotFoundError('Head fine-tuning requires original train.pt and val.pt in SOURCE_ROOT/data')
    shards = {s: TrajectoryShard.load(paths[s]) for s in ('train', 'val')}
    if any(shards[s].split != s or shards[s].dt != cfg.dt for s in shards):
        raise ValueError('Training shard split or dt mismatch')
    model = copy.deepcopy(model).float()
    model.requires_grad_(False)
    model.kinetic.coefficients.requires_grad_(True)
    history = train_one_step(model, shards['train'], shards['val'], cfg,
                            TrainConfig(epochs=options['head_epochs'], batch_size=256,
                                        learning_rate=options['head_lr'], weight_decay=0,
                                        patience=options['head_epochs'], seed=seed,
                                        max_train_pairs=options['head_pairs'], device=options['device']))
    return widen_to_double(model).eval(), history.as_dict()


@torch.no_grad()
def endpoint(model, shard, cfg, steps):
    field = shard.trajectories[:, 0]
    for _ in range(steps):
        field = model(field, shard.potential, shard.alpha, shard.beta, cfg.dt)
    return field


@torch.no_grad()
def state_errors(pred, target, domain):
    phase = global_phase(pred, target, domain)
    aligned = pred * torch.exp(-1j * phase[:, None])
    return {'raw': float(relative_l2_per_sample(pred, target, domain).mean()),
            'aligned': float(relative_l2_per_sample(aligned, target, domain).mean())}


@torch.no_grad()
def dispersion(model, cfg, options):
    k = model.kinetic.k_squared[:cfg.grid_size // 2 + 1].sqrt()
    records = []
    for alpha in sorted(set((*cfg.alpha_range, sum(cfg.alpha_range) / 2))):
        for beta in sorted(set((*cfg.beta_range, sum(cfg.beta_range) / 2))):
            p = torch.tensor([[alpha, beta]], dtype=torch.float64)
            rate = model.kinetic(p)[0, :len(k)]
            rate = rate - rate[0]
            exact = -alpha * k.square()  # evaluation only
            relative = (rate - exact).abs() / exact.abs().clamp_min(1e-12)
            bad = torch.nonzero((k > 0) & (relative > options['dispersion_tolerance'])).flatten()
            records.append({'alpha': alpha, 'beta': beta, 'rate': rate.tolist(),
                            'relative': relative.tolist(), 'knee': int(bad[0]) if len(bad) else None,
                            'ood_max_relative': float(relative[options['fit_k'] + 1:].max())})
    return {'k': k.tolist(), 'curves': records}


def run_check(source_root, output_root, cfg=None, options=None):
    cfg = cfg or DataConfig()
    options = {**DEFAULTS, **(options or {})}
    if not 1 <= options['stride'] <= options['rollout_steps'] or options['rollout_steps'] % options['stride']:
        raise ValueError('rollout_steps must be a positive multiple of stride')
    if not options['degrees'] or not options['seeds'] or not options['probe_seeds']:
        raise ValueError('degrees, seeds and probe_seeds must be nonempty')
    if not any(b > cfg.initial_bandwidth for b in options['bandwidths']):
        raise ValueError('G4 requires at least one bandwidth above training support')
    torch.set_num_threads(2)
    sources, provenance = load_sources(source_root, cfg, options['seeds'])
    identity = {'data': asdict(cfg), 'options': options, 'source': provenance,
                'experiment_sha256': file_digest(Path(__file__))}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(output_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    models, fits, curves = {}, {}, {}
    for seed, source in sources.items():
        models[seed] = {'K0-tanh': source}
        for degree in options['degrees']:
            arm = f'poly{degree}'
            model, fit = fit_polynomial(source, cfg, degree=degree, fit_k=options['fit_k'],
                                        parameter_points=options['parameter_points'])
            fits[f'{seed}/{arm}'] = fit
            models[seed][arm] = model
            if options['head_epochs']:
                tuned, history = fine_tune_head(model, source_root, cfg, options, seed)
                models[seed][arm + '-tuned'] = tuned
                fits[f'{seed}/{arm}-tuned'] = {'history': history}
        for arm, model in models[seed].items():
            curves[f'{seed}/{arm}'] = dispersion(model, cfg, options)
            if arm != 'K0-tanh':
                torch.save({'state_dict': model.state_dict(), 'identity': identity,
                            'seed': seed, 'arm': arm, 'fit': fits.get(f'{seed}/{arm}'),
                            'loader': 'Rebuild with fit_polynomial(source, cfg, degree, fit_k); then load state_dict'},
                           root / f'{arm}-seed{seed}.pt')
    rows, cascades = [], {}
    for probe_seed in options['probe_seeds']:
        references = {}
        for bandwidth in sorted(set([cfg.initial_bandwidth, *options['bandwidths']])):
            probe_cfg = replace(cfg, n_train=0, n_val=0, n_test=options['probe_batch'],
                                steps=options['rollout_steps'], initial_bandwidth=bandwidth,
                                substeps=options['reference_substeps'], seed=probe_seed)
            shard = generate_shard(probe_cfg, 'test')
            refined = endpoint(SubsteppedReference(cfg.domain, 2 * options['reference_substeps']),
                               shard, cfg, options['rollout_steps'])
            temporal = state_errors(shard.trajectories[:, -1], refined, cfg.domain)['raw']
            tail = float(energy_fraction_above(shard.trajectories.flatten(0, 1), cfg.domain,
                                               .75 * cfg.max_wave_number).max())
            references[bandwidth] = (shard, refined, temporal, tail)
        # G9: match the repository cascade arm: V=0, alpha=.9, beta=.3, training bandwidth.
        generator = torch.Generator().manual_seed(probe_seed + 30_000)
        initial = sample_initial_conditions(cfg.domain, options['probe_batch'], cfg.initial_bandwidth,
                                           cfg.mass_range, generator)
        potential = torch.zeros_like(initial.real)
        a = torch.full((len(initial),), .9, dtype=torch.float64)
        b = torch.full((len(initial),), .3, dtype=torch.float64)
        args = (cfg.domain, initial, potential, a, b, cfg.dt)
        kw = dict(steps=options['rollout_steps'], stride=options['stride'], cutoff=cfg.initial_bandwidth)
        reference = cascade_series(SubsteppedReference(cfg.domain, 2 * options['reference_substeps']), *args, **kw)
        coarse = cascade_series(SubsteppedReference(cfg.domain, options['reference_substeps']), *args, **kw)
        ref_spec = np.asarray(reference['spectra'])
        ref_fraction = np.asarray(reference['fraction_above_cutoff'])
        g9_convergence = float(np.linalg.norm(np.asarray(coarse['spectra']) - ref_spec) / np.linalg.norm(ref_spec))
        g9_tail = float((ref_spec[:, np.asarray(reference['wave_numbers']) > .75 * cfg.max_wave_number].sum(1)
                         / ref_spec.sum(1)).max())
        cascades[f'reference/{probe_seed}'] = reference
        for seed, arms in models.items():
            for arm, model in arms.items():
                print(f'Evaluate seed={seed}, probe={probe_seed}, {arm}', flush=True)
                g4 = {}
                for bandwidth, (shard, target, temporal, tail) in references.items():
                    pred = endpoint(model, shard, cfg, options['rollout_steps'])
                    g4[str(bandwidth)] = {**state_errors(pred, target, cfg.domain),
                                          'reference_error': temporal, 'reference_tail': tail,
                                          'mass_drift': float((l2_mass(pred, cfg.domain) /
                                                              l2_mass(shard.trajectories[:, 0], cfg.domain) - 1).abs().max())}
                series = cascade_series(model, *args, **kw)
                cascades[f'{seed}/{arm}/{probe_seed}'] = series
                spec = np.asarray(series['spectra'])
                fraction = np.asarray(series['fraction_above_cutoff'])
                complete = series['steps'] == reference['steps']
                spectrum_error = float(np.linalg.norm(spec - ref_spec) / np.linalg.norm(ref_spec)) if complete else None
                fraction_error = float(np.max(np.abs(fraction - ref_fraction)) /
                                       max(float(ref_fraction.max()), options['cascade_floor'])) if complete else None
                signal = float(ref_fraction.max())
                trustworthy = (all(v['reference_error'] <= options['reference_tolerance'] and
                                   v['reference_tail'] <= options['tail_threshold'] for v in g4.values())
                               and g9_convergence <= options['reference_tolerance']
                               and g9_tail <= options['tail_threshold'])
                dispersion_ok = max(c['ood_max_relative'] for c in curves[f'{seed}/{arm}']['curves']) <= options['dispersion_tolerance']
                g4_ok = all(v['aligned'] <= options['g4_tolerance'] for bw, v in g4.items()
                            if int(bw) > cfg.initial_bandwidth)
                g9_ok = complete and spectrum_error <= options['g9_tolerance'] and fraction_error <= options['g9_tolerance']
                control = g4[str(cfg.initial_bandwidth)]
                baseline_control = (control if arm == 'K0-tanh' else next(
                    r['G4'][str(cfg.initial_bandwidth)] for r in rows
                    if r['seed'] == seed and r['probe_seed'] == probe_seed and r['arm'] == 'K0-tanh'))
                g1_ok = all(control[m] <= max(options['g1_absolute_floor'],
                                             options['g1_ratio_guard'] * baseline_control[m])
                            for m in ('raw', 'aligned'))
                verdict = ('SMOKE ONLY' if options['smoke'] else
                           'INCONCLUSIVE: reference gate failed' if not trustworthy else
                           'INCONCLUSIVE: reference cascade too weak' if signal < options['cascade_floor'] else
                           'candidate passes quick thresholds' if dispersion_ok and g1_ok and g4_ok and g9_ok else
                           'does not pass quick thresholds')
                rows.append({'seed': seed, 'probe_seed': probe_seed, 'arm': arm, 'G4': g4,
                             'G9_spectrum_error': spectrum_error, 'G9_fraction_error': fraction_error,
                             'G9_reference_signal': signal, 'G9_reference_error': g9_convergence,
                             'G9_reference_tail': g9_tail, 'dispersion_ok': dispersion_ok, 'verdict': verdict})
                rows[-1]['G1_guard_pass'] = g1_ok
    summary = {**identity, 'run_id': run_id, 'exploratory': True, 'fits': fits,
               'dispersion': curves, 'rows': rows, 'cascades': cascades,
               'limitation': 'Post-hoc teacher distillation; reference gates use time refinement and a tail screen, '
                             'not a full spatial convergence study. Paired from-scratch training is needed '
                             'to attribute a repair to the basis during original training.'}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return root, summary


def plot_results(root, summary):
    import matplotlib.pyplot as plt
    root = Path(root)
    seed = summary['options']['seeds'][0]
    probe = summary['options']['probe_seeds'][0]
    arms = [r['arm'] for r in summary['rows'] if r['seed'] == seed and r['probe_seed'] == probe]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for arm in arms:
        record = summary['dispersion'][f'{seed}/{arm}']
        curve = record['curves'][len(record['curves']) // 2]
        k = np.asarray(record['k'])
        axes[0].plot(k, curve['rate'], label=arm)
    axes[0].plot(k, -curve['alpha'] * k ** 2, 'k--', label='exact (evaluation only)')
    axes[0].axvline(summary['options']['fit_k'], color='grey', linestyle=':')
    axes[0].set(title='Centered kinetic rate', xlabel='|k|', ylabel='κ(k) − κ(0)')
    for row in summary['rows']:
        if row['seed'] == seed and row['probe_seed'] == probe:
            axes[1].plot([int(b) for b in row['G4']], [v['aligned'] for v in row['G4'].values()],
                         marker='o', label=row['arm'])
    axes[1].set(title='G1 / G4 endpoint error', xlabel='Initial bandwidth', ylabel='Aligned relative L2')
    for arm in ['reference', *arms]:
        key = f'reference/{probe}' if arm == 'reference' else f'{seed}/{arm}/{probe}'
        series = summary['cascades'][key]
        axes[2].plot(np.asarray(series['steps']) * summary['data']['dt'], series['fraction_above_cutoff'], label=arm)
    axes[2].set(title='G9 nonlinear cascade', xlabel='t', ylabel='Energy fraction above training bandwidth')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    path = root / 'basis_quick_check.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
