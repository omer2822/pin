"""Phase adapters used by both notebooks and explicit CLI stages."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from .artifacts import atomic_json, verified_copy
from .config import DataConfig, config_hash
from .experiments import pick_device
from .misspecification import MisspecificationConfig
from .train import TrainConfig
from .workflow import Workflow, stable_hash


def evaluate_phase(workflow: Workflow, phase: int, *, seeds=None, device='cpu',
                   lambdas=(0., .01, .1, 1., 10.), fractions=(.05, .1, .25, .5, 1.),
                   sigmas=(0., .1), gammas=(0., .001), noise=(0., .001, .01, .05),
                   grid=128, arms=None, allow_budget_bound=False, train_config=None,
                   c1g_lambdas=()):
    seeds = workflow.seeds if seeds is None else list(seeds)
    if not seeds:
        raise ValueError('At least one seed is required')
    device = pick_device(device)
    horizon = min(100, workflow.data_config.steps)
    settings = dict(phase=phase, seeds=seeds, device=device, lambdas=list(lambdas),
                    fractions=list(fractions), sigmas=list(sigmas), gammas=list(gammas),
                    noise=list(noise), grid=grid, arms=arms, horizon=horizon,
                    allow_budget_bound=allow_budget_bound,
                    requested_train_config=asdict(train_config) if train_config else None)
    if c1g_lambdas:
        # Only when requested, so pre-C1g report identities are unchanged.
        settings['c1g_lambdas'] = list(c1g_lambdas)
    payload = {'phase': phase, 'data_hash': config_hash(workflow.data_config),
               'seeds': seeds, 'quick': workflow.quick, 'selected_horizon': horizon,
               'evaluation_config': settings, 'checkpoint_sources': {},
               'exploratory': workflow.quick, 'evaluation_datasets': {}}
    measured = {}

    def prepare(selected_phase, **kwargs):
        return workflow.prepare(selected_phase, seeds=seeds, generate=False, train_config=train_config, **kwargs)

    def remember(job, record):
        payload['evaluation_datasets'][job.spec['data_hash']] = {'test_sha256': workflow._digest(job.paths['test'])}
        payload['checkpoint_sources'][job.identifier] = {
            'path': record['path'], 'sha256': record['sha256'], 'spec': job.spec,
            'converged': record['converged'], 'source': record['source']}
        payload['exploratory'] |= not record['converged']

    def restore(job):
        model, record = workflow.restore(job, allow_budget_bound=allow_budget_bound)
        remember(job, record)
        return model.to(device), record

    def measure(job):
        if job.identifier not in measured:
            _, record = restore(job)
            measured[job.identifier] = workflow.evaluate(
                job, device=device, checkpoints=(horizon,), allow_budget_bound=allow_budget_bound)
        return measured[job.identifier]

    if phase == 7:
        from scripts import run_phase7 as runner
        jobs = prepare(7, lambdas=lambdas, c1g_lambdas=c1g_lambdas)
        # Preflight the entire selection before any expensive evaluation.
        for job in jobs:
            restore(job)
        def summarize(selected):
            per_seed = []
            for job in selected:
                metrics = measure(job)
                per_seed.append({'seed': job.seed, 'metrics': metrics,
                                 'history': metrics['history'], 'converged': metrics['converged'],
                                 'config': {'data': asdict(job.data_config), 'train': asdict(job.train_config)}})
            return {
                'physics_weight': selected[0].physics_weight, 'per_seed': per_seed,
                'parameter_count': per_seed[0]['metrics']['parameters'],
                **runner.aggregate_pino_seed_metrics(per_seed, horizon=horizon)}
        weights = sorted({job.physics_weight for job in jobs})
        families = ('A', 'C1', 'C1g') if any(j.name == 'C1g' for j in jobs) else ('A', 'C1')
        payload['sweeps'] = {
            name: {str(weight): summarize(selected)
                   for weight in weights
                   if (selected := [j for j in jobs if j.name == name and j.physics_weight == weight])}
            for name in families}
        payload['baselines'] = {name: payload['sweeps'][name]['0.0'] for name in families}
        payload['baselines']['B-loop'] = summarize([j for j in jobs if j.name == 'B-loop'])
        payload['sweep'] = payload['sweeps']['A']  # historical A-only report consumers
        payload['variant'] = ('7a-A-PINO-B-loop-C1-C1-PDE-C1g' if 'C1g' in families
                              else '7a-A-PINO-B-loop-C1-C1-PDE')
        payload['reference_lines'] = {'model_A_one_step': payload['baselines']['A']['one_step_mean']}
        payload['confound'] = (
            'The Crank–Nicolson residual conserves mass when solved exactly; a soft penalty '
            'does not guarantee exact conservation. Its finite-step dynamics differ from '
            'C1 split-step dynamics and the reference solver. Report both complete lambda sweeps.')
        payload['control_arm'] = (
            'A and C1 lambda=0 and B-loop reuse checkpoints only when data, architecture '
            'and training protocol match. Positive weights start from the same per-family '
            'seed initialization, not from fitted baseline weights. B-loop has no PDE penalty.')
        payload['comparison_complete'] = any(weight > 0 for weight in weights)
    elif phase == 8:
        from scripts import run_phase8 as runner
        config = replace(train_config or workflow.train_config or TrainConfig(), device=device)
        # Noise and resolution use original full-data base models independently of
        # the selected sample-efficiency fractions.
        base_jobs = prepare(8, fractions=(1.,))
        sample_jobs = prepare(8, fractions=fractions)
        for job in base_jobs + sample_jobs:
            restore(job)
        models = {seed: {} for seed in seeds}
        for job in base_jobs:
            models[job.seed][job.name] = restore(job)[0]
        samples = {}
        for fraction in fractions:
            selected = prepare(8, fractions=(fraction,))
            per_seed = {seed: {} for seed in seeds}
            histories = {}
            for job in selected:
                result = measure(job)
                per_seed[job.seed][job.name] = result
                histories[f'{job.seed}:{job.name}'] = {'history': result['history'], 'converged': result['converged']}
            samples[str(float(fraction))] = {
                'fraction': fraction, 'max_train_pairs': selected[0].train_config.max_train_pairs,
                'by_model': runner.aggregate_measurements(per_seed, horizon), 'training': histories}
        shards = workflow.shards()
        payload.update(runner.resolution_sections(models, shards, workflow.data_config, grid, config))
        args = argparse.Namespace(noise=noise, seeds=seeds, train_fractions=fractions)
        payload['robustness'] = runner.robustness_section(models, shards, workflow.data_config, config, args,
                                                         sample_measurements=samples)
        payload.update(grid=grid, checkpoint_models=list(runner.MODEL_NAMES))
    elif phase == 9:
        from scripts import run_phase9 as runner
        jobs = prepare(9, sigmas=sigmas, gammas=gammas)
        for job in jobs:
            restore(job)
        payload['parameter_counts'] = {j.name: restore(j)[0].parameter_count() for j in jobs}
        payload['gates'] = runner.bitwise_gate(workflow.data_config)
        payload['sweeps'], payload['crossover'] = {}, {}
        for dial, values in (('sigma', sigmas), ('gamma', gammas)):
            measurements = {}
            for value in values:
                spec = MisspecificationConfig(**{'nonlocal_sigma' if dial == 'sigma' else 'gain_loss_gamma': float(value)})
                identifier = spec.identifier(workflow.data_config)
                selected = [j for j in jobs if j.spec['data_hash'] == identifier]
                per_seed = {seed: {} for seed in seeds}
                for job in selected:
                    metrics = measure(job)
                    per_seed[job.seed][job.name] = {'metrics': metrics, 'converged': metrics['converged'],
                                                  'history': metrics['history'], 'checkpoint': metrics['checkpoint']}
                measurements[str(float(value))] = {
                    'value': value, 'identifier': identifier, 'config': spec.as_dict(),
                    'reference': spec.provenance(workflow.data_config),
                    'reuses_production_shards': spec.is_exact, 'seeds': seeds,
                    'note': 'Checkpoint-only measurements; the exact arm is shared across both sweeps.',
                    **runner.aggregate_seed_measurements(per_seed, selected_horizon=horizon)}
            payload['sweeps'][dial] = {'measurements': measurements}
            payload['crossover'][dial] = ({model: runner.locate_crossover(measurements, model)
                                           for model in runner.COMPARISON_MODELS} if measurements else {})
    elif phase == 6:
        from scripts import run_phase6 as runner
        from .checkpoints import load_checkpoint_payload
        selected_arms = list(runner.ALL_ARMS if arms is None else arms)
        if not selected_arms or any(a not in runner.ALL_ARMS for a in selected_arms):
            raise ValueError('Select valid Phase 6 arms')
        rows = [r for r in workflow.inventory() if r['seed'] in seeds]
        kinetics = {r['metadata']['architecture'].get('kinetic_mode', 'K0') for r in rows}
        if len(kinetics) != 1:
            raise ValueError('Phase 6 requires one kinetic mode per evaluation')
        kinetic = kinetics.pop()
        models = {seed: {} for seed in seeds}
        for row in rows:
            name = row['name']
            if not row['converged'] and not (workflow.quick or allow_budget_bound):
                raise RuntimeError(f'Phase 6 checkpoint is budget-bound: {row["path"]}')
            payload['exploratory'] |= not row['converged']
            payload['checkpoint_sources'][row['sha256']] = {'path': row['path'], 'sha256': row['sha256'],
                                                           'converged': row['converged']}
            if '/' not in name:
                cp = load_checkpoint_payload(Path(row['path']))
                model = runner._model_from_checkpoint(cp, workflow.data_config, expected_name=name)
                model.load_state_dict(cp.state_dict, strict=True)
                models[row['seed']][name] = model.eval().to(device)
        # Quick G arms generate missing test shards: keep them outside the imported
        # run.  Full runs only read shards, so they use the source directly.
        shift_root = workflow.source_root / 'data'
        if workflow.quick:
            shift_root = workflow.output_root / 'phase6-evaluation-data'
            for path in (workflow.source_root / 'data').rglob('*.pt'):
                verified_copy(path, shift_root / path.relative_to(workflow.source_root / 'data'))
        payload['gates'] = runner.run_gates(workflow.data_config.domain, workflow.data_config)
        payload['experiments'] = runner.run_selected_arms(
            selected_arms, workflow.data_config, seeds=seeds, kinetic=kinetic, device=device,
            quick=workflow.quick, checkpoint_root=workflow.source_root, shift_root=shift_root, models_by_seed=models,
            allow_budget_bound=workflow.quick or allow_budget_bound)
        payload.update(alpha_train_range=list(workflow.data_config.alpha_range),
                       cascade_cutoff=workflow.data_config.initial_bandwidth)
    else:
        raise ValueError('Expected phase 6, 7, 8 or 9')
    payload['identifier'] = f'phase{phase}-' + stable_hash({
        'settings': settings, 'checkpoints': payload['checkpoint_sources'],
        'evaluation_datasets': payload['evaluation_datasets'],
        'test': workflow._digest(workflow._paths(workflow.source_root / 'data', config_hash(workflow.data_config),
                                                 splits=('test',))['test'])})
    if payload['exploratory'] and not workflow.quick:
        payload['identifier'] += '-budget-bound'  # never mistaken for converged results
    output = workflow.output_root / 'reports' / payload['identifier']
    (output / 'plots').mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'metrics.json', payload)
    runner.make_plots(payload, output)
    payload['output'] = str(output)
    return payload


def probe_phase7(workflow: Workflow, jobs, *, long_lambdas=(0., .01), long_steps=5000, long_batch=32,
                 n_rollout=100, cn_trajectories=16, allow_budget_bound=False, verbose=True):
    """Phase 7 follow-up probes on existing checkpoints; nothing is trained.

    Every job gets its one-step distance to the exact CN step and a phase-aligned
    rollout against the stored test frames (both cheap).  Jobs whose lambda is in
    ``long_lambdas`` also get a ``long_steps`` invariant-only rollout.  The CN oracle
    and a single Strang step are measured the same way as reference lines.  Everything
    is float64 on CPU.  Writes its own report, named by the probe settings and
    checkpoint hashes, so it never alters the Phase 7 comparison it follows.
    """
    import time
    import torch
    from scripts import run_phase7 as runner
    from .evaluation.phase7_probes import (CrankNicolsonStep, aligned_rollout,
                                           long_horizon_invariants, reference_one_step)
    from .experiments import rollout_inputs
    from .precision import widen_to_double
    from .solvers.split_step import SplitStepNLSOperator

    if not jobs:
        raise ValueError('No Phase 7 jobs to probe')
    long_wanted = {float(w) for w in long_lambdas}
    data, domain = workflow.data_config, workflow.data_config.domain
    test = workflow.shards(jobs[0])['test']
    inputs = rollout_inputs(test, data, 'cpu', n=n_rollout, dtype=torch.complex128)
    long_inputs = {k: v[:long_batch] for k, v in inputs.items() if k != 'trajectories'}
    checkpoints = tuple(s for s in (1, 10, 20, 50, 100, 200) if s <= data.steps)
    settings = {'long_lambdas': sorted(long_wanted), 'long_steps': long_steps,
                'long_batch': long_batch, 'n_rollout': n_rollout,
                'cn_trajectories': cn_trajectories, 'checkpoints': list(checkpoints),
                'allow_budget_bound': allow_budget_bound}
    oracle = CrankNicolsonStep(domain)
    parameters = (test.potential.double(), test.alpha.double(), test.beta.double(), data.dt)
    subset = test.trajectories[:cn_trajectories].to(torch.complex128)

    def probe(model, *, long_horizon):
        # Distance to the exact CN step on the same inputs: whether a larger lambda
        # actually moves the model toward CN, not merely away from the data.
        measured = {
            'distance_to_cn': reference_one_step(model, domain, subset, *parameters,
                                                 chunk=4, against=oracle),
            'rollout': aligned_rollout(model, domain, inputs['initial'], inputs['trajectories'],
                                       inputs['potential'], inputs['alpha'], inputs['beta'],
                                       data.dt, checkpoints=checkpoints),
            'long_horizon': None}
        if long_horizon:
            measured['long_horizon'] = long_horizon_invariants(
                model, domain, long_inputs['initial'], long_inputs['potential'],
                long_inputs['alpha'], long_inputs['beta'], data.dt, steps=long_steps)
        return measured

    def log(message):
        if verbose:
            print(message, flush=True)

    references = {}
    full = test.trajectories.to(torch.complex128)
    for name, operator in (('CN exact', oracle), ('Strang 1-step', SplitStepNLSOperator(domain))):
        start = time.time()
        references[name] = {'one_step': reference_one_step(operator, domain, full, *parameters),
                            **probe(operator, long_horizon=True)}
        log(f'{name}: one-step {references[name]["one_step"]:.3e}, distance to CN '
            f'{references[name]["distance_to_cn"]:.3e} ({time.time() - start:.0f}s)')

    sources, arms = {}, {}
    for job in jobs:
        model, record = workflow.restore(job, allow_budget_bound=allow_budget_bound)
        sources[job.identifier] = {'sha256': record['sha256'], 'converged': record['converged'],
                                   'source': record['source']}
        arm = job.name if job.physics_weight == 0 else f'{job.name}+PDE'
        start = time.time()
        measured = probe(widen_to_double(model, device='cpu').eval(),
                         long_horizon=job.physics_weight in long_wanted)
        arms.setdefault(f'{arm}|{job.physics_weight:g}', {
            'arm': arm, 'family': job.name, 'lambda': job.physics_weight,
            'per_seed': {}})['per_seed'][str(job.seed)] = measured
        long = measured['long_horizon']
        log(f'{arm} lambda={job.physics_weight:g} seed={job.seed}: distance to CN '
            f'{measured["distance_to_cn"]:.3e}'
            + (f', long-horizon diverged_at={long["diverged_at"]}' if long else '')
            + f' ({time.time() - start:.0f}s)')

    exploratory = workflow.quick or not all(s['converged'] for s in sources.values())
    identifier = 'phase7-probes-' + stable_hash({
        'settings': settings, 'checkpoints': sources, 'test': workflow._digest(jobs[0].paths['test'])})
    if exploratory and not workflow.quick:
        identifier += '-budget-bound'
    payload = {'phase': 7, 'variant': '7a-probes', 'identifier': identifier,
               'data_hash': config_hash(data), 'settings': settings, 'exploratory': exploratory,
               'checkpoint_sources': sources, 'references': references,
               'arms': list(arms.values()),
               'note': ('float64 on CPU. distance_to_cn uses the first cn_trajectories test '
                        'trajectories; the aligned rollout uses the same n_rollout samples as the '
                        'Phase 7 table; long-horizon drift uses the first long_batch initial '
                        'conditions, with trends fitted beyond fit_from steps. The exact CN step '
                        'conserves discrete mass AND energy, so the residual pulls toward both.')}
    output = workflow.output_root / 'reports' / identifier
    (output / 'plots').mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'metrics.json', payload)
    runner.make_probe_plots(payload, output)
    payload['output'] = str(output)
    return payload


def add_workflow_arguments(parser):
    parser.add_argument('--stage', choices=('prepare', 'train', 'evaluate'), default=None,
                        help='explicit checkpoint workflow; omitted preserves legacy behavior')
    parser.add_argument('--source-root', type=Path, help='Phase 6 standalone artifact directory')
    parser.add_argument('--output-root', type=Path, help='separate persistent workflow directory')
    parser.add_argument('--cache-root', type=Path, default=None)
    parser.add_argument('--source-config', type=Path,
                        help='JSON with original data and train configurations; missing training settings remain unknown')
    parser.add_argument('--train-config', type=Path, help='JSON overrides for the requested training protocol (never redefines the source)')
    parser.add_argument('--allow-budget-bound', action='store_true', help='admit budget-bound checkpoints; results are labelled exploratory')


def parse_workflow_args(parser, argv=None):
    """Preserve legacy defaults, but inherit source seeds in explicit workflows."""
    import sys
    raw = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(raw)
    if args.stage is not None:
        if not any(token == '--seeds' or token.startswith('--seeds=') for token in raw):
            args.seeds = None
        if any(token == '--epochs' or token.startswith('--epochs=') for token in raw):
            raise ValueError('With --stage, use --train-config for new training overrides; --source-config declares the original run')
    return args


def run_stage(phase, args):
    if args.source_root is None or args.output_root is None:
        raise ValueError('--stage requires --source-root and --output-root')
    original = json.loads(args.source_config.read_text()) if args.source_config else {}
    data = DataConfig(**original['data']) if 'data' in original else None
    config = TrainConfig(**original['train']) if 'train' in original else None
    workflow = Workflow(args.source_root, args.output_root, data_config=data,
                        train_config=config, cache_root=args.cache_root)
    requested = None
    if args.train_config:
        if workflow.train_config is None:
            raise ValueError('Declare the original protocol with --source-config before overriding it')
        requested = replace(workflow.train_config, **json.loads(args.train_config.read_text()))
    options = {'seeds': args.seeds, 'train_config': requested}
    for cli, key in (('lambdas', 'lambdas'), ('c1g_lambdas', 'c1g_lambdas'), ('train_fractions', 'fractions'), ('sigmas', 'sigmas'), ('gammas', 'gammas')):
        if hasattr(args, cli):
            options[key] = getattr(args, cli)
    if args.stage == 'evaluate':
        for key in ('noise', 'grid', 'arms'):
            if hasattr(args, key):
                options[key] = getattr(args, key)
        return evaluate_phase(workflow, phase, device=args.device,
                              allow_budget_bound=args.allow_budget_bound, **options)
    if phase == 6:
        if args.stage == 'train':
            raise ValueError('Phase 6 is imported; use its existing trainer for new Phase 6 arms')
        return {'inventory': workflow.inventory()}
    jobs = workflow.prepare(phase, **options)
    if args.stage == 'train':
        workflow.train(jobs, device=pick_device(args.device))
    rows = workflow.status(jobs)
    print(json.dumps(rows, indent=2, default=str))
    return {'jobs': rows}
