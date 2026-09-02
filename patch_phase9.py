import re

with open("spno/scripts/run_phase9.py", "r") as f:
    content = f.read()

# 1. Add imports
imports_to_add = """
import dataclasses
from pathlib import Path
from spno.experiments import load_shards, evaluate_model, DATA_ROOT
from spno.train import train_one_step
from spno.data.datasets import generate_shard, shard_paths, assert_no_leakage
"""
content = content.replace("from spno.train import TrainConfig\n", "from spno.train import TrainConfig\n" + imports_to_add)

# 2. Update locate_crossover to fail-closed
locate_crossover_new = """def locate_crossover(by_dial: dict, model: str) -> dict:
    \"\"\"Where a model's A-relative rollout error first reaches 1.0, or a bounded note.

    A located crossover is the result; a bounded one ("no crossover within the swept
    range") is also publishable.  The bounded form is emitted explicitly rather than
    leaving the key absent, so a reader cannot mistake "not found" for "not looked for".
    \"\"\"

    dials = sorted(float(key) for key in by_dial)
    ratios = [by_dial[str(d)].get("relative_to_A", {}).get(model) for d in dials]
    
    if not ratios or any(r is None for r in ratios):
        raise ValueError(f"Missing evaluation data for {model}; cannot draw conclusions about crossover.")

    for dial, ratio in zip(dials, ratios):
        if ratio >= 1.0:
            return {"crossover": dial, "bounded": None}
    return {
        "crossover": None,
        "bounded": f"no crossover within [{min(dials, default=0)}, "
        f"{max(dials, default=0)}]: {model} still beats A at every swept value",
    }"""
content = re.sub(r"def locate_crossover\(by_dial: dict, model: str\) -> dict:.*?    }", locate_crossover_new, content, flags=re.DOTALL)

# 3. Update the loop
loop_old = """    for dial, values in (("sigma", args.sigmas), ("gamma", args.gammas)):
        for value in values:
            spec = (
                MisspecificationConfig(nonlocal_sigma=value)
                if dial == "sigma"
                else MisspecificationConfig(gain_loss_gamma=value)
            )
            payload["sweeps"][dial][str(value)] = {
                "value": value,
                "identifier": spec.identifier(data_config),
                "reuses_production_shards": spec.is_exact,
                "config": spec.as_dict(),
                "seeds": seeds,
                "note": "retrain all five models on this shard set, evaluate, and "
                "record rollout error relative to A under 'relative_to_A'",
                "relative_to_A": {},
            }"""

loop_new = """    for dial, values in (("sigma", args.sigmas), ("gamma", args.gammas)):
        for value in values:
            spec = (
                MisspecificationConfig(nonlocal_sigma=value)
                if dial == "sigma"
                else MisspecificationConfig(gain_loss_gamma=value)
            )
            identifier = spec.identifier(data_config)
            
            # Load or generate shards
            try:
                shards = load_shards(data_config, identifier=identifier)
            except FileNotFoundError:
                paths = shard_paths(DATA_ROOT, identifier)
                paths["train"].parent.mkdir(parents=True, exist_ok=True)
                shards = {}
                reference = spec.reference(data_config)
                for split in ("train", "val", "test"):
                    print(f"generating {split} for {identifier} ...")
                    shard = generate_shard(
                        data_config,
                        split,
                        reference=reference,
                        reference_metadata=spec.provenance(data_config)
                    )
                    shard.save(paths[split])
                    shards[split] = shard
                assert_no_leakage(shards)
                
            # Train models and collect errors
            print(f"\\n--- {dial}={value} ({identifier}) ---")
            train_config = TrainConfig(
                epochs=epochs,
                batch_size=256,
                learning_rate=1e-3,
                patience=6,
                device=device,
            )
            
            # We track the rollout error at step 100 for each model and seed
            errors = {name: [] for name in builders.keys()}
            
            for seed in seeds:
                config_for_seed = dataclasses.replace(train_config, seed=seed)
                for name, build_fn in builders.items():
                    print(f"  training {name} (seed {seed})...")
                    model = build_fn()
                    # Re-seed exactly like the models do to ensure consistency
                    torch.manual_seed(seed) 
                    train_one_step(model, shards["train"], shards["val"], data_config, config_for_seed)
                    evaluated = evaluate_model(model, shards, data_config, config_for_seed, checkpoints=(100,))
                    # Index 0 corresponds to checkpoint 100 since it's the only one
                    rollout_error = evaluated["rollout"]["relative_error"][0]
                    errors[name].append(rollout_error)
            
            avg_errors = {name: sum(errs)/len(errs) for name, errs in errors.items()}
            relative_to_A = {
                name: avg_errors[name] / avg_errors["A"] 
                for name in builders.keys() if name != "A"
            }
            
            payload["sweeps"][dial][str(value)] = {
                "value": value,
                "identifier": identifier,
                "reuses_production_shards": spec.is_exact,
                "config": spec.as_dict(),
                "seeds": seeds,
                "note": "retrain all five models on this shard set, evaluate, and "
                "record rollout error relative to A under 'relative_to_A'",
                "relative_to_A": relative_to_A,
            }"""
content = content.replace(loop_old, loop_new)

with open("spno/scripts/run_phase9.py", "w") as f:
    f.write(content)
