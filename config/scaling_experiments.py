# This file defines a list of experiments to run on the tiny_stories dataset.
# It relies on train_tinystories.py for baseline parameters, and overrides them.

from math import isclose

# We want to leverage config/train_tinystories.py for baseline run parameters.
# We won't `exec` it here; run_experiments.py will handle that by including "config/train_tinystories.py" as a base.
# We'll just define the experiments list and logic for architecture selection.

def determine_arch_and_variant(n_dims):
    first_dim = n_dims[0]
    all_equal = all(d == first_dim for d in n_dims)

    if all_equal:
        model_architecture = 'original'
        shape_variant = 'symmetry'
    else:
        model_architecture = 'diamond'
        shape_variant = 'stay_wide'
    return model_architecture, shape_variant

dims_list = [
    [768]*12,
    [384]*12,
    [512]*12,
    [640]*12,
    [896]*12,
]

experiments = []
for dims in dims_list:
    arch, variant = determine_arch_and_variant(dims)

    # Create a short run name, e.g. "original_768" or "original_512"
    # If all dims equal, just pick the dimension once for run_name.
    # If diamond, you could pick a descriptive name like "diamond_stay_wide_384to768" etc.
    # Here all are equal dims, so just use the first dim:
    base_dim = dims[0]
    run_name = f"{arch}_{variant}_{base_dim}"

    # Sanity checks
    if arch == 'original' and not all(d == dims[0] for d in dims):
        raise ValueError("For original architecture, all n_dims must be equal!")
    if arch == 'diamond' and all(d == dims[0] for d in dims):
        raise ValueError("For diamond architecture with custom dims, vary them to make sense.")

    exp = {
        "model_architecture": arch,
        "shape_variant": variant,
        "n_dims": dims,
        "max_iters": 2000000,
        "lr_decay_iters": 2000000,
        "run_name": run_name,  # short and descriptive
    }
    experiments.append(exp)
