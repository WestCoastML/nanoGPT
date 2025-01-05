"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
import json
from ast import literal_eval

def safe_eval(val):
    """Safely evaluate a string that might be a list, dict, or primitive type."""
    try:
        # First try JSON parse
        return json.loads(val)
    except json.JSONDecodeError:
        # Then try Python literal_eval for non-JSON but valid Python literals
        try:
            return literal_eval(val)
        except (SyntaxError, ValueError):
            # Fallback to plain string if neither JSON nor literal_eval works
            return val

# Store sweep/command-line parameters first
sweep_params = {}
config_files = []

# First pass: collect all parameters and config files
for arg in sys.argv[1:]:
    # Strip leading '--' if present
    if arg.startswith('--'):
        arg = arg[2:]
    
    if '=' in arg:
        # It's a parameter override
        key, val = arg.split('=', 1)
        sweep_params[key] = safe_eval(val)
    else:
        # It's a config file
        config_files.append(arg)

# Add config_path from sweep params if it exists
if 'config_path' in sweep_params:
    config_files.append(sweep_params['config_path'])

# Process all config files first
for config_file in config_files:
    print(f"Overriding config with {config_file}:")
    with open(config_file) as f:
        config_content = f.read()
        print(config_content)
    exec(config_content, globals())
    # Store the last config file as config_path
    if config_file == config_files[-1]:
        globals()['config_path'] = config_file

# Now apply sweep/command-line parameters to override config file values
for key, val in sweep_params.items():
    if key == 'config_path':
        globals()['config_path'] = val  # Ensure config_path is set in globals
        continue
        
    if key not in globals():
        raise ValueError(f"Unknown config key: {key}. Available keys: {sorted(globals().keys())}")

    # Handle type conversion
    original_val = globals()[key]
    
    # Special handling for lists
    if isinstance(original_val, list) and isinstance(val, list):
        if original_val:  # If list is not empty
            elem_type = type(original_val[0])
            val = [elem_type(x) for x in val]
    else:
        # For non-list values, ensure the type matches
        if not isinstance(original_val, list):
            try:
                val = type(original_val)(val)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"Cannot convert value for '{key}' to type {type(original_val)}: {e}"
                )
    
    print(f"Overriding: {key} = {val}")
    globals()[key] = val

# Optional: Validate critical parameters
critical_params = ['base_dim', 'n_layer', 'head_dim']
for param in critical_params:
    if param in globals():
        globals()[param] = int(globals()[param])