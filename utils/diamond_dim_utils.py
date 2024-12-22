def calculate_diamond_dims(n_layer, base_dim, max_dim, head_dim, shape_variant='symmetry'):
    """
    Calculate dimensions based on shape_variant.
    shape_variant could be:
    - 'symmetry': symmetrical diamond (already implemented)
    - 'stay_wide': increase to max_dim, stay at max_dim for several layers, then decrease
    - 'kite': like a kite: grow, plateau, then shrink differently
    """
    mid = (n_layer - 1) // 2
    
    if shape_variant == 'symmetry':
        # existing logic (as before)
        dims = []
        for i in range(n_layer):
            if i <= mid:
                target_dim = base_dim + (i * (max_dim - base_dim) // mid)
            else:
                target_dim = base_dim + ((n_layer - 1 - i) * (max_dim - base_dim) // mid)
            actual_dim = round(target_dim / head_dim) * head_dim
            dims.append(max(base_dim, min(actual_dim, max_dim)))
        return dims

    elif shape_variant == 'stay_wide':
        # grow to max_dim by half, then stay at max_dim for some layers, then shrink
        # for simplicity, let's say we spend 1/4th layers growing, 1/2 staying, 1/4 shrinking
        quarter = n_layer // 4
        dims = []
        for i in range(n_layer):
            if i < quarter:
                # grow from base_dim to max_dim
                frac = i / quarter
                target_dim = base_dim + int(frac * (max_dim - base_dim))
            elif i < quarter + (n_layer // 2):
                # stay wide (max_dim)
                target_dim = max_dim
            else:
                # shrink back down
                pos_in_shrink = i - (quarter + (n_layer // 2))
                shrink_len = n_layer - (quarter + (n_layer // 2))
                frac = 1 - (pos_in_shrink / shrink_len)
                target_dim = base_dim + int(frac * (max_dim - base_dim))
            actual_dim = round(target_dim / head_dim) * head_dim
            dims.append(max(base_dim, min(actual_dim, max_dim)))
        return dims

    elif shape_variant == 'kite':
        # Example: Increase to max by mid, then gently slope down but not symmetrical
        # This is just an example pattern
        dims = []
        for i in range(n_layer):
            if i <= mid:
                # grow linearly to max_dim
                frac = i / mid
                target_dim = base_dim + int(frac * (max_dim - base_dim))
            else:
                # shrink more slowly
                steps_down = (n_layer - 1 - mid)
                frac = (i - mid) / steps_down
                target_dim = max_dim - int(frac * (max_dim - base_dim) * 0.5) # shrink less steeply
            actual_dim = round(target_dim / head_dim) * head_dim
            dims.append(max(base_dim, min(actual_dim, max_dim)))
        return dims

    else:
        raise ValueError(f"Unknown shape_variant: {shape_variant}")
