def calculate_diamond_dims(n_layer, base_dim, max_dim, head_dim):
    """
    Calculate diamond-shaped dimensions for transformer layers.

    Parameters:
        n_layer (int): Total number of layers.
        base_dim (int): Dimension at the smallest layers.
        max_dim (int): Dimension at the largest (midpoint) layers.
        head_dim (int): Must be divisible by head_dim for attention heads.

    Returns:
        list[int]: List of dimensions for each layer.
    """
    mid = (n_layer - 1) // 2  # Middle index for symmetry
    dims = []
    for i in range(n_layer):
        if i <= mid:
            target_dim = base_dim + (i * (max_dim - base_dim) // mid)
        else:
            target_dim = base_dim + ((n_layer - 1 - i) * (max_dim - base_dim) // mid)

        # Round to the nearest multiple of head_dim
        actual_dim = round(target_dim / head_dim) * head_dim
        dims.append(max(base_dim, min(actual_dim, max_dim)))

    return dims
