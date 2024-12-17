import unittest
from utils.diamond_dim_utils import calculate_diamond_dims

class TestDiamondArchitecture(unittest.TestCase):
    def test_symmetric_diamond(self):
        n_layer = 12
        base_dim = 384
        max_dim = 1024
        head_dim = 64
        dims = calculate_diamond_dims(n_layer, base_dim, max_dim, head_dim)
        # symmetrical check
        self.assertEqual(dims, dims[::-1])

    def test_divisibility_by_head_dim(self):
        n_layer = 12
        base_dim = 384
        max_dim = 1024
        head_dim = 64
        dims = calculate_diamond_dims(n_layer, base_dim, max_dim, head_dim)
        for dim in dims:
            self.assertEqual(dim % head_dim, 0, f"Dimension {dim} not divisible by {head_dim}")

if __name__ == '__main__':
    unittest.main()
