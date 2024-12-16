import unittest
from model import GPT, GPTConfig

class TestModelIntegrity(unittest.TestCase):
    def test_model_initialization(self):
        from utils.diamond_dim_utils import calculate_diamond_dims
        config = GPTConfig(
            block_size=256,
            vocab_size=1000,
            n_layer=6,
            layer_dims=[256, 512, 768, 768, 512, 256],
            n_heads=[4, 8, 12, 12, 8, 4],
            dropout=0.1
        )
        model = GPT(config)
        self.assertIsNotNone(model)
        self.assertEqual(model.config.n_layer, 6)

if __name__ == '__main__':
    unittest.main()
