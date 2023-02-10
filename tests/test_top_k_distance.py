import unittest
import torch
from bioval.metrics.top_k_distance import TopKDistance

class TestTopKDistance(unittest.TestCase):
    def test_init(self):
        # Test default values
        topk = TopKDistance()
        self.assertEqual(topk.method, 'euclidean')
        self.assertEqual(topk.aggregate, 'mean')
        self.assertEqual(topk._methods.keys(), ['euclidean', 'cosine', 'correlation', 'chebyshev', 'minkowski', 'cityblock'])
        self.assertEqual(topk._aggregs.keys(), ['mean', 'median', 'robust_mean'])

        # Test custom values
        topk = TopKDistance(method='cosine', aggregate='median')
        self.assertEqual(topk.method, 'cosine')
        self.assertEqual(topk.aggregate, 'median')
    
    def test_call(self):
        # Test error on invalid method
        topk = TopKDistance()
        with self.assertRaises(ValueError):
            topk(torch.zeros(2, 3), torch.zeros(2, 3), k_range=[1, 5, 10, 100])
            topk.method = 'invalid'
        
        # Test error on invalid aggregate
        topk = TopKDistance(method='cosine')
        with self.assertRaises(ValueError):
            topk(torch.zeros(2, 3), torch.zeros(2, 3), k_range=[1, 5, 10, 100])
            topk.aggregate = 'invalid'
        
        # Test error on invalid arr1 dimension
        topk = TopKDistance(method='cosine', aggregate='median')
        with self.assertRaises(ValueError):
            topk(torch.zeros(2), torch.zeros(2, 3), k_range=[1, 5, 10, 100])
        
        # Test error on invalid arr2 dimension
        topk = TopKDistance(method='cosine', aggregate='median')
        with self.assertRaises(ValueError):
            topk(torch.zeros(2, 3), torch.zeros(2), k_range=[1, 5, 10, 100])
        
        # Test valid call
        arr1 = torch.zeros(2, 3)
        arr2 = torch.zeros(2, 3)
        topk = TopKDistance(method='cosine', aggregate='median')
        result = topk(arr1, arr2, k_range=[1, 5, 10, 100])
        self.assertIsInstance(result, dict)

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
    # call TestTopKDistance.test_init
    # call TestTopKDistance.test_call
