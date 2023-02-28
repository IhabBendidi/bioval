import unittest
import torch
import bioval.utils.gpu_manager as gpu_manager
from bioval.metrics.conditional_evaluation import ConditionalEvaluation
import numpy as np
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

class TestConditionalEvaluation(unittest.TestCase):
    def test_init(self):
        # Test default values
        topk = ConditionalEvaluation()
        self.assertEqual(topk.method, 'euclidean')
        self.assertEqual(topk.aggregate, 'mean')
        self.assertEqual(list(topk._methods.keys()), ['euclidean','cosine','correlation','chebyshev','minkowski','cityblock'])
        self.assertEqual(list(topk._aggregs.keys()), ['mean', 'median', 'robust_mean'])

        # Test custom values
        topk = ConditionalEvaluation(method='cosine', aggregate='median')
        self.assertEqual(topk.method, 'cosine')
        self.assertEqual(topk.aggregate, 'median')

        
    
    def test_call(self):
        # Test error on invalid method
        topk = ConditionalEvaluation()
        with self.assertRaises(ValueError):
            topk(torch.zeros(2, 3), torch.zeros(2, 3), k_range=[1, 5, 10])
            topk.method = 'invalid'

        # Test error on invalid aggregate
        topk = ConditionalEvaluation(method='cosine')
        with self.assertRaises(ValueError):
            topk(torch.zeros(2, 3), torch.zeros(2, 3), k_range=[1, 5, 10])
            topk.aggregate = 'invalid'
        
        # Test error on invalid arr1 dimension
        topk = ConditionalEvaluation(method='cosine', aggregate='median')
        with self.assertRaises(ValueError):
            topk(torch.zeros(2), torch.zeros(2, 3), k_range=[1, 5, 10])
        
        # Test error on invalid arr2 dimension
        topk = ConditionalEvaluation(method='cosine', aggregate='median')
        with self.assertRaises(ValueError):
            topk(torch.zeros(2, 3), torch.zeros(2), k_range=[1, 5, 10])
        
        # test tensor type
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        model = ConditionalEvaluation()
        result = model(arr1, arr2)
        self.assertIsInstance(result, dict)

        arr1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[1, 2, 3], [4, 5, 6]])
        model = ConditionalEvaluation()
        result = model(arr1, arr2)
        self.assertIsInstance(result, dict)

        arr1 = torch.tensor([[1, 2, 3]])
        arr2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        model = ConditionalEvaluation()
        with self.assertRaises(ValueError) as context:
            model(arr1, arr2)
        self.assertTrue("First tensor should have at least 2 classes" in str(context.exception))

        arr1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        arr2 = torch.tensor([[1, 2, 3]])
        model = ConditionalEvaluation()
        with self.assertRaises(ValueError) as context:
            model(arr1, arr2)
        self.assertTrue("Second tensor should have at least 2 classes" in str(context.exception))

        best_gpu = gpu_manager.get_available_gpu()

        if best_gpu is not None:

            arr1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
            arr2 = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda:'+str(best_gpu))
            model = ConditionalEvaluation()
            with self.assertRaises(ValueError) as context:
                model(arr1, arr2)
            self.assertTrue("First tensor and second tensor should be on the same device" in str(context.exception))
        
        # Test valid call
        arr1 = torch.zeros(2, 3)
        arr2 = torch.zeros(2, 3)
        topk = ConditionalEvaluation(method='cosine', aggregate='median')
        result = topk(arr1, arr2, k_range=[1, 5, 10])
        self.assertIsInstance(result, dict)

        arr1 = torch.Tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        arr2 = torch.Tensor([
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1]
        ])
        control = torch.Tensor([1, 2, 3])

        topk = ConditionalEvaluation()
        output = topk(arr1, arr2, control,k_range=[1, 5, 10])

        self.assertTrue('control_score' in output)
        self.assertTrue('class_control_scores' in output)
        control_score = output['control_score']
        self.assertIsInstance(control_score, float)

        control_score_2 = output['class_control_scores']
        self.assertIsInstance(control_score_2, list)

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
    # call TestTopKDistance.test_init

    # call TestTopKDistance.test_call
