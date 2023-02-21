# Bioval
Bioval is a Python package that provides a collection of evaluation metrics for comparing the similarity of two tensors. The package supports tensors of shape (N, I, H, W, C) or (N, H, W, C) or (N, I, F) or (N, F), where N is the number of classes, I is the number of instances, H is the height, W is the width, C is the number of channels, and F is the number of features.

## Installation
Bioval can be installed from PyPI using the following command:

```bash
Copy code
poetry build
pip3 install dist/bioval-{version_number}.tar.gz
```

## Usage
To use Bioval, create an instance of one of the evaluation classes, and call the __call__ method, passing in two tensors and a range of k values:

```python
Copy code
from bioval.metrics import TopKSimilarity

topk = TopKSimilarity()

# Test on 2D tensors
arr1 = torch.randn(100, 10)
arr2 = torch.randn(100, 10)
print("2D tensors")
start_time = time.time()
print(topk(arr1, arr2, k_range=[1, 5, 10, 20, 50, 100]))
print("Time elapsed: {:.2f}s".format(time.time() - start_time))

# Repeat test on 3D tensors
arr1 = torch.randn(100, 10, 10)
arr2 = torch.randn(100, 10, 10)
print("3D tensors")
start_time = time.time()
print(topk(arr1, arr2, k_range=[1, 5, 10, 20, 50, 100]))
print("Time elapsed: {:.2f}s".format(time.time() - start_time))
```

## Available Metrics
The following evaluation metrics are available in Bioval:

### TopKSimilarity
This metric measures the top-k similarity between two tensors, both within and across classes. The metric returns the following scores: intra_top1, intra_top5, inter_corr, inter_p, mean_ranks, and exact_matching.

```python
Copy code
from bioval.metrics import TopKSimilarity

# Create an instance of the TopKSimilarity class
topk = TopKSimilarity()

# Compute top-k similarity scores between two tensors
arr1 = torch.randn(100, 10)
arr2 = torch.randn(100, 10)
scores = topk(arr1, arr2, k_range=[1, 5, 10, 20, 50, 100])
```

### Other Metrics
Other evaluation metrics, such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Pearson Correlation Coefficient, are also available in Bioval. Please refer to the documentation for details on how to use these metrics.

## Documentation
Please refer to the docstrings in the code for detailed documentation on each metric.

## Contributing
Contributions are welcome! Please refer to the [Contribute]() file for details on how to contribute to Bioval.

## License
Bioval is licensed under the MIT License. Please refer to the [License]() file for details.