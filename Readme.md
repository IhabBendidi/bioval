# Bioval
Bioval is a Python package made to provide a wrapper and an easy access to a collection of evaluation metrics for comparing the similarity of two tensors, adapted to different evaluation processes of generative models applied to biological images, and by extension to natural images. The package support unconditional comparison cases, in which we have no labels separating different classes/conditions, and also supports conditional comparisons, with metrics adapted to evaluating the appropriateness of the generated images to all the distributions of real conditions. The package also supports distance of generated conditions from the real negative control condition, used in most biological contexts.

The package supports tensors of shape (N, I, H, W, C) or (N, H, W, C) or (N, I, F) or (N, F), where N is the number of classes/conditions, I is the number of instances, H is the height, W is the width, C is the number of channels, and F is the number of features.

## Installation
Bioval can be installed from PyPI using the following command (from the root of this package):

```bash
poetry build
pip3 install dist/bioval-{version_number}.tar.gz
```

Pypip installation will be supported soon.

## Available Metrics
The following evaluation metrics are available in Bioval:

### Overall KID/FID
This metric measures the KID/FID between two tensors, while being agnostic to the classes and conditions. this metric returns the overall KID/FID score.


### IntraClass Conditional Evaluation
This metric measures the top-k similarity between two tensors, both within and across classes. The metric returns the following scores: intra_top1, intra_top5, and intra_top10. In a detailled output context, it also returns the distance matrix between all generated and true conditions, computed according to the chosen distance method.

### InterClass Conditional Evaluation
This metric evaluates the pearson correlation between the distances between generated classes and the distance between real classes. This evaluates if the generated class images keep the same class relationships and distances. This metric returns inter_corr as the interclass correlations between real and generated, and in a detailled output context, it also returns inter_p.

### Distance from Control
This metric evaluates the difference between the distances of both generated class images and real class images from real control compounds, to measure if their relationship with the control compound (DMSO for example) is preserved. The metric returns the mean difference between the distance of real and generated conditions from control, as well as the individual difference for each class in the detailled output. The smaller the score, the more adapted the generated images are in the biological context.

## Usage settings 

### Aggregated

In an aggregated setting, the features for each class/condition are aggregated together using a chosen aggregation method, before performing interclass/intraclass comparisons. The aggregated of the features of each class/condition is computed, and the resulting tensor is of shape (N, F), where N is the number of classes/conditions, and F is the number of features.

The following aggregation methods are available:

- Mean

- Median

- Robust Mean


### Distributed

In a distributed setting, we compute the distributed distance between the features of each class/condition. This approach takes an exponentially longer time to perform, but achieve more precise results.

The following distributed distance methods are available:

- KID

- FID


## Quick Usage 

### Object Instanciation

Importing the wrapper class : 

```python
from bioval.metrics.conditional_evaluation import ConditionalEvaluation
```

Creating an object instance of the wrapper class :

```python
topk = ConditionalEvaluation()
```

The following parameters can be passed to the object constructor:

- In an aggregated setting : 

    - aggregate: The aggregation method to use. Can be 'mean', 'median', or 'robust_mean'. Default is 'mean'. It is ignored in a distributed setting.

    - method: The intraclass top-k similarity method to use. Can be 'cosine', 'euclidean', or 'chebyshev'. Default is 'euclidean'. It is ignored in a distributed setting.

- In a distributed setting :

    - distributed_method: The distributed distance method to use. Can be 'fid' or 'kid'. Default is 'kid'. It is ignored in an aggregated setting.



### Aggregated Usage of the metrics

Computing all the metrics for two 2D and two 3D tensors with the created object, accross all the conditions for the 3D tensors: 

```python
# Test on 2D tensors
arr1 = torch.randn(100, 10)
arr2 = torch.randn(100, 10)
print("2D tensors")
print(topk(arr1, arr2, k_range=[1, 5, 10, 20, 50, 100]))

# Repeat test on 3D tensors
arr1 = torch.randn(100, 10, 10)
arr2 = torch.randn(100, 10, 10)
print("3D tensors")
print(topk(arr1, arr2, k_range=[1, 5, 10, 20, 50, 100]))
```

Output is a dictionary with all the computed metrics, and the same no matter the shape of the two arrays.

### Detailled Usage 

#### Aggregated 

```python
from bioval.metrics.conditional_evaluation import ConditionalEvaluation

# Create an instance of the ConditionalEvaluation class
topk = ConditionalEvaluation()

# Compute top-k similarity scores between two tensors
arr1 = torch.randn(100, 10)
arr2 = torch.randn(100, 10)
scores = topk(arr1, arr2, k_range=[1, 5, 10, 20, 50, 100])
```

#### Distributed 
```python

from bioval.metrics.conditional_evaluation import ConditionalEvaluation

topk = ConditionalEvaluation(distributed_method='fid') # or 'kid' or 'mmd'
arr1 = torch.randn(30, 20, 10, 10, 3) * 256
arr2 = torch.randn(30, 20, 10, 10, 3) * 256
print(topk(arr1, arr2, k_range=[1, 5, 10],aggregated=False)) # returns a dict of scores
```

### Outputs

#### Aggregated without detailled output

if we call the function : 
```python

topk(arr1, arr2, k_range=[1, 5, 10],detailed_output=False)
```

The shape of the output is : 

```python
{
    'control_score': scalar, 
    'inter_corr': scalar, 
    'intra_top1': scalar, 
    'intra_top5': scalar, 
    'intra_top10': scalar,
}
```

#### Aggregated with detailled output

if we call the function : 

```python

topk(arr1, arr2, k_range=[1, 5, 10],detailed_output=True)
```

The shape of the output is : 
TBD


#### Distributed without detailled output


if we call the function : 

```python

topk(arr1, arr2, k_range=[1, 5, 10],aggregated=False,detailed_output=False)
```

TBD

#### distributed with detailled output with control included


if we call the function : 

```python

topk(arr1, arr2,control, k_range=[1, 5, 10],aggregated=False,detailed_output=True)
```

The shape of the output is : 


```python
{
    'control_score': scalar, 
    'class_control_scores': list of scores, one for each class, 
    'inter_corr': scalar, 
    'inter_p': scalar,p-value of the pearson correlation, 
    'intra_kid': scalar, it can also be intra_fid or intra_mmd if chosen in the distributed_method argument, measured per class,
    'matrix': Tensor, with distances between all real and generated classes using distance method of choice,
    'intra_top1': scalar,
    'intra_top5': scalar,
    'mean_ranks': scalar,
    'exact_matching': scalar,
}
```

## Documentation
Please refer to the docstrings in the code for detailed documentation on each metric.

## Contributing
Contributions are welcome! Please refer to the [Contribute]() file for details on how to contribute to Bioval.

## License
Bioval is licensed under the MIT License. Please refer to the [License]() file for details.
