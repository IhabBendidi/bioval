# Bioval
Bioval is a Python package made to provide a wrapper and an easy access to a collection of evaluation metrics for comparing the similarity of two tensors, adapted to different evaluation processes of generative models applied to biological images, and by extension to natural images. The package support unconditional comparison cases, in which we have no labels separating different classes/conditions, and also supports conditional comparisons, with metrics adapted to evaluating the appropriateness of the generated images to all the distributions of real conditions. The package also supports distance of generated conditions from the real negative control condition, used in most biological contexts.

The package supports both images (with any number of channels) and encoded features of images. The compared tensors need to have the same shape. This package performs encoding of images using the InceptionV3 encoder pretrained on ImageNet dataset, but can also handle features encoded using different encoders.

## Installation

### Installation from source
Bioval can be built using the following command (from the root of this package):

```bash
poetry build
pip install dist/bioval-{version_number}.tar.gz
```

### Installation from Pypip

You can install the current alpha version through the command :

```bash
pip install bioval
```

pip install bioval

## Available Metrics
The following evaluation metrics are available in Bioval:

### Overall KID/FID
This metric measures the KID/FID between two tensors, while being agnostic to the classes and conditions. this metric returns the overall KID/FID score.

### Intraclass KID/FID

This metric measures the KID/FID score between two tensors (generated tensor and real tensor in the context of image generation) of the same class condition, and computes the mean of the score over all the classes. This metric returns the intraclass KID/FID score.


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


### Input tensors

The package supports both images (with any number of channels) and encoded features of images. The input tensors can be of shape (N, I, H, W, C) or (N, H, W, C) or (N, I, F) or (N, F), where N is the number of classes/conditions, I is the number of instances, H is the height, W is the width, C is the number of channels, and F is the number of features.

The compared tensors need to have the same shape.

Examples of possible input tensors : 

```python
# 2D tensors (100 conditions, with one image 
# embedding in each condition of size 1024)
arr1 = torch.randn(100, 1024)
arr2 = torch.randn(100, 1024)

# 3D tensors (100 conditions, with 10 image embeddings 
# in each condition of size 1024)
arr1 = torch.randn(100, 10, 1024)
arr2 = torch.randn(100, 10, 1024)

# 4D tensors (100 conditions, with one image
# in each condition of size 256x256x3)
arr1 = torch.randn(100, 256, 256, 3)
arr2 = torch.randn(100, 256, 256, 3)

# 5D tensors (100 conditions, with 10 images
# in each condition of size 256x256x3)
arr1 = torch.randn(100, 10, 256, 256, 3)
arr2 = torch.randn(100, 10, 256, 256, 3)


# 5D tensors (100 conditions, with 10 images
# in each condition of size 256x256x5 with 5 channels)
arr1 = torch.randn(100, 10, 256, 256, 5)
arr2 = torch.randn(100, 10, 256, 256, 5)


# control array is a torch.Tensor object of shape (I, H, W, C) or 
# (H, W, C) or (I, F) or (F), where C is the number of channels, 
# H is the height, and W is the width of the input image, and 
# I the number of instances, and F is the number of features.

# control array with 10 images and 3 channels
control = torch.randn(10, 256, 256, 3)

# control array with 1 image and 3 channels
control = torch.randn(256, 256, 3)

# control array with 10 images and an embedding of size 1024
control = torch.randn(10, 1024)

# control array with 1 image and an embedding of size 1024
control = torch.randn(1024)

```

In the case where nb of channels in input images differs from 3 channels, each channel is encoded separately, and the embeddings of each channel are concatenated into one embedding. For example, if the input images have 5 channels, the output embedding will be of size 5*embedding_size.


### Aggregated Usage of the metrics

Computing all the metrics for our input tensors can be done using our created instance of our object, and passing the input tensors to it. 

```python

output = topk(arr1, arr2, k_range=[1, 5, 10, 20, 50, 100])
```

The Object instance can be called with the following parameters:

- arr1: The first input tensor. Can be of shape (N, I, H, W, C) or (N, H, W, C) or (N, I, F) or (N, F), where N is the number of classes/conditions, I is the number of instances, H is the height, W is the width, C is the number of channels, and F is the number of features.

- arr2: The second input tensor. Can be of shape (N, I, H, W, C) or (N, H, W, C) or (N, I, F) or (N, F), where N is the number of classes/conditions, I is the number of instances, H is the height, W is the width, C is the number of channels, and F is the number of features. Array 1 and Array 2 must have the same shape.

- control: The control tensor. Default is None when the distance from control metric is not of desired. When not None, it is a torch.Tensor object of shape (I, H, W, C) or (H, W, C) or (I, F) or (F), where C is the number of channels, H is the height, and W is the width of the input image, and I the number of instances, and F is the number of features.

- k_range: The range of k values to use for the top-k similarity metrics. Default is [1, 5, 10]. If the range is bigger than the number of conditions, the range is truncated to the number of conditions. 

- aggregated : Whether to use the aggregated (if True) or distributed metrics (if False). Default is True.

- detailed : Whether to return the detailed results of the metrics (if True) or not (if False). Default is True.

- batch_size : The batch size to use for the encoding of the input images, to avoid out of memory issues. Default is 256.

- percent : Percentage of images in each condition to be used for unconditional FID computation. Default is 0.1 (10%)


Output is a dictionary with all the computed metrics, and the same no matter the shape of the two arrays.
    
```javascript
{
        # KID/FID scores
        'overall_kid': scalar, # if distributed method is kid and aggregated is False
        'overall_fid': scalar, # if distributed method is fid and aggregated is False
        'intra_kid': scalar, # if distributed method is kid and aggregated is False
        'intra_fid': scalar, # if distributed method is fid and aggregated is False

        # Control metric scores
        'control_score': scalar, 
        'class_control_scores': numpy array, # if detailed output is True, control scores for each condition

        # Interclass scores
        'inter_corr': scalar, # interclass metric
        'inter_pval': scalar, # if detailed output is True, p value of correlation
        'interclass_matrix_1': numpy array, # if detailed output is True, interclass matrix of array 1
        'interclass_matrix_2': numpy array, # if detailed output is True, interclass matrix of array 2

        # intraclass scores
        'intra_top1': scalar, 
        'intra_top5': scalar, 
        'intra_top10': scalar,
        'mean_ranks': scalar, # if detailed output is True, mean ranks of intraclass distances matrix
        'matrix': numpy array, # if detailed output is True, intraclass distances matrix between all conditions
}
```

## Contributing
Contributions are welcome! Please refer to the [Contribute](https://github.com/IhabBendidi/bioval/blob/main/Contribute.md) file for details on how to contribute to Bioval.

## License
Bioval is licensed under the MIT License. Please refer to the [License](https://github.com/IhabBendidi/bioval/blob/main/LICENSE) file for details.
