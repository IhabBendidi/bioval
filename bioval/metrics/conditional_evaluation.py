import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import time
import bioval.utils.gpu_manager as gpu_manager
import bioval.utils.distance as distance
import bioval.utils.aggregation as aggregation

from scipy.stats import pearsonr





class ConditionalEvaluation():
    """
    This class computes top K distance between two tensors.

    Parameters
    ----------
    method : str, optional
        The method for computing the distance between the two tensors. The options are 'euclidean', 'cosine',
        'correlation', 'chebyshev', 'minkowski', and 'cityblock'. The default method is 'euclidean'.

    aggregate : str, optional
        The method for aggregating a 3D tensor into a 2D tensor. The options are 'mean', 'median', and 'robust_mean'.
        The default aggregate method is 'mean'.

    Attributes
    ----------
    method : str
        The method for computing the distance between the two tensors.

    aggregate : str
        The method for aggregating a 3D tensor into a 2D tensor.

    _methods : dict
        A dictionary containing all the available methods for computing the distance between the two tensors.

    _aggregs : dict
        A dictionary containing all the available methods for aggregating a 3D tensor into a 2D tensor.
    """
    def __init__(self, method: str = 'euclidean',aggregate: str = 'mean'):
        self._method = method
        self._aggregate = aggregate
        self._methods = distance.get_distance_functions()
        self._aggregs = aggregation.get_aggregation_functions()
        self.inception = models.inception_v3(pretrained=True)
        # Set the model to evaluation mode
        self.inception.eval()
        if self._method not in list(self._methods.keys()):
            raise ValueError(f"Method {self._method} not available. Available methods are {list(self._methods.keys())}")

        if self._aggregate not in list(self._aggregs.keys()):
            raise ValueError(f"Aggregation method {self._aggregate} not available. Available aggregation methods are {list(self._aggregs.keys())}")

    @property
    def method(self):
        """
        Get the value of the 'method' property.
        Returns:
                str: The comparison method to be used for calculating the distance between arrays.
        """
        return self._method

    @method.setter
    def method(self, value):
        """
        Set the value of the 'method' property.

        Args:
            value (str): The comparison method to be used for calculating the distance between arrays.

        Raises:
            ValueError: If the value provided for 'method' is not in the list of valid methods.
        """
        if value not in self._methods:
            raise ValueError("Invalid method, choose from {}".format(self._methods))
        self._method = value

    @property
    def aggregate(self):
        """
        Get the value of the 'aggregate' property.

        Returns:
            str: The aggregation method to be used to convert 3D tensors to 2D tensors.
        """
        return self._aggregate

    @aggregate.setter
    def aggregate(self, value):
        """
        Set the value of the 'aggregate' property.

        Args:
            value (str): The aggregation method to be used to convert 3D tensors to 2D tensors.

        Raises:
            ValueError: If the value provided for 'aggregate' is not in the list of valid aggregations.
        """
        if value not in self._aggregs:
            raise ValueError("Invalid aggregation method, choose from {}".format(self._aggregs))
        self._aggregate = value

    def __call__(self, arr1: torch.Tensor, arr2: torch.Tensor, k_range=[1, 5, 10]) -> dict:
        """
        This function is used to compare two tensors and return a dictionary with the scores for each value in k_range. The comparison is performed based on the method and aggregation set for the class. The tensors should be either 2D or 3D tensors for embeddings, or 4D or 5D tensors for images.

        The function checks the input tensors, raises an error if they are not torch.Tensors, or if they do not have the same number of classes or features, or if they are not on the same device.

        The function also checks the values of method and aggregate, raises an error if they are not in the list of defined methods or aggregations.

        The function aggregates the tensors if they are 3D tensors, then it computes the comparison matrix using the specified method and computes the diagonal ranks of the comparison matrix. Finally, the function computes the scores for each value in k_range and returns a dictionary with the scores.

        Parameters:
            arr1 (torch.Tensor): The first tensor to be compared.
            arr2 (torch.Tensor): The second tensor to be compared.
            k_range (List[int]): A list of values for the range of k.

        Returns:
            dict: A dictionary with the scores for each value in k_range.


        Example
        -------
        >>> arr1 = torch.tensor([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.1, 0.3, 0.6]])
        >>> arr2 = torch.tensor([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.1, 0.3, 0.6]])
        >>> top_k_distance = TopKDistance()
        >>> top_k_distance(arr1, arr2)
        {'top1': tensor(1.5000), 'top5': tensor(6.5000), 'mean_ranks': tensor(50.1375), 'exact_matching': tensor(0.)}
        """   
        # check if format is correct and prepare data if its in image format
        arr1,arr2 = self._prepare_data_format(arr1, arr2,k_range)
        dict_score = {}
        #### Inter class metric
        dict_score = self._compute_interclass_scores(arr1, arr2,dict_score)
        #### Intra class metric
        dict_score = self._compute_intraclass_scores(arr1, arr2,k_range,dict_score)
        return dict_score


    
    @staticmethod
    def _compute_diag_ranks(matrix: torch.Tensor) -> torch.Tensor:
        """
        Computes the diagonal ranks of a given matrix.

        Parameters:
        matrix (torch.Tensor): 2D tensor of similarity scores between two arrays.

        Returns:
        torch.Tensor: 1D tensor of ranks of the diagonal elements in the input matrix. 
                    Ranks start from 1, with 1 being the highest score.

        Example:
        >>> matrix = torch.tensor([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.1, 0.3, 0.6]])
        >>> _compute_diag_ranks(matrix)
        tensor([1, 3, 2])
        """
        # Sort matrix in descending order along each column
        _, indices = torch.sort(matrix, dim=0, descending=True)

        # Initialize ranks tensor with zeros
        ranks = torch.zeros(matrix.shape[0], dtype=torch.int64)

        # Compute the rank of each diagonal element by finding its index
        for i in range(matrix.shape[0]):
            ranks[i] = (indices[:, i] == i).nonzero()[0].item() + 1  # to have index starting at 1

        return ranks


    @staticmethod
    def _compute_pearson_correlation(matrix_1, matrix_2):
        """
        Calculates the Pearson correlation coefficient and the p-value between two matrices.

        Args:
            matrix_1 (torch.Tensor): First input matrix.
            matrix_2 (torch.Tensor): Second input matrix.

        Returns:
            A tuple containing the Pearson correlation coefficient (float) and the p-value (float) between the two matrices.

        Raises:
            TypeError: If either matrix_1 or matrix_2 is not a PyTorch Tensor.
        """
        # Compute the Pearson correlation between the entire vectors
        correlation, p_value = pearsonr(matrix_1.cpu().numpy(), matrix_2.cpu().numpy())

        # Return the vector-wise correlation and p-value
        return correlation, p_value
    

    def _compute_interclass_scores(self,arr1: torch.Tensor, arr2: torch.Tensor,output : dict) -> dict:
        """
        Computes the interclass scores of two matrices using the specified comparison method.
        Interclass metric is a metric that allows the comparison of classes of two different sets or matrices. 

        Args:
        - matrix_1: a tensor of shape (num_samples_1, num_features) representing the first matrix
        - matrix_2: a tensor of shape (num_samples_2, num_features) representing the second matrix
        - output: a dictionary containing the scores of comparison. 

        Returns:
        A dictionary containing the correlation score and the p-value score of the interclass metric comparison.

        Raises:
        - ValueError: if any of the matrices does not have a valid shape or if the comparison method is not valid.

        """
        #### Inter class metric
        matrix_1 = self._methods[self._method](arr1, arr1)
        matrix_2 = self._methods[self._method](arr2, arr2)
        # delete the diagonal of each matrix
        matrix_1 = matrix_1[~torch.eye(matrix_1.shape[0], dtype=bool)].view(matrix_1.shape[0], -1)
        matrix_2 = matrix_2[~torch.eye(matrix_2.shape[0], dtype=bool)].view(matrix_2.shape[0], -1)

        # flatten the matrices
        matrix_1 = matrix_1.flatten()
        matrix_2 = matrix_2.flatten()

        # Compute correlations between the two matrices
        corr,pvalue = self._compute_pearson_correlation(matrix_1, matrix_2)

        # add the correlation score to the dictionary
        output['inter_corr'] = corr
        output['inter_p'] = pvalue
        return output
    
    def _compute_intraclass_scores(self,arr1: torch.Tensor, arr2: torch.Tensor,k_range : list,output : dict) -> dict:
        # get the matrix for comparison using the specified method
        matrix = self._methods[self._method](arr1, arr2)
        # compute the diagonal ranks of the comparison matrix
        ranks = self._compute_diag_ranks(matrix)
        # compute the scores for each value in k_range
        mean_ranks = torch.mean(ranks.float(), dim=0)
        
        # compute the scores for each value in k_range
        for k in k_range :
            number_values = k 
            r = (ranks <= number_values).sum()
            r = (r/matrix.shape[0]) * 100
            output['intra_top'+str(k)] = r
        # add the mean ranks score to the dictionary
        output['mean_ranks'] = (mean_ranks/matrix.shape[0]) * 100
        # add the exact matching score to the dictionary
        r_exact = (ranks == 0).sum()
        output['exact_matching'] = (r_exact/matrix.shape[0]) * 100
        return output
    
    def _prepare_data_format(self,arr1: torch.Tensor, arr2: torch.Tensor,k_range : list) -> tuple:
        # check if arr1 and arr2 are tensors of the same shape and raise error if not
        if isinstance(arr1, np.ndarray):
            arr1 = torch.tensor(arr1)
        if isinstance(arr2, np.ndarray):
            arr2 = torch.tensor(arr2)
        if not isinstance(arr1, torch.Tensor):
            raise TypeError(f"First tensor should be a torch.Tensor but got {type(arr1)}")
        if not isinstance(arr2, torch.Tensor):
            raise TypeError(f"Second tensor should be a torch.Tensor but got {type(arr2)}")
        # check if arr1 and arr2 are floats, and convert to float if they are not
        if arr1.dtype != torch.float:
            # try to convert to float and if error, raise error that input should have float dtype
            try:
                arr1 = arr1.float()
            except:
                raise TypeError(f"First tensor should have float dtype but got {arr1.dtype}")
        if arr2.dtype != torch.float:
            # try to convert to float and if error, raise error that input should have float dtype
            try:
                arr2 = arr2.float()
            except:
                raise TypeError(f"Second tensor should have float dtype but got {arr2.dtype}")
        if arr2.dtype != torch.float:
            arr2 = arr2.float()
        # Check if the number of classes is greater than 1
        if arr1.shape[0] < 2:
            raise ValueError(f"First tensor should have at least 2 classes but got {arr1.shape[0]}")
        if arr2.shape[0] < 2:
            raise ValueError(f"Second tensor should have at least 2 classes but got {arr2.shape[0]}")
        # control if the tensors have the same shape for the 0 dimension
        if arr1.shape[0] != arr2.shape[0]:
            raise ValueError(f"First tensor and second tensor should have the same number of classes in dimension 0 but got {arr1.shape[0]} and {arr2.shape[0]} respectively")
        # check if arr1 and arr2 are 2D or 3D or 4D or 5D tensors and raise error if not
        if arr1.ndim not in [2, 3,4,5]:
            raise ValueError(f"First tensor should be a 2D or 3D tensor for embeddings, or 4D or 5D tensor for images, but got {arr1.ndim}D tensor")
        if arr2.ndim not in [2, 3,4,5]:
            raise ValueError(f"Second tensor should be a 2D or 3D tensor for embeddings, or 4D or 5D tensor for images, but got {arr2.ndim}D tensor")
        # check if number of classes if less than max value of k_range
        if arr1.shape[0] < max(k_range):
            # modify k_range to have only values less than number of classes
            k_range = [k for k in k_range if k <= arr1.shape[0]]
        # check if both arrays are on the same device
        if arr1.device != arr2.device:
            raise ValueError(f"First tensor and second tensor should be on the same device but got {arr1.device} and {arr2.device} respectively")
        # here the code for inception model
        if arr1.ndim in [4,5] :
            # call inception function
            arr1 = self._extract_inception_embeddings(arr1)
        if arr2.ndim in [4,5] :
            # call inception function
            arr2 = self._extract_inception_embeddings(arr2)
        # control if both tensors have the same shape for the embedding dimension (if vector of size 2D or 3D)
        if arr1.ndim in [2, 3] and arr2.ndim in [2, 3] and arr1.shape[-1] != arr2.shape[-1]:
            raise ValueError(f"First tensor and second tensor should have the same number of features in dimension -1 but got {arr1.shape[-1]} and {arr2.shape[-1]} respectively")
                # check if method and aggregate attributes are valid
        if self._method not in self._methods:
            raise ValueError(f"{self._method} not in list of defined methods. Please choose from {list(self._methods.keys())}")
        if self._aggregate not in self._aggregs:
            raise ValueError(f"{self._aggregate} not in list of defined aggregations. Please choose from {list(self._aggregs.keys())}")
        # aggregate the arrays if they are 3D tensors
        if arr1.ndim == 3:
            arr1 = self._aggregs[self._aggregate](arr1)
        if arr2.ndim == 3:
            arr2 = self._aggregs[self._aggregate](arr2)
        return arr1,arr2
        





        


    def _extract_inception_embeddings(self,images: torch.Tensor)  -> torch.Tensor:
        """
        Extract Inception embeddings from a batch of images.

        This function takes in a batch of images as a `torch.Tensor` and returns the embeddings generated
        by the Inception model. The input image can either be of shape (number of classes, number of images in classes, height, width, channels) or (number of classes, height, width, channels).
        The function performs the following steps:
            1. Resizes and crops the images to 299x299.
            2. Normalizes the images according to the mean and standard deviation specified.
            3. Passes the images through the Inception model and extracts the embeddings.
            4. Returns the embeddings with the original shape if input was 5 dimensional or reshaped if input was 4 dimensional.

        Args:
            images (torch.Tensor): A batch of images with shape (number of classes, number of images in classes, height, width, channels) or (number of classes, height, width, channels)

        Returns:
            torch.Tensor: The embeddings generated from the Inception model for the given batch of images.

        Raises:
            ValueError: If the input image shape is not 4 or 5 dimensional.

        """
        transform = transforms.Compose([    
            transforms.Resize(299),    
            transforms.CenterCrop(299),    
            transforms.ToTensor(),    
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        original_shape = images.shape
        # Convert images to a tensor
        if len(images.shape) == 5:
            # case when input is ('number of classes','number of images in classes', 'height','width', 'channels')
            images = images.reshape(original_shape[0]*original_shape[1], images.shape[2], images.shape[3], images.shape[4])        
        elif len(images.shape) == 4:
            # case when input is ('number of classes', 'height','width', 'channels')
            pass
        else:
            raise ValueError("Input image shape should be either 5 or 4 dimensional")
        # get the device of the images
        device = images.device
        # move the inception model to the same device as the images
        self.inception.to(device)
        # Transfer the images to cpu
        images = images.cpu()
        images = images.numpy()
        images = images.astype('uint8')
        images = [Image.fromarray(image) for image in images]
        images = torch.stack([transform(image) for image in images])
        # transfer images to the device of the inception model
        images = images.to(device)
        # Forward pass the images through the model
        with torch.no_grad():
            embeddings = self.inception(images).detach()
        if len(original_shape) == 5:
            embeddings = embeddings.reshape(original_shape[0],original_shape[1],-1 )
        elif len(original_shape) == 4:
            embeddings = embeddings.reshape(images.shape[0], -1)
        return embeddings








    
# test the class
if __name__ == '__main__':
    best_gpu = gpu_manager.get_available_gpu()
    
    topk = ConditionalEvaluation()

    # test on 2D tensors
    arr1 = torch.randn(100, 10)
    arr2 = torch.randn(100, 10)
    print("2D tensors")
    start_time = time.time()
    print(topk(arr1, arr2, k_range=[1, 5, 10, 20, 50, 100]))
    print("Time elapsed: {:.2f}s".format(time.time() - start_time))

    # repreat test on 3D tensors
    arr1 = torch.randn(100, 10, 10)
    arr2 = torch.randn(100, 10, 10)
    print("3D tensors")
    start_time = time.time()
    print(topk(arr1, arr2, k_range=[1, 5, 10, 20, 50, 100]))
    print("Time elapsed: {:.2f}s".format(time.time() - start_time))

    # test on 4D tensors

    arr1 = torch.randn(100, 10, 10,3) * 256
    arr2 = torch.randn(100, 10, 10, 3) * 256
    print("4D tensors on CPU")
    start_time = time.time()
    print(topk(arr1, arr2, k_range=[1, 5, 10, 20, 50, 100]))
    print("Time elapsed: {:.2f}s".format(time.time() - start_time))

    # test on 5D tensors
    arr1 = torch.randn(100, 2, 10, 10, 3) * 256
    arr2 = torch.randn(100, 2, 10, 10, 3) * 256
    print("5D tensors on CPU")
    start_time = time.time()
    print(topk(arr1, arr2, k_range=[1, 5, 10, 20, 50, 100]))
    print("Time elapsed: {:.2f}s".format(time.time() - start_time))
    


    if best_gpu is not None:
        # test on 2D tensors
        arr1 = torch.randn(100, 10)
        arr2 = torch.randn(100, 10)
        arr1 = arr1.cuda(best_gpu)
        arr2 = arr2.cuda(best_gpu)
        print("2D tensors on GPU")
        start_time = time.time()
        print(topk(arr1, arr2, k_range=[1, 5, 10]))
        print("Time elapsed: {:.2f}s".format(time.time() - start_time))

        # repreat test on 3D tensors
        arr1 = torch.randn(110, 10, 10)
        arr2 = torch.randn(110, 10, 10)
        arr1 = arr1.cuda(best_gpu)
        arr2 = arr2.cuda(best_gpu)
        print("3D tensors on GPU")
        start_time = time.time()
        print(topk(arr1, arr2, k_range=[1, 5, 10]))
        print("Time elapsed: {:.2f}s".format(time.time() - start_time))

        arr1 = torch.randn(110, 10, 10,3) * 256
        arr2 = torch.randn(110, 10, 10, 3) * 256
        # pass the arrays to first gpu
        arr1 = arr1.cuda(best_gpu)
        arr2 = arr2.cuda(best_gpu)
        print("4D tensors on GPU")
        start_time = time.time()
        print(topk(arr1, arr2, k_range=[1, 5, 10]))
        print("Time elapsed: {:.2f}s".format(time.time() - start_time))

        # test on 5D tensors
        arr1 = torch.randn(110, 2, 10, 10, 3) * 256
        arr2 = torch.randn(110, 2, 10, 10, 3) * 256
        arr1 = arr1.cuda(best_gpu)
        arr2 = arr2.cuda(best_gpu)
        print("5D tensors on GPU")
        start_time = time.time()
        print(topk(arr1, arr2, k_range=[1, 5, 10]))
        print("Time elapsed: {:.2f}s".format(time.time() - start_time))


        # test on 5D tensors
        arr1 = torch.randn(8, 2, 10, 10, 3) * 256
        arr2 = torch.randn(8, 2, 10, 10, 3) * 256
        arr1 = arr1.cuda(best_gpu)
        arr2 = arr2.cuda(best_gpu)
        print("5D tensors on GPU, 8 classes")
        start_time = time.time()
        print(topk(arr1, arr2, k_range=[1, 5, 10]))
        print("Time elapsed: {:.2f}s".format(time.time() - start_time))

        # test on 5D tensors
        arr1 = torch.randn(50, 2, 10, 10, 3) * 256
        arr2 = torch.randn(50, 2, 10, 10, 3) * 256
        arr1 = arr1.cuda(best_gpu)
        arr2 = arr2.cuda(best_gpu)
        print("5D tensors on GPU, 50 classes")
        start_time = time.time()
        print(topk(arr1, arr2, k_range=[1, 5, 10]))
        print("Time elapsed: {:.2f}s".format(time.time() - start_time))

        # test on 5D tensors
        arr1 = torch.randn(150, 2, 10, 10, 3) * 256
        arr2 = torch.randn(150, 2, 10, 10, 3) * 256
        arr1 = arr1.cuda(best_gpu)
        arr2 = arr2.cuda(best_gpu)
        print("5D tensors on GPU, 150 classes")
        start_time = time.time()
        print(topk(arr1, arr2, k_range=[1, 5, 10]))
        print("Time elapsed: {:.2f}s".format(time.time() - start_time))
    

