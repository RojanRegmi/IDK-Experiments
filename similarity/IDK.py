import random

import numpy as np

"""
All comments and definitions are according to the paper:

Ting, Liu et. al, A New Distributional Treatment for Time Series and An Anomaly Detection, 2022
"""


def IK_inne_fm(X,psi,t=100):

    """
    This function implements a Kernel Inner Product algorithm for computing feature maps.

    This is according to the definition 2 in the paper. We want to map our values into a binary matrix with dimensions t x ψ.

    Here,

    X: the Dataset

    psi: represents the number of randomly selected points(samples) used in  each iteration

    returns: a binary array

    """
        

    onepoint_matrix = np.zeros((X.shape[0], (int)(t*psi)), dtype=int)
    for time in range(t):
        sample_num = psi 
        sample_list = [p for p in range(len(X))]
        sample_list = random.sample(sample_list, sample_num)
        sample = X[sample_list, :] # Sample is randomly selected points in X of size psi.

        #The following lines compute the squared euclidean distances between each point in 'X' and 'sample'.
        tem1 = np.dot(np.square(X), np.ones(sample.T.shape))  # n*psi . The np.square(X) elementwise squares X. The dot product with np.ones() gives the sum along an axis.
        tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)  # n*psi

        """
        point2sample is the squared euclidean distance.
        It's shape is n * psi.
        Each element i,j contains the distances from the ith sample in X to the jth sample in psi.
        """

        # The following lines identify the minimum distances from each point in 'X' to the randomly selected points in 'sample'.
        sample2sample = point2sample[sample_list, :]
        row, col = np.diag_indices_from(sample2sample)
        sample2sample[row, col] = 99999999 # Diagonal Elements are set to a large value to ensure that the minimum distance is not from the point to itself.
        radius_list = np.min(sample2sample, axis=1) # This line calculates the minimum distance from each point in 'X' to its nearest neighbour in the current sample

        """
            According to definition 2 in the paper, We are looking for each point in X, for it's closest point in the sample.

            The feature mapping kernel converts this index to 1 and everything else is zero.
        """

        min_point2sample_index=np.argmin(point2sample, axis=1) # Minimum distance from each sample to X
        min_dist_point2sample = min_point2sample_index+time*psi
        point2sample_value = point2sample[range(len(onepoint_matrix)), min_point2sample_index]
        ind=point2sample_value < radius_list[min_point2sample_index] # This line creates a boolean array where each element is true if the distance from a point in X to its closest neighbour in the sample is less than the corresponding radius
        onepoint_matrix[ind,min_dist_point2sample[ind]]=1

    return onepoint_matrix

def IDK(X,psi,t=100):

    """
    This function returns the Isolation Distributional Kernel as defined in Definition 3 of the paper.
    """
    point_fm_list=IK_inne_fm(X=X,psi=psi,t=t)
    feature_mean_map=np.mean(point_fm_list,axis=0)
    return np.dot(point_fm_list,feature_mean_map)/t

