import numpy as np
import random
import sys

sys.path.append('..')

from .IDK import IDK

def IDK_Comp(X1, X2, psi1, width, psi2, t=100):

    """ 
        This function implements the IDK Squared Algorithm mentioned in the paper.

        Inputs:

        X1, X2: Input Time-Series
        width: Size of Window
        
    """
     
    window_num = int(np.ceil(X1.shape[0] / width)) # Calculate the number of windows that can be created.

    featuremap_count = np.zeros((window_num, t * psi1))
    onepoint_matrix = np.full((X1.shape[0], t), -1)

    for time in range(t):
        sample_num = psi1
        sample_list = [p for p in range(X1.shape[0])]

        # Randomly sample indices of X2
        sample_list_X1 = random.sample(sample_list, sample_num)
        sample_X2 = X2[sample_list_X1, :]

        # Compute distances between points in X1 and X2
        tem1 = np.dot(np.square(X1), np.ones(sample_X2.T.shape))
        tem2 = np.dot(np.ones(X1.shape), np.square(sample_X2.T))
        # tem1 = np.dot(np.square(sample_X1), np.ones(sample_X2.T.shape))
        #tem2 = np.dot(np.ones(sample_X1.shape), np.square(sample_X2.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X1, sample_X2.T)

        sample2sample = point2sample[sample_list_X1, :]
        row, col = np.diag_indices_from(sample2sample)
        sample2sample[row, col] = np.inf

        radius_list = np.min(sample2sample, axis=1)
        min_dist_point2sample = np.argmin(point2sample, axis=1)

        for i in range(X1.shape[0]):
            if point2sample[i][min_dist_point2sample[i]] < radius_list[min_dist_point2sample[i]]:
                onepoint_matrix[i][time] = min_dist_point2sample[i] + time * psi1
                featuremap_count[(int)(i / width)][onepoint_matrix[i][time]] += 1

    for i in range((int)(X1.shape[0] / width)):
        featuremap_count[i] /= width

    isextra = X1.shape[0] - (int)(X1.shape[0] / width) * width
    if isextra > 0:
        featuremap_count[-1] /= isextra

    if isextra > 0:
        featuremap_count = np.delete(featuremap_count, [featuremap_count.shape[0] - 1], axis=0)

    return IDK(featuremap_count, psi=psi2, t=100)

def IDK_Comp_att2(X1, X2, psi1, width, psi2, t=100):

    """ 
        This function implements the IDK Squared Algorithm mentioned in the paper.

        Inputs:

        X1, X2: Input Time-Series
        width: Size of Window
        
    """

    X = np.concatenate((X1, X2))

    window_num = int(np.ceil(X1.shape[0] / width)) # Calculates the number of windows that can be created by X.
    featuremap_count = np.zeros((window_num, t * psi1))
    onepoint_matrix = np.full((X1.shape[0], t), -1)

    for time in range(t):
        sample_num = psi1
        sample_list = [p for p in range(X.shape[0])]
        sample_list = random.sample(sample_list, sample_num)
        sample = X[sample_list, :]

        tem1 = np.dot(np.square(X1), np.ones(sample.T.shape))  # n*psi
        tem2 = np.dot(np.ones(X1.shape), np.square(sample.T))
        point2sample = tem1 + tem2 - 2 * np.dot(X1, sample.T)  # n*psi


        sample2sample = point2sample[sample_list, :]
        row, col = np.diag_indices_from(sample2sample)
        sample2sample[row, col] = 99999999

        radius_list = np.min(sample2sample, axis=1)
        min_dist_point2sample = np.argmin(point2sample, axis=1)  # index

        for i in range(X1.shape[0]):
            if point2sample[i][min_dist_point2sample[i]] < radius_list[min_dist_point2sample[i]]:
                onepoint_matrix[i][time] = min_dist_point2sample[i] + time * psi1
                featuremap_count[(int)(i / width)][onepoint_matrix[i][time]] += 1

    # feature map of D/width
    for i in range((int)(X1.shape[0] / width)):
        featuremap_count[i] /= width
    isextra = X1.shape[0] - (int)(X.shape[0] / width) * width
    if isextra > 0:
        featuremap_count[-1] /= isextra

    if isextra > 0:
        featuremap_count = np.delete(featuremap_count, [featuremap_count.shape[0] - 1], axis=0)

    return IDK(featuremap_count, psi=psi2, t=100)

