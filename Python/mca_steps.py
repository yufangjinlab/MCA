import numpy as np

"""
Ramin Mohammadi. mca_steps.py: File holds functions that perform these MCA steps: computing the fuzzy matrices and computing the coordinate matrices.
    Copyright (C) 2023  Ramin Mohammadi

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses """
    
# Reference: https://github.com/RausellLab/CelliD



"""
Computes fuzzy matrix
input: 
    - X: a 2D numpy array of the input dataset passed into RunMCA() and after preprocessing has been done
returns a list:
    - Z: numpy 2D array   
    - Dc: numpy vector (array)
"""
def MCAStep1(X: np.ndarray) -> dict:
    AM = X
    rmin = np.min(AM, axis=1) #get min val in each row
    rmax = np.max(AM, axis=1) #get max val in each row
    range_list = rmax - rmin 
    # subtract the min value for a row from each scalar in a row
    for index in range(len(rmin)):
        AM[index] -= rmin[index]
    # divide each value in a row vector by the row's range difference of (max - min) in a row
    for index in range(len(range_list)):
        AM[index] /= range_list[index]
        AM[index] = np.nan_to_num(AM[index]) # turns nan values to 0 in case where a number is divided by 0
    # fuzzy matrix stored in FM
    FM = np.concatenate((AM, 1 - AM), axis=0).astype('float64')
    AM = None
    total = np.sum(FM)
    colsum = np.sum(FM, axis=0) # d_c -> vector containing sum of each column
    rowsum = np.sum(FM, axis=1) # d_r -> vector containing sum of each row
    for index in range(len(rowsum)): # iterate the rows
        FM[index] /= np.sqrt(colsum)
        FM[index] = np.nan_to_num(FM[index]) # turns nan values to 0 in case where a number is divided by 0
    for index in range(len(rowsum)):
        FM[index] /= np.sqrt(rowsum[index])
        FM[index] = np.nan_to_num(FM[index]) 
    Dc = 1/(np.sqrt(rowsum/total))
    Dc = np.nan_to_num(Dc) 
    Dc = np.reshape(Dc, (np.shape(Dc)[0], 1))# turn Dc into a column vector 
    return {"Z": FM , # Z is S, matrix of standardized relative frequencies?
            "Dc": Dc} 

"""
Computes coodinates
input:
    - Z: numpy 2D array from MCAStep1
    - V: numpy 2D array of Right singuar vectors from SVD computation
    - Dc: numpy vector (array) from MCAStep1
returns a list:
    - cellsCoordinates: numpy 2D array of cells coordinates where rows are cells and there are nmcs number of columns where columns represent dimensions
    - features: numpy 2D array of features coordinates where rows are features (genes) and there are nmcs number of columns where columns represent dimensions
"""
def MCAStep2(Z: np.ndarray, V: np.ndarray, Dc: np.ndarray) -> dict:
    AV = V 
    AZ = Z
    ADc = Dc
    Dc.reshape(1, np.shape(Dc)[0]) # turn column vector into a row vector
    FeaturesCoordinates = np.dot(AZ , AV)
    AZcol = np.shape(AZ)[1] # num columns in AZ
    AZ = None
    for index in range(np.shape(FeaturesCoordinates)[0]):
        FeaturesCoordinates[index] *= ADc[index]
    ADc = None
    cellsCoordinates = np.sqrt(AZcol) * AV
    return {"cellsCoordinates": cellsCoordinates,
            "featuresCoordinates": FeaturesCoordinates[ :int(np.shape(FeaturesCoordinates)[0]/2)] }