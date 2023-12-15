import numpy as np
import pandas as pd

"""
Ramin Mohammadi. dist.py responsible for functions related to finding distances between the coordinate matrices resulted from MCA calculations.
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
Small intermediate function for euclidean distance calculation between
MCA feature coordinates and cell coordinates. Due to MCA pseudo barycentric relationship, 
the closer a gene g is to a cell c, the more specific to such a cell it can be considered

input:
    - X: MCA object (defined in mca.py) containing 2 dataframes, one containing cell coordinates and one containing gene coordinates
    - dims: a range that specifies which columns to use in the coordinate data frames since there are 50 columns/dimensions. By default will use all 50 columns
    - features: String vector of feature names (rows) to subset from feature coordinates. If not specified will use all features 
    - cells: String vector of cell names (rows) to subset from cell coordinates. If not specified will use all cells 
return:
    - CellGeneDistances: numpy 2d array with genes as rows and cells as columns. Each value represents the distance between gene x and cell y
"""
def GetDistances(X: object, dims=range(50), features: list = None, cells: list = None):
    # features if passed a value should be a list of specific row names (features) as strings that want to use
    if (features is not None):
        X.featuresCoordinates = X.featuresCoordinates.loc[features]
    # cells if passed a value should be a list of specific row names (cells) as strings that want to use
    if (cells is not None):
        X.cellsCoordinates = X.cellsCoordinates.loc[cells]
        
    # reduce dimensionality by taking only first dims columns in each coordinate matrix
    X.featuresCoordinates = X.featuresCoordinates.iloc[:, dims]
    X.cellsCoordinates = X.cellsCoordinates.iloc[:, dims]
    
    print("\nCalculating distances between cells and genes...\n")
    #calculate distance between cells and genes
    CellGeneDistances = pairDist(X.featuresCoordinates, X.cellsCoordinates)
    
    return CellGeneDistances
    

"""
Helper function that performs DISTANCE CALCULATIONS
input: 
    -Ar: pandas data frame for GENE coordinates 
    -Br: pandas data frame for the CELL coordinates 
returns: 
    -a 2D numpy array of distances between each pair of cells and genes where rows are genes (features) and columns
"""
def fastPDist(Ar: pd.DataFrame, Br: pd.DataFrame) -> np.ndarray:
    m = Ar.shape[0]
    n = Br.shape[0]
    k = Ar.shape[1]
    
    A = Ar.to_numpy()
    B = Br.iloc[:n, :k].to_numpy()
    # element wise sqaure the matrices then compute sum of elements in each row
    An = np.sum(np.square(A), axis=1)
    Bn = np.sum(np.square(B), axis=1)
    
    # turn An and Bn into column vectors
    An = np.reshape(An, (m, 1)) 
    Bn = np.reshape(Bn, (n, 1))
 
    C = -2 * np.dot(A, np.transpose(B))
    C += An # add each column of C by column vector An
    C += np.transpose(Bn) # add each column of C by row vector Bn
    
    return np.sqrt(C) # element wise sqaure root every value in matrix C

"""
Calls helper function fastPDist() that performs distance calulations then creates a pandas data frame with the returned data adding the row and column names
input: pandas data frames for feature and cell coordinates
returns: a pandas data frame with the distance between each pair of cells and genes where rows are genes (features) and columns are cells 
"""
def pairDist(features: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    res = fastPDist(features, cells)
    res_df = pd.DataFrame(res, index = features.index, columns = cells.index)
    return res_df