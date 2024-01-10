import numpy as np

"""
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
    along with this program.  If not, see <https://www.gnu.org/licenses>."""
    
# Reference: https://github.com/RausellLab/CelliD


"""
Computes fuzzy matrix and matrix of standardized relative frequencies
input: 
    - arr: a 2D numpy array with cells as rows and genes as columns, consisting of preprocessed raw gene count data
returns a dictionary:
    - Z: S_N,K, matrix of standardized relative frequencies
    - D_r: D_r ^-1/2, vector representing diagonal matrix of row sums of the matrix of relative frequencies ^-1/2
    - D_c: D_c ^-1/2, vector representing diagonal matrix of column sums of the matrix of relative frequencies ^-1/2
    - X: fuzzy-coded indicator matrix 
"""
def MCAStep1(arr: np.ndarray) -> dict:    
    ############## Generate fuzzy-coded indicator matrix: ############
    # rmin , rmax , and range_list are vectors
    rmin = np.min(arr, axis=0) # get min val in each gene column -> min(M_p) 
    rmax = np.max(arr, axis=0) # get max val in each gene column -> max(M_p)
    range_list = rmax - rmin  # max(M_p) - min(M_p) for each gene row p
    # m_np − min(M_p) -> subtract the min value for a row from each scalar in a row:
    for index in range(len(rmin)): # iterate column by column and value by value in rmin
        arr[:, index] -= rmin[index]
    # m_np − min(M_p) / max(M_p) - min(M_p)
    for index in range(len(range_list)):
        arr[:, index] /= range_list[index]
        arr[:, index] = np.nan_to_num(arr[:, index]) # turns nan values to 0 in case where a number is divided by 0
    
    # Fuzzy matrix stored in FM, size: N cells, K = 2P genes 
    # below puts fuzzy matrix in cellid paper intended format of complementary values (by columns)
    complement = 1 - arr
    rows, columns = np.shape(arr)
    FM = np.empty(shape=(rows, columns * 2)) # fuzzy matrix ends up with total of K = 2P categories/columns
    index = 0
    for column in range(0, np.shape(FM)[1]):
        if column % 2 == 0:
            FM[:, column] = arr[:, index] # x_p+ stored in even columns
        else:
            FM[:, column] = complement[:, index] # x_p- complement stored in odd columns
            index += 1
    X=FM # fuzzy matrix stored for later reference
    # -------------------------------------- End of generating fuzzy matrix 
    
    ############## Generate matrix of standardized relative frequencies #########
    # S = D_r ^−1/2 * FM * D_c ^−1/2  
    # OR  
    # S = (1 / D_r ^1/2) * FM  * (1 / D_c ^1/2) -> the inverse of diagonal matrices are simply the reciprocal of each value
       
    # Matrix of relative frequencies: 
    # F_N,K = 1/NP * X , where there are N cells and P genes
    FM = np.divide(FM , (np.shape(arr)[0] * np.shape(arr)[1]))  # scales/shrinks the data. Does not affect coordinate position of cells, utlimately not affecting final prediction
    # -------------------------------------- 
    
    # D_r here is a vector (in paper is a diagonal matrix) of sums of each row, size n cells 
    D_r = np.sum(FM, axis=1)
    
    # D_c here is a vector of sums of each column, size K (2P genes)
    D_c = np.sum(FM, axis=0) # D_c -> vector containing sum of each row in FM
        
    # order of multiplying following matrices DOES matter:
    
    # (1 / D_r ^1/2) * FM
    # update D_r to be D_r ^ -1/2
    D_r = np.sqrt(np.reciprocal(D_r))
    # nxn diagonal matrix * nxk matrix = nxk , essentially just multiply the the nth row in FM by nth value in D_r
    for n in range(len(D_r)):
        FM[n] *= D_r[n] 
        
    # ((1 / D_r ^1/2) * FM)  * (1 / D_c ^1/2)
    # update D_c to be D_c ^ -1/2
    D_c = np.sqrt(np.reciprocal(D_c))
    # nxk matrix * kxk diagonal matrix = nxk , essentially just multiply the the ith column in FM by ith value in D_c
    for i in range(len(D_c)):
        FM[:,i] *= D_c[i]
    # -------------------------------- End of Generating matrix of standardized relative frequencies
    print("Matrix of standardized relative frequencies:\n", FM, "\n", np.shape(FM))
    print("D_r ^-1/2:\n", D_r, "\n", np.shape(D_r))
    print("D_c ^-1/2:\n", D_c, "\n", np.shape(D_c))
    print("Fuzzy-coded indicator matrix:\n", X, "\n", np.shape(X))

    return {"Z": FM,    # S_N,K, matrix of standardized relative frequencies
            "D_r": D_r, # D_r ^-1/2
            "D_c": D_c, # D_c ^-1/2
            "X": X      # Fuzzy-coded indicator matrix
            }

"""
Computes cell (row standard) and gene (column principal) coodinates
input:
    - S: matrix of standardized relative frequencies
    - U: numpy 2D array of left singuar vectors as columns from SVD computation of S
    - D_r: numpy vector representing D_r ^-1/2
    - D_c: numpy vector representing D_c ^-1/2
returns a dictionary:
    - cellCoordinates: numpy 2D array of cells coordinates where rows are cells and there are j number of columns where columns represent dimensions
    - geneCoordinates: numpy 2D array of gene coordinates where rows are features (genes) and there are j number of columns where columns represent dimensions
"""
def MCAStep2(S: np.ndarray, U: np.ndarray, D_r: np.ndarray, D_c: np.ndarray) -> dict:
    ################## Row standard coordinates : Φ = D_r ^−1/2 * U = D_r ^−1/2 * S * V * D_a ^−1 ################    
    # Φ = D_r ^-1/2 * U
        # D_r size: n cells vector
        # U size: n cells x j dimensions
        # Φ size: n cells x j dimensions
    row_coordinates = np.empty(shape=(np.shape(U)))
    # https://solitaryroad.com/c108.html
    # diagonal matrix = [[k1,0,0], [0,k2,0], [0,0,k3]]
    # matrix = [[x11,x12,x13], [x21,x22,x23], [x31,x32,x33]]
    # diagonal matrix * matrix = [[k1*x11, k1*x12, k1*x13], [k2*x21, k2*x22, k2*x23], [k3*x31, k3*x32, k3*x33]]
    for i in range(np.shape(U)[1]):
        row_coordinates[:,i] = D_r * U[:,i]
    
    ################## Column principal coordinates : G = D_c ^−1/2 * S^T * U = D_c ^−1/2 * V * D_a #############
    # G = D_c ^−1/2 * S^Transposed * U
        # D_c size: k vector where k = 2P genes
        # S^T size: k x n cells where k = 2p genes
        # U   size: n cells x j dimensions
        # G   size: k x j dimensions where k = 2p genes
    S = np.transpose(S) # S^T
    column_coordinates = np.empty(shape=(np.shape(S)))
    # diagonal matrix * matrix = [[k1*x11, k1*x12, k1*x13], [k2*x21, k2*x22, k2*x23], [k3*x31, k3*x32, k3*x33]]
    for i in range(np.shape(column_coordinates)[1]): # (D_c ^−1/2 * S^Transposed)
        column_coordinates[:,i] = D_c * S[:,i]
    column_coordinates = np.dot(column_coordinates, U) # (D_c ^−1/2 * S^Transposed) * U
    
    # only keep + complememnt version of each gene
    rows, columns = np.shape(column_coordinates)
    ret = np.empty(shape=(int(rows/2), columns)) # fuzzy matrix ends up with total of K = 2P categories/columns
    index = 0
    for row in range(rows):
        if row % 2 == 0:
            ret[index] = column_coordinates[row]
            index += 1
    
    return {"cellCoordinates": row_coordinates,
            "geneCoordinates": ret }