import numpy as np
import pandas as pd
import mca_steps, csv, time, scipy

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
    
#Reference: https://github.com/RausellLab/CelliD
    
    
# MCA CLASS responsible for holding the results of MCA, mainly being the cell and gene coordinates, 
# which are pandas dataframes each containing a 2D matrix of numerical values, and the fuzzy-coded indicator matrix
class MCA:
    def __init__(self, cellCoordinates: np.ndarray, geneCoordinates: np.ndarray, X: np.ndarray, 
                genesN: list, cellsN: list, j: int):
        mca_strings = [f"MCA_{i}" for i in range(1, j+1)]
        df1 = pd.DataFrame(cellCoordinates, index=cellsN, columns=mca_strings)
        df2 = pd.DataFrame(geneCoordinates, index=genesN, columns=mca_strings)        
        self.cellCoordinates = df1
        self.geneCoordinates = df2
        self.X = X # fuzzy-coded indicator matrix

"""
Performs MCA (multiple correspondence analysis) on input data (preprocessed raw gene count data), generating cell and gene coordinates in j dimensional space

input:
    - arr: a pandas data frame with cells as rows and genes as columns
    - j: number of dimensions in coordinate space to compute and store, default set to 50
    - genes: String vector of gene names that want to be used. If not specified all genes will be used.
    - write_results_to_csv: If True, will write several results from MCA method to csv files.
output:
    - MCA object containing:
        - cellCoordinates: pandas data frame where rows are cells and columns are j (default value for j is 50) different dimensions
        - geneCoordinates: pandas data frame where rows are genes and columns are j (default value for j is 50) different dimensions
        - X: fuzzy-coded indicator matrix
"""
def RunMCA(arr: pd.DataFrame, j: int = 50, genes: list = None, write_results_to_csv = False):
    # preprocessing matrix
    # ------------------------------------------------
    
    # genes if passed a value should be a list of specific column names genes as strings that want to be used
    if (genes is not None):
        arr = arr.loc[:, genes]
    
    # Remove columns with all zeros
    arr = arr.loc[:, (arr != 0).any(axis=0)]

    # Remove columns with duplicated column names
    arr = arr.loc[:, ~arr.columns.duplicated()]
    
    # Get row names (cell names) as 'cellsN' and column names (gene names) as 'genesN' after the preprocessing step
    cellsN = arr.index
    genesN = arr.columns
    
    arr = arr.to_numpy()
    # -------------------------------------------------
    
    ######### Fuzzy Matrix computation- MCAStep1 #########
    start_time = time.time()
    
    print("Computing Fuzzy Matrix...")
    MCAPrepRes = mca_steps.MCAStep1(arr)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds\n") # Timer used to determine how long it took to run MCAStep1
    
    if write_results_to_csv:
        with open("Results_csv/Z_matrix_stand_rel_freq.csv", 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for row in MCAPrepRes["Z"]:
                csv_writer.writerow(row)
        with open("Results_csv/D_r_neg_one_half.csv", 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(MCAPrepRes["D_r"])
        with open("Results_csv/D_c_neg_one_half.csv", 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(MCAPrepRes["D_c"])
        with open("Results_csv/X_fuzzy_coded_matrix.csv", 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for row in MCAPrepRes["X"]:
                csv_writer.writerow(row)
    #######################################################
    
    ########## Singular Value Decomposition (SVD) computation ############      
    start_time = time.time()

    print("Computing SVD...")
    # partial Singular Value Decomposition        
    # input matrix size: n cells x d columns   (d = 2p genes)
    # parameter k: number of singular values and vectors to compute (default is 50)
    # SVD returns:
        # U, size: n x k, (left singular vectors as columns) 
        # S, size: k vector, (singular values)
        # V_Transposed, size: , (right singular vectors as rows)
    U, S, Vh = scipy.sparse.linalg.svds(MCAPrepRes["Z"], k=j, which='LM') 
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds\n")
        
    # Note: In the context of Singular Value Decomposition (SVD), the sign of the values in the matrices does not matter for the mathematical correctness or validity of the decomposition. The singular value decomposition is unique up to a sign convention.
        
    # reshape and transpose Vh to allow MCAStep2() to multiply matrix of standardized relative frequencies by Vh
    # SVD returns Vh in transposed form so must undo to get right singular vectors as columns
    Vh = np.transpose(Vh)
    
    # put singular values in descending order 
    S = np.flip(S)
    
    print("SVD (Singular Value Decomposition) of standardized relative frequencies matrix:")
    print("U (left singular vectors as columns):\n", U, "\n", np.shape(U))
    print("S (singular values):\n", S, "\n", np.shape(S))
    print("Vh (right singular vectors as columns):\n", Vh, "\n", np.shape(Vh))
    
    if write_results_to_csv:
        with open("Results_csv/PartialSVD_U.csv", 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for row in U:
                csv_writer.writerow(row)
        with open("Results_csv/PartialSVD_S.csv", 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(S)
        with open("Results_csv/PartialSVD_Vh.csv", 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for row in Vh:
                csv_writer.writerow(row)
    #####################################################
    
    ############## Coordinate Computations- MCAStep2 ###############
    start_time = time.time()

    print(f"Computing Coordinates for a {j} dimensional space...")
    coordinates = mca_steps.MCAStep2(S = MCAPrepRes["Z"], U=U, D_r=MCAPrepRes["D_r"], D_c=MCAPrepRes["D_c"]) # main logic
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds\n")
    
    # MCA is an object that stores gene and cell coordinates, and fuzzy coded indicator matrix
    mca = MCA(cellCoordinates=coordinates["cellCoordinates"], geneCoordinates=coordinates["geneCoordinates"],
                X=MCAPrepRes["X"], genesN=genesN, cellsN=cellsN, j=j)
    
    print("Cell coordinates:\n", mca.cellCoordinates)
    print("Gene coordinates:\n", mca.geneCoordinates)
    
    if write_results_to_csv:
        mca.geneCoordinates.to_csv('Results_csv/geneCoordinates.csv')
        mca.cellCoordinates.to_csv('Results_csv/cellCoordinates.csv')
    ################################################################
    
    return mca