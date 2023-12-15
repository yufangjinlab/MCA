import numpy as np
import pandas as pd
import mca_steps, csv, time, scipy

"""
Ramin Mohammadi. mca.py contains the RunMCA() function which performs MCA (multiple correspondence analysis) on input data
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
    
#Reference: https://github.com/RausellLab/CelliD
    
    
    

# MCA class responsible for holding the results of MCA mainly being the cell and gene coordinates which are pandas dataframes each containing a 2D matrix of numerical values
# creating a new MCA object by using it's constructor creates two new pandas data frames for the cell and gene coordinates
class MCA:
    # constructor
    def __init__(self, cellsCoordinates: np.ndarray, featuresCoordinates: np.ndarray):
        df1 = pd.DataFrame(cellsCoordinates)
        df2 = pd.DataFrame(featuresCoordinates)
        self.cellsCoordinates = df1
        self.featuresCoordinates = df2 


"""
Performs MCA (multiple correspondence analysis) on input data
input:
    - X: a pandas data frame with labelled rows (genes) and columns (cells)
    - nmcs: number of components to compute and store, default set to 50
    - features: String vector of feature names that want to be used. If not specified all features will be taken.
output:
    - MCA object containing:
        - featuresCoordinates: pandas data frame where rows are genes and columns are nmcs (default value for nmcs is 50) different dimensions
        - cellsCoordinates: pandas data frame where rows are cells and columns are nmcs (default value for nmcs is 50) different dimensions
        - stdev: numpy array containing the singular values created during MCA when performing Singular Value Decomposition
"""
def RunMCA(X: pd.DataFrame, nmcs: int = 50, features: list = None):
    # preprocessing matrix
    # ------------------------------------------------
    
    # features if passed a value should be a list of specific row names (features) as strings that want to use
    if (features is not None):
        X = X.loc[features]
    
    # Remove rows with all zeros in the columns
    X = X.loc[(X != 0).any(axis=1)]

    # Remove rows with empty row names
    X = X[X.index.str.len() > 0]

    # Remove rows with duplicated row names
    X = X.loc[~X.index.duplicated()]
    
    X_arr = X.to_numpy()

    #Get column names (cell names) as 'cellsN' and row names (gene names) as 'featuresN' after the preprocessing step
    cellsN = X.columns
    featuresN = X.index
    # -------------------------------------------------
    
    ######### Fuzzy Matrix computation- MCAStep1 #########
    start_time = time.time()
    
    print("Computing Fuzzy Matrix")
    MCAPrepRes = mca_steps.MCAStep1(X_arr)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds") # Timer used to determine how long it took to run MCAStep1
    
    # Optionally print results of MCAStep1
    # print("MCAStep1 results:")
    # print("Z", MCAPrepRes["Z"].shape, MCAPrepRes["Z"][:19, :5])
    # print("Dc", MCAPrepRes["Dc"].shape, MCAPrepRes["Dc"])
    
    # Optionally write results of MCAStep1 to csv files to see entire results
    # with open("Results_csv/MCAStep1_Z.csv", 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     for row in MCAPrepRes["Z"]:
    #         csv_writer.writerow(row)
    # with open("Results_csv/MCAStep1_Dc.csv", 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     for row in MCAPrepRes["Dc"]:
    #         csv_writer.writerow(row)
    #######################################################
    
    ########## Singular Value Decomposition (SVD) computation ############      
    start_time = time.time()

    print("Computing SVD")
    # Python's U = R's u (Left singular vectors)
    # Python's S = R's d (Singular values)
    # Python's Vh = R's v (Right singular vectors)
    #U, S, Vh = np.linalg.svd(MCAPrepRes["Z"]) # full svd
    
    #partial SVD, only takes 33 seconds with Baron compared to 5 minutes for full svd
    # U, S, and Vh generate same output as in R
    U, S, Vh = scipy.sparse.linalg.svds(MCAPrepRes["Z"], k=nmcs+1, which='LM') 
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    
    # make U (left singular vectors) only keep first value in every row b/c that is what is done in R's irlba svd computation
    #U = U[:, 0].reshape(np.shape(U)[0], 1) # full svd
    
    #Note: In the context of Singular Value Decomposition (SVD), the sign of the values in the matrices does not matter for the mathematical correctness or validity of the decomposition. The singular value decomposition is unique up to a sign convention.

    # take only the first nmcs singular values
    S = S[:nmcs+1]
    
    # reshape and transpose Vh to match with R output of performing SVD with irlba package
    Vh = np.transpose(Vh)  # partial and full svd
    
    #partial svd results match R's but correct if reading python out put right to left sp reverse column orders
    Vh = np.fliplr(Vh)
    U = np.fliplr(U) 
    S = np.flip(S)
    
    #full svd vh editing
    #Vh = np.reshape(Vh, (np.shape(Vh)[1], np.shape(Vh)[0])) 
    # # make Vh (right singular vectors) only take first nmcs columns b/c that is what is done in R output of performing SVD with irlba package
    # Vh = Vh[:, :nmcs+1]
    
    # Optionally print results of SVD
    # print("SVD results:")
    # print("U", U.shape, U)
    # print("S", S.shape, S)
    # print("Vh", Vh.shape, Vh)
    
    # Optionally write results of SVD to csv files
    # with open("Results_csv/PartialSVD_U.csv", 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     for row in U:
    #         csv_writer.writerow(row)
    # with open("Results_csv/PartialSVD_S.csv", 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(S)
    # with open("Results_csv/PartialSVD_Vh.csv", 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     for row in Vh:
    #         csv_writer.writerow(row)
    #####################################################
    
    ############## Coordinate Comutations- MCAStep2 ###############
    start_time = time.time()

    print("Computing Coordinates")
    coordinates_list = mca_steps.MCAStep2(Z = MCAPrepRes["Z"], V = Vh, Dc = MCAPrepRes["Dc"])
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    
    # MCA is an object that will create and hold two pandas data frames: cellCoordinates, featureCoordinates (gene coordinates)
    mca = MCA(cellsCoordinates=coordinates_list["cellsCoordinates"], featuresCoordinates=coordinates_list["featuresCoordinates"])
    
    # exclude first column in cellCoordinates b/c it is just -1 that are not shown in R's cellCoordinates first column
    mca.cellsCoordinates = mca.cellsCoordinates.iloc[:, 1:]
    # likewise, exclude first column in featureCoordinates b/c it is just -1 that are not shown in R
    mca.featuresCoordinates = mca.featuresCoordinates.iloc[:, 1:]
    
    # add row names (feature and cell names) for the pandas data frames
    mca.featuresCoordinates.index = featuresN
    mca.cellsCoordinates.index = cellsN
     
    # add column names (ith dimension) for the pandas data frames
    mca_strings = [f"MCA_{i}" for i in range(1, nmcs+1)]
    mca.cellsCoordinates.columns = mca_strings
    mca.featuresCoordinates.columns = mca_strings
    
    print("Features coordinates shape: ", mca.featuresCoordinates.shape, mca.featuresCoordinates)
    print("Cells coordinates shape: ", mca.cellsCoordinates.shape, mca.cellsCoordinates)
    
    # Store singular values in MCA object
    mca.stdev = S[1:] # exclude first singular values
    ################################################################
    
    return mca





