import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

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
Euclidean distance calculation between gene and cell coordinates among j dimensions. 
The closer a gene g is to a cell c, the more specific to such a cell it can be considered.

input:
    - cellCoordinates
    - geneCoordinates 
    - genes_filter: String vector of genes names to subset from gene coordinates. If not specified will use all genes 
    - cells_filter: String vector of cell names to subset from cell coordinates. If not specified will use all cells 
    - write_dist_to_csv: if True, writes the resulting euclidean distance matrix to a .csv file
return:
    - CellGeneDistances: numpy 2d array with genes as rows and cells as columns. Each value represents the distance between gene x and cell y
"""
def GetDistances(cellCoordinates: pd.DataFrame, geneCoordinates: pd.DataFrame, X: np.ndarray = None, barycentric=False, 
                 cells_filter: list = None, genes_filter: list = None, write_dist_to_csv = False):
    # filter genes by a list of specific genes as strings that want to be used
    if (genes_filter is not None):
        geneCoordinates = geneCoordinates.loc[genes_filter]
    # filter cells by a list of specific cells as strings that want to be used
    if (cells_filter is not None):
        cellCoordinates = cellCoordinates.loc[cells_filter]
            
    print("\nCalculating distances between cells and genes...\n")
    # calculate euclidean distance between cells and genes
    CellGeneDistances = dist(cells=cellCoordinates, genes=geneCoordinates, X=X, barycentric=barycentric)
    print("Distance matrix:\n", CellGeneDistances)
    if write_dist_to_csv:
        CellGeneDistances.to_csv('Results_csv/CellGeneDistances.csv')
    
    return CellGeneDistances

"""
Euclidean distance calculations between every gene and cell based off the gene and cell coordinates among j dimensions
input: 
    - genes: pandas data frame for GENE coordinates
    - cells: pandas data frame for CELL coordinates 
    - X: fuzzy coded indicator matrix 
    - barycentric: if True, will calculate gene coordinates through barycentric relationship with cell coordinates using X
returns: 
    -a 2D numpy array of distances between each pair of cells and genes where rows are genes (features) and columns are cells
"""
# At this stage, only gene category coordinates g_p+ , conveying presence of gene expression relative to the maximum per gene, are retained for downstream analysis.
def dist(cells: pd.DataFrame, genes: pd.DataFrame, X: np.ndarray, barycentric: bool) -> np.ndarray:
    
    if(not barycentric):
        # euclidian distance between gene p and cell n among all j dimensions
        # cell matrix: n cells x j dimensions
        # gene matrix: p genes x j dimensions
        euclidean_dist = euclidean_distances(genes.to_numpy(), cells.to_numpy())
 
    # barycentric relationship gene coordinates: g_kj = (1 / N∑n=1 x_nk) * N∑n=1 (x_nk ϕ_nj)
    else:       
        print("Finding gene coordinates through barycentric relationship: ")

        g = np.empty(shape=(np.shape(X)[1], np.shape(cells)[1])) 
        rows, cols = np.shape(g)
        x = np.sum(X, axis=0)
        for k in range(rows):
            for j in range(cols):
                g[k,j] = (1 / x[k] ) * np.sum(X[:,k] * cells.iloc[:, j])
        
        # take only g+ so even rows in g
        g_plus = np.empty(shape=(int(np.shape(X)[1] / 2), np.shape(cells)[1]))
        index = 0
        for row in range(0, np.shape(g)[0]):
            if row % 2 == 0:
                g_plus[index] = g[row]
                index += 1
        euclidean_dist = euclidean_distances(g_plus, cells.to_numpy()) 
                
    return pd.DataFrame(euclidean_dist, index = genes.index, columns = cells.index)