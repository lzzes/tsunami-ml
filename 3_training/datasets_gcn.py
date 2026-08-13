import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

def read_input(csv_files):
    df = pd.read_csv(csv_files, index_col= False,
                    header = 0,
                    names = ["Lon", "Lat", "Depth", "Slip"])
    return df

def read_output(csv_files):
    df = pd.read_csv(csv_files, index_col=False,
                     header = 0,
                     names = ["Lon", "Lat", "Arrival", "MaxHeight", "InundateT","InundateH"])
    return df

def load_data():
    # Read all csv files for training
    input_data_frames = []
    output_data_frames = []

    # Read input csv files and save
    for csv_file in input_csv_files:
        df_x = read_input(csv_file)
        input_data_frames.append(df_x)
    
    # Combine into a large input dataframe
    df_x = pd.concat(input_data_frames, axis=0, ignore_index=True)

    # Read output csv files and save
    for csv_file in output_csv_files:
        df_ycsv = read_output(csv_file)
        output_data_frames.append(df_ycsv)

    # Combine into a large output dataframe
    df_ycsv = pd.concat(output_data_frames, axis=0, ignore_index=True)

    # Slice off lat/lon points in output grid
    df_y = df_ycsv.iloc[:,2:]

    return df_x, df_y

def make_blocks(df_x, df_y, x_block_size=1896, y_block_size=520):        # updated for shorter version from 175 to100
    ''' 
    Splits x and y data frames into appropriately sized blocks,
    splits into train and test sets, and normalizes the train
    and test sets separately
    '''
    # Compute number of blocks
    num_blocks_x = len(df_x) // x_block_size
    num_blocks_y = len(df_y) // y_block_size

    # Check number of input/output blocks is the same
    if num_blocks_x == num_blocks_y:
        print(f"Number of blocks: {num_blocks_x}")
    else:
        raise Exception("Number of input and output blocks not equal")
    
    # Split dataframes into blocks
    input_blocks = [(df_x[i*x_block_size:(i+1)*x_block_size])for i in range(num_blocks_x)]
    output_blocks = [(df_y[i*y_block_size:(i+1)*y_block_size])for i in range(num_blocks_y)]

    # Establish train/test indices
    train_idx, test_idx = train_test_split(range(len(input_blocks)), test_size=0.2, random_state=218)

    # Save indices
    np.savetxt("test_idx.txt", test_idx)
    np.savetxt("train_idx.txt", train_idx)

    # Create lists of blocks
    train_input_blocks = [input_blocks[i] for i in train_idx]
    train_output_blocks = [output_blocks[i] for i in train_idx]
    test_input_blocks = [input_blocks[i] for i in test_idx]
    test_output_blocks = [output_blocks[i] for i in test_idx]

    # Concatenate for normalization
    train_x = pd.concat(train_input_blocks, axis=0, ignore_index=True)
    train_y = pd.concat(train_output_blocks, axis=0, ignore_index=True)
    test_x = pd.concat(test_input_blocks, axis=0, ignore_index=True)
    test_y = pd.concat(test_output_blocks, axis=0, ignore_index=True)

    # Apply mean-stdev standardization
    def standardize_input(df, num_col=4):
        '''
        Standardizes columns 
        Must specify number of columns in dataframe
        '''
        for col in range(num_col):
            col_mean = np.mean(df.iloc[:,col])
            col_std = np.std(df.iloc[:,col])

            if col_std==0:
                df.iloc[:,col] = 0
            else:
                df.iloc[:,col] = (df.iloc[:,col].values - col_mean) / col_std
        return df
    
    def standardize_output(df, statistics = [0,0,0,0]):
        '''
        Standardizes columns 
        Must specify number of columns in dataframe
        Does not consider rows without data
        '''

        df_standardized = df.copy()

        # Replace NAN values with large sentinel value
        arr_time = df_standardized['Arrival']
        time_mask = df_standardized['InundateT'].astype(bool)
        df_standardized.loc[~time_mask, 'Arrival'] = 9999999

        # Compute column statistics of inundated areas
        if statistics[0]==0:
            t_mean = arr_time[time_mask].mean()
            t_std = arr_time[time_mask].std()
        else:
            t_mean = statistics[0]
            t_std  = statistics[1]

        # Normalize time data
        df_standardized.loc[time_mask, "Arrival"] = (df_standardized.loc[time_mask, 'Arrival'] - t_mean)/t_std

        # Replace NAN height values
        height = df_standardized['MaxHeight']
        height_mask = df_standardized['InundateH'].astype(bool)
        df_standardized.loc[~height_mask,'MaxHeight'] = 9999999

        def z_standardize_height(df_st, h, mask):

            # Compute height statistics
            if statistics[2]==0:
                h_mean = h[mask].mean()
                h_std = h[mask].std()
            else:
                h_mean = statistics[2]
                h_std  = statistics[3]

            # Normalize height data
            df_st.loc[mask, "MaxHeight"] = (df_st.loc[mask, 'MaxHeight'] - h_mean)/h_std

            return h_mean, h_std
        
        def minmax_standardize_height(df_st, h, mask):

            hmax = np.max(df_st.loc[mask, 'MaxHeight'])
            hmin = np.min(df_st.loc[mask, 'MaxHeight'])

            df_st.loc[mask, "MaxHeight"] = (df_st.loc[mask,'MaxHeight']-hmin)/(hmax - hmin)

            return hmax, hmin
        
        h_mean, h_std = height[height_mask].mean(), height[height_mask].std()
        #h_mean, h_std = z_standardize_height(df_standardized, height, height_mask)

        # h_max, h_min = minmax_standardize_height(df_standardized, height, height_mask)

        # h_mean = h_max
        # h_std = h_min

        stats = [t_mean, t_std, h_mean, h_std]

        return df_standardized, stats
    

    
    # Standardize data
    train_x = standardize_input(train_x)
    train_y, stats = standardize_output(train_y)
    test_x  = standardize_input(test_x)
    test_y, test_stats  = standardize_output(test_y, statistics=stats)

    np.savetxt("output-stats.txt", stats, header="T mean T std H min H max")


    # Split blocks into tensors
    train_input  = [torch.tensor(train_x[i*x_block_size:(i+1)*x_block_size].values, dtype =torch.float32) 
                    for i in range(len(train_input_blocks))]
    train_output = [torch.tensor(train_y[i*y_block_size:(i+1)*y_block_size].values, dtype =torch.float32) 
                    for i in range(len(train_output_blocks))]
    test_input   = [torch.tensor(test_x[i*x_block_size:(i+1)*x_block_size].values, dtype = torch.float32) 
                    for i in range(len(test_input_blocks))]
    test_output  = [torch.tensor(test_y[i*y_block_size:(i+1)*y_block_size].values, dtype = torch.float32) 
                    for i in range(len(test_output_blocks))]

    # Create paired tuples for blocks
    train_blocks = list(zip(train_input, train_output))
    test_blocks = list(zip(test_input, test_output))

    return train_blocks, test_blocks

class BlockDataset(Dataset): # Not used, see Lazy Dataset in train_test_cnn.py
    '''Dataset for list of blocks. each block is (data, label) tensor created by make_blocks'''

    def __init__(self, blocks, debug=False):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        return self.blocks[index]

input_filelist = np.loadtxt("Input7/input_filelist.txt",dtype=str)
input_csv_files = [f"Input7/{fname}" for fname in input_filelist]

output_filelist = np.loadtxt("/u/home/l/lzzes/project-lsmeng/Output9/output_filelist.txt",dtype=str)
output_csv_files = [f"/u/home/l/lzzes/project-lsmeng/Output9/{fname}" for fname in output_filelist]
