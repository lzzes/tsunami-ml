#import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

from sklearn.model_selection import train_test_split

#import tempfile
import torch
#import torch.nn as nn

#from tqdm import tqdm

from pathlib import Path

from model_gcn import RuptureNet2D

from datasets_gcn import BlockDataset, load_data, make_blocks

import numpy as np
import time
import pandas as pd


def main(Train_flag, Test_flag, checkpoint_no):

    config = {
        'batch_size': 64,
        'lr': 1e-4,
        'gcn_in_features': 4,
        'gcn_hidden_dim': 128,
        'gcn_out_dim': 256,
        'conv1_dim': 256, #increase dimensions
        'conv1_layers': 2, 
        'num_transformer_blocks': 6, ###
        'num_transformer_heads': 8,
        'transformer_hidden_dim': 768, # inc from 256
        'conv2_dim': 256, # inc from 32
        'dropout_pro': 0.4, # inc from 0.3
        }
    # Hyperparameters
    num_epochs = 51
    debug = False

    # Dataset

    data_cache_path = Path("./data_cache").resolve()
    data_cache_path.mkdir(parents=True, exist_ok=True)

    if (data_cache_path / "done.flag").exists():
        print("Data already cached.")
    else:
        df_x, df_y = load_data()
        train_blocks, test_blocks = make_blocks(df_x, df_y)

        # Save each block as a separate file
        for i, block in enumerate(train_blocks):
            torch.save(block, data_cache_path / f"train_{i}.pt")
        for i, block in enumerate(test_blocks):
            torch.save(block, data_cache_path / f"test_{i}.pt")

        # marker file
        (data_cache_path / "done.flag").touch()
        print("Data cached.")

    # Store filenames
    train_files = sorted([f for f in data_cache_path.glob("train_*.pt")])
    test_files = sorted([f for f in data_cache_path.glob("test_*.pt")])

    # Create blocks
    train_blocks = [torch.load(f, map_location='cpu', weights_only=True) for f in train_files]
    test_blocks = [torch.load(f, map_location='cpu', weights_only=True) for f in test_files]
    
    test_dataset = BlockDataset(test_blocks, debug=debug)

    # Model
    model = RuptureNet2D(config)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Load edges
    edge_index = np.loadtxt("fault_edges.txt", dtype=int)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Define model
    model = model.to(device)


 
    model.load_state_dict(torch.load("Training_GCN_v7/best_model.pth"))

    model.eval()

    print("Model loaded in eval mode")

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)


    count = 0

    # Run through model
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader):
            preds = model(inputs,edge_index)
            
            preds = preds.detach().cpu().numpy().flatten()
            targets = targets.detach().cpu().numpy().flatten()

            pdf = pd.DataFrame({"preds": preds})
            pdf.to_csv(f"Testing_GCN_v7/eval-{idx}-pred.csv", index = False)

            tdf = pd.DataFrame({"targets": targets})
            tdf.to_csv(f"Testing_GCN_v7/eval-{idx}-true.csv", index = False)
            print(count)
            count += 1

    inputs_single = inputs[0:1]  # reuse last batch, take one sample
    start = time.perf_counter()
    pred_single = model(inputs_single, edge_index)
    end = time.perf_counter()
    print(f"Single sample inference: {(end - start) * 1000:.3f} ms")

        # To numpy array
#         y_pred_np = y_pred.detach().cpu().numpy()
#        y_true_np = y_true.detach().cpu().numpy()

        #print(y_pred_np.type())
#        print(y_pred_np)
#        print(y_true_np)

if __name__ == "__main__":
    main(Train_flag=0, Test_flag=1, checkpoint_no=0)
    
    
