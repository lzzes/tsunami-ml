from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

from sklearn.model_selection import train_test_split


import torch

from pathlib import Path

from model_gcn import RuptureNet2D

from datasets_gcn import BlockDataset, load_data, make_blocks

from loss_gcn import loss_calc


import numpy as np
import os
import time
import pandas as pd

class LazyDataset(Dataset):
    '''Dataset that loads blocks from disk on demand to save RAM'''

    def __init__(self, block_files, debug=False):
        """
        Args:
            block_files: List of file paths to saved blocks, or directory containing block files
            debug: Enable debug mode
        """
        self.debug = debug
        
        # Handle both list of files and directory path
        if isinstance(block_files, str):
            # Assume it's a directory path
            self.block_files = [
                os.path.join(block_files, f) 
                for f in sorted(os.listdir(block_files))
                if f.endswith(('.pt', '.pth', '.pkl'))
            ]
        else:
            # Assume it's already a list of file paths
            self.block_files = block_files
            
        if self.debug:
            print(f"LazyBlockDataset initialized with {len(self.block_files)} block files")

    def __len__(self):
        return len(self.block_files)

    def __getitem__(self, index):
        """Load and return block from disk"""
        filepath = self.block_files[index]
        
        try:
            block = torch.load(filepath, map_location='cpu',weights_only=True)
                    
            if self.debug:
                print(f"Loaded block from {filepath}")
                
            return block
            
        except Exception as e:
            if self.debug:
                print(f"Error loading {filepath}: {e}")
            raise

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
    # train_dataset = LazyDataset(train_files, debug=debug)
    # test_dataset = LazyDataset(test_files, debug=debug)

    train_blocks = [torch.load(f, map_location='cpu', weights_only=True) for f in train_files]
    test_blocks = [torch.load(f, map_location='cpu', weights_only=True) for f in test_files]
    
    train_dataset = BlockDataset(train_blocks, debug=debug)
    test_dataset = BlockDataset(test_blocks, debug=debug)

    # Model
    model = RuptureNet2D(config)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Load from Checkpoint
    if checkpoint_no != 0:
        checkpoint_path = torch.load(f"Training_GCN_v7/checkpoint_{checkpoint_no}.pth", map_location="cpu")
        model.load_state_dict(checkpoint_path['model_state_dict'])

    # Load edges
    edge_index = np.loadtxt("fault_edges.txt", dtype=int)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Define model
    model = model.to(device)

    if Train_flag==1:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        # Find best model later
        best_val_loss = 5.084

        # Prepare data
        train_idx, val_idx = train_test_split(range(len(train_dataset)), train_size=0.75, random_state=218)

        train_split = SubsetRandomSampler(train_idx)
        val_split = SubsetRandomSampler(val_idx)

        batch_size = config['batch_size']

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=train_split)        # reduced from 8 to 2
        val_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=val_split)            # reduced from 8 to 2


        # Training loop
        for iteration in range(num_epochs):

            epoch = iteration+1+checkpoint_no

            model.train()
            total_loss = 0.0

            
            batch_no = 1

            for batch in train_dataloader:
                
                if (batch_no)%5==0:
                    print(f"Batch: {batch_no}")
                
                # TRAINING

                inputs, targets = batch
                inputs, targets = (
                    inputs.to(device),
                    targets.to(device)
                )

                optimizer.zero_grad()
                outputs = model(inputs, edge_index)

                loss = loss_calc(outputs, targets)

                print(f"Train Loss = {loss.item():.5f}")

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Save the loss
                total_loss += loss.item()

                batch_no += 1
                del inputs, targets, outputs, loss

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch}/{num_epochs} Train Loss: {avg_train_loss:.5f}")

            ##################
            # VALIDATION
            ##################

            val_loss = 0.0
            model.eval()

            with torch.inference_mode():
                for batch in val_dataloader:
                    # Load validation data
                    inputs, targets = batch
                    inputs, targets = (inputs.to(device), targets.to(device))
                    
                    # Send through model
                    outputs = model(inputs, edge_index)
                
                    # Calculate loss
                    loss = loss_calc(outputs, targets)
                    print(f"Val loss: {loss.item():.5f}")

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch {epoch}/{num_epochs} Val Loss: {avg_val_loss:.5f}")


            with open(f"Training_GCN_v7/training_log.csv", "a") as file:
                file.write(f"{epoch},{avg_train_loss},{avg_val_loss},\n")
                file.flush()

            # Save best model
            if avg_val_loss < best_val_loss:
                torch.save(model.state_dict(), "Training_GCN_v7/best_model.pth")
                best_val_loss = avg_val_loss

            # Checkpoint every 5 epochs
            if (epoch) % 2 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, f"Training_GCN_v7/checkpoint_{epoch}.pth")
                print(f"Checkpointed at Epoch {epoch}")

    elif Test_flag==1:
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
    else:
        print("Please specify train or test.")

if __name__ == "__main__":
    main(Train_flag=1, Test_flag=0, checkpoint_no=8)
    
    
