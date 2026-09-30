#import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.model_selection import train_test_split

#import tempfile
import torch

from pathlib import Path

from model_gcn import RuptureNet2D

from datasets_gcn import BlockDataset, load_data, make_blocks

from loss_gcn import loss_calc

import numpy as np
import os
import time
import pandas as pd

n = os.environ.get("NSLOTS") or os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count()
n = int(n)
torch.set_num_threads(n)
print(f"torch.set_num_threads({n})")

def main(Train_flag, Test_flag, out_dir, checkpoint_no):

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
    max_epochs = 50
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

    # Model
    model = RuptureNet2D(config)
    device = torch.device("cpu")

    # Configure output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load edges
    edge_index = np.loadtxt("fault_edges.txt", dtype=int)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()


    if Train_flag==1:

        # Data
        train_files = sorted([f for f in data_cache_path.glob("train_*.pt")], key=lambda f: int(f.stem.split("_")[1]))
        train_blocks = [torch.load(f, map_location='cpu', weights_only=True) for f in train_files]
        train_dataset = BlockDataset(train_blocks, debug=debug)

        # Load model
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        best_val_loss = float('inf')

        # Load from Checkpoint
        if checkpoint_no != 0:
            checkpoint_path = torch.load(f"{out_dir}/checkpoint_{checkpoint_no}.pth", map_location="cpu")
            model.load_state_dict(checkpoint_path['model_state_dict'])
            optimizer.load_state_dict(checkpoint_path['optimizer_state_dict'])
            best_val_loss = checkpoint_path.get('best_val_loss', float('inf'))

        model = model.to(device)

        # Prepare data
        train_idx, val_idx = train_test_split(range(len(train_dataset)), train_size=0.75, random_state=1107)
        train_split = SubsetRandomSampler(train_idx)
        val_split = SubsetRandomSampler(val_idx)

        batch_size = config['batch_size']

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, sampler=train_split)        # reduced from 8 to 2
        val_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, sampler=val_split)            # reduced from 8 to 2


        # Training loop
        iteration = 0
        epoch = 0

        while epoch < max_epochs:

            epoch = iteration + 1 + checkpoint_no

            model.train()
            total_loss = 0.0
            batch_no = 1

            t_loss = 0.0
            h_loss = 0.0
            BCE_loss = 0.0

            for batch in train_dataloader:
                
                # TRAINING

                inputs, targets = batch
                inputs, targets = (
                    inputs.to(device),
                    targets.to(device)
                )

                optimizer.zero_grad()
                outputs = model(inputs, edge_index)

                loss, t_MSE, h_MSE, BCE = loss_calc(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Save the loss
                total_loss += loss.item()

                # Individual losses
                t_loss += t_MSE.item()
                h_loss += h_MSE.item()
                BCE_loss += BCE.item()

                batch_no += 1
                del inputs, targets, outputs, loss

            avg_train_loss = total_loss / len(train_dataloader)
            avg_t = t_loss / len(train_dataloader)
            avg_h = h_loss / len(train_dataloader)
            avg_BCE = BCE_loss / len(train_dataloader)
            print(f"Epoch {epoch}/{max_epochs} Train Loss: {avg_train_loss:.5f}")

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
            print(f"Epoch {epoch}/{max_epochs} Val Loss: {avg_val_loss:.5f}")


            with open(f"{out_dir}/training_log.csv", "a") as file:
                file.write(f"{epoch},{avg_train_loss},{avg_val_loss},{avg_t},{avg_h},{avg_BCE}\n")
                file.flush()

            # Save best model
            if avg_val_loss < best_val_loss:
                torch.save(model.state_dict(), f"{out_dir}/best_model.pth")
                best_val_loss = avg_val_loss

            # Checkpoint every 5 epochs
            if (epoch) % 2 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                }, f"{out_dir}/checkpoint_{epoch}.pth")
                print(f"Checkpointed at Epoch {epoch}")

            iteration += 1

    elif Test_flag==1:

        model.load_state_dict(torch.load(f"{out_dir}/best_model.pth"))
        model = model.to(device)
        model.eval()

        print("Model loaded in eval mode")

        test_files = sorted([f for f in data_cache_path.glob("test_*.pt")], key=lambda f: int(f.stem.split("_")[1]))
        test_blocks = [torch.load(f, map_location='cpu', weights_only=True) for f in test_files]
        test_dataset = BlockDataset(test_blocks, debug=debug)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

        test_idx = np.loadtxt("test_idx.txt", dtype = int)

        count = 0

        # Run through model
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                preds = model(inputs,edge_index)
                
                preds = preds.detach().cpu().numpy().flatten()
                targets = targets.detach().cpu().numpy().flatten()

                pdf = pd.DataFrame({"preds": preds})
                pdf.to_csv(f"{out_dir}/eval-{test_idx[idx]}-pred.csv", index = False)

                tdf = pd.DataFrame({"targets": targets})
                tdf.to_csv(f"{out_dir}/eval-{test_idx[idx]}-true.csv", index = False)
                print(count)
                count += 1

        inputs_single = inputs[0:1]  # reuse last batch, take one sample
        start = time.perf_counter()
        pred_single = model(inputs_single, edge_index)
        end = time.perf_counter()
        print(f"Single sample inference: {(end - start) * 1000:.3f} ms")

    else:
        print("Please specify train or test.")

if __name__ == "__main__":
    main(Train_flag=0, Test_flag=1, out_dir="Training0904", checkpoint_no=0)
    
    
