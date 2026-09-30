from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_gcn import RuptureNet2D
from eval_data import load_eval_data, make_eval_blocks, EvalBlockDataset


def main(eval_data_dir="Eval_Data",
         input_stats_path="input_stats.txt",
         model_checkpoint="Training_GCN_v7/best_model.pth",
         output_dir="Eval_Data/PTHA"):

    config = {
        'batch_size': 64,
        'lr': 1e-4,
        'gcn_in_features': 4,
        'gcn_hidden_dim': 128,
        'gcn_out_dim': 256,
        'conv1_dim': 256,
        'conv1_layers': 2,
        'num_transformer_blocks': 6,
        'num_transformer_heads': 8,
        'transformer_hidden_dim': 768,
        'conv2_dim': 256,
        'dropout_pro': 0.4,
        }

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the TRAINING set's input normalization stats (see compute_input_stats.py).
    # Eval data must be normalized with these, not its own mean/std.
    input_stats = np.loadtxt(input_stats_path)

    # Load and block the eval data
    df_x = load_eval_data(eval_data_dir)
    eval_blocks = make_eval_blocks(df_x)
    eval_dataset = EvalBlockDataset(eval_blocks)

    # Model
    device = torch.device("cpu")
    model = RuptureNet2D(config)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded in eval mode")

    # Load fault graph edges
    edge_index = np.loadtxt("fault_edges.txt", dtype=int)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    eval_loader = DataLoader(eval_dataset, batch_size=config['batch_size'],
                              shuffle=False, num_workers=0)

    batch_times = []
    batch_sizes = []
    run_start = time.perf_counter()

    count = 0
    with torch.no_grad():
        for idx, inputs in enumerate(eval_loader):
            inputs = inputs.to(device)

            batch_start = time.perf_counter()

            preds = model(inputs, edge_index)

            batch_end = time.perf_counter()
            batch_times.append(batch_end - batch_start)
            batch_sizes.append(inputs.shape[0])

            preds = preds.detach().cpu().numpy().flatten()

            pdf = pd.DataFrame({"preds": preds})
            pdf.to_csv(f"{output_dir}/eval-{idx}-pred.csv", index=False)
            print(count)
            count += 1

    run_end = time.perf_counter()

    batch_times = np.array(batch_times)
    batch_sizes = np.array(batch_sizes)
    np.savetxt("batch_info.txt", np.column_stack((batch_times, batch_sizes)))


    total_inference_time = batch_times.sum()

    print(f"\nBatches evaluated: {len(batch_times)}")
    print(f"Total samples: {4864}")
    print(f"Mean batch inference time: {batch_times.mean() * 1000:.3f} ms "
          f"(std: {batch_times.std() * 1000:.3f} ms)")
    print(f"Mean per-sample inference time: {(total_inference_time / 4864) * 1000:.3f} ms")
    print(f"Total inference time (sum over batches, excludes I/O): {total_inference_time:.3f} s")
    print(f"Total wall-clock time (inference + CSV writes): {run_end - run_start:.3f} s")

if __name__ == "__main__":
    main()
