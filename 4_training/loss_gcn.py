import torch
import numpy as np
import torch.nn as nn
import torch

# 6/26/26 Update
# MSE = t_mse + 2*h_mse
# weighted for better fitting

# 7/10/26 Update
# MSE = t_mse + 1000*h_mse
# to make sure scale matches

# 8/3/26 Update
# MSE = t_mse + 2*h_mse
# z scale for h again

# 8/13/26 Update
# MSE = t_mse + h_mse
# weight = (1 + target)**2 instead of just (1+ target)
# will unstandardize h completely

# Masked loss function
def masked_MSE(pred, target, inundate, nan_value=9999999):
    # Separate time target for NAN mask
    mask = inundate.bool()

    # LV = mask.unsqueeze(-1).expand_as(target)  # (B,100,2)
    # LV = LV.to(pred.device)

    LV = mask

    # Count total valid
    valid_count = LV.sum()
    if valid_count == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # Calculate MSE then mask
    sq_err = ((pred - target)**2)

    masked_sq_err = sq_err[LV]

    MSE = masked_sq_err.sum()/valid_count

    return MSE

def weighted_MSE(pred, target, inundate):
    # Separate time target for NAN mask
    mask = inundate.bool()

    # LV = mask.unsqueeze(-1).expand_as(target)  # (B,100,2)
    # LV = LV.to(pred.device)

    LV = mask

    # Count total valid
    valid_count = LV.sum()
    if valid_count == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # Calculate MSE then mask
    k = 2 # quadratic scaling for now
    weight = (1 + target)**k
    sq_err = weight*((pred - target)**2)

    masked_sq_err = sq_err[LV]

    MSE = masked_sq_err.sum()/valid_count

    return MSE

def loss_calc(preds_all,targets_all):
    # Separate real values
    t_preds = preds_all[:,:,0]
    t_targets = targets_all[:,:,0]
    h_preds = preds_all[:,:,1]
    h_targets = targets_all[:,:,1]
    inundate_preds = preds_all[:,:,2]
    t_inundate_targets = targets_all[:,:,2]
    h_inundate_targets = targets_all[:,:,3]

    # Calculate MSE
    t_MSE = masked_MSE(t_preds,t_targets,t_inundate_targets)
    h_MSE = weighted_MSE(h_preds,h_targets,h_inundate_targets)

    # Calculate BCE
    bce_func = nn.BCEWithLogitsLoss()
    BCE = bce_func(inundate_preds, h_inundate_targets)

    MSE = t_MSE + h_MSE

        # Calculate loss
    loss = MSE + BCE
    print(f"MSE: {MSE.item():.5f}; BCE: {BCE.item():.5f}")
    print(f"\t t_MSE: {t_MSE:.5f}, h_MSE: {h_MSE:.8f}")

    return loss
