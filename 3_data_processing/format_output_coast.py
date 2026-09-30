#####
# Created by L. Su 8/6/25
#####

# This code will read fgmax results
# and output a csv file for the 
# relevant points along Japan's coastline


#####
# IMPORTS
#####
import numpy as np

#####
# PRESETS
#####

# Load coastline data
lon_coast, lat_coast = np.loadtxt("jp-500-grid.txt", unpack = True)

# Data size
datasize=2500

# File directory
directory = "/mnt/c/Users/lizzi/OneDrive/Desktop/Meng_Group/Tsunami_ML/2026/tsunami-ml/2_synthetic_data/fgmax"

#####
# UDFs
#####

def load_fgmax(file):
    # Load fgmax data
    data = np.loadtxt(file) 

    # Extract values from array
    lon = data[:,0]
    lat = data[:,1]
    bathy = data[:,3]
    depth = data[:,4]
    time = data[:,-1]
    
    # Round values for matching
    fglon = np.round(lon, 7)
    fglat = np.round(lat, 7)

    return fglon, fglat, bathy, depth, time

def match_coast(fglon, fglat):
    '''
    returns indices that would match the coastline
    '''

    # Round values for matching
    latgrid = np.round(lat_coast, 7)
    longrid = np.round(lon_coast, 7)

    idx = []

    # Find coast coordinates 
    for i in range(len(longrid)):

        # Mask fgmax data
        mask = (fglon == longrid[i]) & (fglat == latgrid[i])
    
        # Check if we found a match
        matches = np.where(mask)[0]
    
        # Take first match
        if len(matches) > 0:
            idx += [matches[0]]
        else:
            print(f"Warning: No match found for point {i}: lon={longrid[i]}, lat={latgrid[i]}")

    idx = np.array(idx, dtype = int)

    return idx

def write_file(fname, no):
    lonj, latj, bathyj, depthj, timej = load_fgmax(fname)

    output = np.zeros((520,6))

    # longitude and latitude
    output[:,0] = lonj[match_indices]
    output[:,1] = latj[match_indices]

    # arrival time
    output[:,2] = timej[match_indices]

    # max height
    height = np.where(bathyj>0,depthj, bathyj + depthj)

    # remove negative fgmax values
    valid_mask = (height < 0) & (height > -100)
    height[valid_mask] = np.zeros_like(height[valid_mask])
    output[:,3] = height[match_indices]
    

    # inundation status
    for k in range(520):
        if (height[match_indices][k] < -999):
            output[k,5] = 0 # not inundated
        elif (height[match_indices][k] > -999):
            output[k,5] = 1 # inundated
        else:
            print(f"Height inundation status for row {k} in file {no} could not be determined")

        if (timej[match_indices][k] < -999):
            output[k,4] = 0
        elif (timej[match_indices][k] > -999):
            output[k,4] = 1
        else:
            print(f"Time inundation status for row {k} in file {no} could not be determined")


    # Save File
    np.savetxt(f"Output0904/output-{no}.csv", output, delimiter = ",", header = "Lon,Lat,Arrival,MaxHeight,InundateT,InundateH", comments = "")

# Find initial indices
fglon0, fglat0, bathy0, depth0, time0 = load_fgmax(f"{directory}/fgmax-j0.txt")

match_indices = match_coast(fglon0, fglat0)

for j in range(datasize):
    fname_j = f"{directory}/fgmax-j{j}.txt"
    run_num = j
    write_file(fname_j, run_num)

    fname_n = f"{directory}/fgmax-n{j}.txt"
    run_num = j + datasize
    write_file(fname_n, run_num)

    if j%500==0:
        print(f"{j} done.")