'''
Processes Fakequakes data for use in neural network

Input: fakequakes .rupt file
Output: GCN input-*.csv file
'''

import numpy as np

def save_input(dir, fault, num):

    # File prefix
    if num < 10:
        prefix = "00000"
    elif num < 100:
        prefix = "0000"
    elif num < 1000:
        prefix = "000"
    elif num < 2432:
        prefix = "00"
    else:
        prefix = ""


    # Load file
    rupture_data = np.loadtxt(f"{dir}/{fault}.{prefix}{num}.rupt") ## Update with address of ruptures folder

    # Calculate total slip
    ss_slip = rupture_data[:,8]
    ds_slip = rupture_data[:,9]

    slip = np.sqrt((ss_slip**2 + ds_slip**2))

    # Load template 
    Longitude, Latitude, Depth, Slip_all, Inundate = np.loadtxt("input-template.txt", unpack = True)

    # Input slip and assign input number
    if fault == 'japan':
        Slip_all[:987] = slip
    elif fault == 'nankai':
        Slip_all[987:] = slip
        num = int(num) + dataset_size
    else:
        print("Provide valid fault name")
        exit()

    # Save file
    input_file = np.column_stack((Longitude, Latitude, Depth, Slip_all))
    np.savetxt(f"Input0904/input-{num}.csv", input_file, delimiter =',', header = "Lon,Lat,Depth,Slip", comments="")

    # Print updates
    if int(num)%100==0:
        print(f"Input {num} on fault {fault} saved.")

# Create all input files
dataset_size = 2500 # update according to the number of ruptures on each fault
fname_1 = "japan"
fname_2 = "nankai"
directory = "/mnt/c/Users/lizzi/OneDrive/Desktop/Meng_Group/Tsunami_ML/2026/tsunami-ml/2_synthetic_data/ruptures"

for i in range(2432,dataset_size):
    save_input(directory, fname_1,i)
    save_input(directory, fname_2,i)