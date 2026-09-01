'''
Processes Fakequakes data for use in neural network

Input: fakequakes .rupt file
Output: GCN input-*.csv file
'''

import numpy as np

def save_input(fault, num):
    # Load file
    rupture_data = np.loadtxt(f"{fault}.{num}.rupt") ## Update with address of ruptures folder

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
    np.savetxt(f"input-{num}.csv", input_file, delimiter =',', header = "Lon,Lat,Depth,Slip", comments="")

    # Print updates
    if int(num)%100==0:
        print(f"Input {num} on fault {fault} saved.")

# Create all input files
dataset_size = 2500 # update according to the number of ruptures on each fault

fname_1 = "japan"
fname_2 = "nankai"
for i in range(2500):
    save_input(fname_1,i)
    save_input(fname_2,i)