import matpotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

num_eval = 0 # number of csv files

for i in range(num_eval):
    preds_i = np.loadtxt(f"Eval_DATA/PTHA/eval-{i}-pred.csv", unpack = True, delimiter = ",", skiprows=1)

    preds = np.append(preds, preds_i)

# Height predictions
height_pred = preds[1::3]

# Binary predictions
in_pred = preds[2::3]
prob = 1/(1+np.exp(-in_pred))
runup_pred = (prob > 0.5).astype(int)

# Create mask
h_in_true = runup_pred.astype(bool)

# Load occurrence rates
xxx, m_rate = np.loadtxt("occurrence-rates.txt",unpack=True)

# Probability of a single event
prob_ev = m_rate/302    # Assuming there's 302 events of each magnitude

# Hazard levels
levels = np.logspace(-2,1,100)

# All magnitudes
ptha_mags = np.array([])
start = 7.5
for i in range(16):
    ptha_mags=np.append(ptha_mags,start*np.ones(302))
    start = start + 0.1

def plot_map(loc, ax=None):
    if ax is None:
        ax = plt.gca()
    lon, lat = np.loadtxt("jp-500-grid.txt",unpack = True)
    clat, clon = np.loadtxt("hr-coast-japan.txt",unpack=True)

    ax.plot(clon, clat,"-k")
    ax.plot(lon[loc],lat[loc],"ro")
    ax.set_xticks([])
    ax.set_yticks([])


def calc_hazard(loc):
    exceedance = np.zeros(100)

    # Calculate exceedance at each level
    for lvl in range(len(levels)):
        L_exceed = 0


        for i in range(4864):
            magnitude = ptha_mags[i]

            if (height_pred[loc::520][i] > levels[lvl]) & (h_in_true[loc::520][i]==True):
                L_exceed += prob_ev[int((magnitude*10))%75]


        exceedance[lvl]= L_exceed

    return exceedance

def plot_hazard(exceedance, title):

    plt.plot(levels, exceedance, "-b")
    plt.semilogx()
    plt.xticks([0.01,0.1,0.5,1,5,10],[0.01,0.1,0.5,1,5,10])
    plt.grid()
    plt.xlabel("Maximum Wave Height [m]")
    plt.ylabel("Exceedance Rate")
    plt.title(f"Hazard Curve: {title}")

def plot_hazard_map(loc, title):
    fig, ax_main = plt.subplots(1, 1, figsize=(4, 3*0.75))

    plt.sca(ax_main)
    exceedance = calc_hazard(loc)
    plot_hazard(exceedance, title)

    # Inset map: lower-left corner, slightly above x-axis and right of y-axis
    ax_inset = ax_main.inset_axes([0.04, 0.04, 0.32, 0.6])
    plot_map(loc, ax=ax_inset)
    ax_inset.set_facecolor((1, 1, 1, 0.5))

    fig.savefig(f"hazard-map-{loc}.png", dpi=400)

    return exceedance

def main(loc_idx, aep_flag=0):
    exceed_loc = plot_hazard_map(293, "Tokyo")
    print("Max exceedance probability", exceed_loc.max())

    if aep_flag==1:
        aep = np.zeros(520)

        for i in range(520):
            ex_i = calc_hazard(i)

            aep[i] = ex_i[np.argmin(np.abs(levels-0.1))]
        np.savetxt("aep.txt", aep)