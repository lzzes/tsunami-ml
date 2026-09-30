import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
Normal = stats.norm

def load_data(num_eval):

    preds = np.array([])

    for i in range(num_eval):
        preds_i = np.loadtxt(f"Eval_Data/PTHA0904/eval-{i}-pred.csv", unpack = True, delimiter = ",", skiprows=1)

        preds = np.append(preds, preds_i)

    # Height predictions
    height_pred = preds[1::3]

    # Inverse log1p transform
    height_pred = np.exp(height_pred) - 1

    # Binary predictions
    in_pred = preds[2::3]
    prob = 1/(1+np.exp(-in_pred))
    runup_pred = (prob > 0.5).astype(int)

    # Create mask
    h_in_true = runup_pred.astype(bool)

    return height_pred, h_in_true

def determine_magnitudes(bin_size):
    # Load occurrence rates
    xxx, m_rate = np.loadtxt("occurrence-rates.txt",unpack=True)

    # Probability of a single event
    prob_ev = m_rate/bin_size    # Assuming there's 302 events of each magnitude

    # Load error


    # All magnitudes
    ptha_mags = np.array([])
    start = 7.5
    for i in range(16):
        ptha_mags=np.append(ptha_mags,start*np.ones(bin_size//2))
        start = start + 0.1

    ptha_mags = np.append(ptha_mags, ptha_mags)

    return prob_ev, ptha_mags


def plot_map(loc, ax=None):
    if ax is None:
        ax = plt.gca()
    lon, lat = np.loadtxt("jp-500-grid.txt",unpack = True)
    clat, clon = np.loadtxt("hr-coast-japan.txt",unpack=True)

    ax.plot(clon, clat,"-k")
    ax.plot(lon[loc],lat[loc],"ro")
    ax.set_xticks([])
    ax.set_yticks([])


def calc_hazard(loc,height_pred,h_in_true,levels,num_eval,ptha_mags,prob_ev):
    exceedance = np.zeros(100)

    # Calculate exceedance at each level
    for lvl in range(len(levels)):
        L_exceed = 0


        for i in range(num_eval*64):
            magnitude = ptha_mags[i]

            if (height_pred[loc::520][i] > levels[lvl]) & (h_in_true[loc::520][i]==True):
                L_exceed += prob_ev[int((magnitude*10))%75]


        exceedance[lvl]= L_exceed

    return exceedance

def calc_hazard_err(loc,height_pred,h_in_true,levels,num_eval,ptha_mags,prob_ev,lower,upper,mean,stdev):

    exceedance = np.zeros(100)

    # Calculate exceedance at each level
    for lvl in range(len(levels)):
        L_exceed = 0 # start with exceedance rate of 0
        threshold = levels[lvl] # establish wave height threshold


        for i in range(num_eval): # for each scenario

            magnitude = ptha_mags[i] # determine the magnitude
            occurrence = prob_ev[int((magnitude*10))%75] # and corresponding occurrence rate

            if h_in_true[loc::520][i]==False: # ignore events without run-up
                continue

            prediction = height_pred[loc::520][i] # find prediction

            # Build CDF
            mask = (prediction < upper) & (prediction < lower)
            center, spread = mean[mask], stdev[mask]

            dist = 1 - Normal.cdf(threshold, loc=center, scale=spread)
            
            L_exceed += dist*occurrence

        exceedance[lvl]= L_exceed

    return exceedance

def plot_hazard(exceedance, title,levels):

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

def main(bin_size=304,num_eval=76,resolution=100,CDF_flag=0):

    # Load data
    height_pred, h_in_true = load_data(num_eval)

    # Calculate normalized event occurrence rates
    prob_ev, ptha_mags = determine_magnitudes(bin_size)

    # Hazard levels
    levels = np.logspace(-2,1,resolution)

    # Set up file
    aep = np.zeros((520, resolution))

    # Discrete or PDF version
    if CDF_flag==1:
        lower, upper, mean, stdev = np.loadtxt("error_rates.txt",unpack=True)

        for k in range(520):
            aep[k,:] = calc_hazard_err(k,height_pred,h_in_true,levels,num_eval,ptha_mags,lower,upper,mean,stdev)
    else:
        for k in range(520):
            aep[k,:] = calc_hazard(k, height_pred, h_in_true, levels, num_eval, ptha_mags, prob_ev)
            if k%50==0:
                print(k)

    np.savetxt("aep.txt", aep)

if __name__=="__main__":
    main()
