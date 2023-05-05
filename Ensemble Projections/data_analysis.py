############################################################################################################################
# Multi-model Analysis of Thermodynamic Sea Ice Models
# Author : Amaury Laridon & Augustin Lambotte
# Course : LPHYS2265 - Sea ice ocean interactions in polar regions
# Goal : Analyze the ensemble of 15 models with Control run (CTL) and 14 with projections run (PR) in order to answer
#        to two major scientific questions.
#        More information on the GitHub Page of the project : https://github.com/AmauryLaridon/TSIM.git
# Date : 03/05/23
############################################################################################################################
#################################################### Packages ##############################################################
import numpy as np
import numpy.ma as ma
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.gridspec as gridspec

################################################### Parameters #############################################################

################################ Script Parameters #######################################
N_mod_CTL = 15  # number of models at disposal for the CTL run [Adim]
N_mod_PR = 14  # number of models at disposal for the PR run [Adim]
N_years_CTL = 1  # number of years in the CTL simulation [Adim]
N_days_CTL = 365 * N_years_CTL  # number of days in the CTL simulation [Adim]
N_years_PR = 100  # number of years in the PR simulation [Adim]
N_days_PR = 365 * N_years_PR  # number of days in the PR simulation [Adim]
Day_0 = 0
################################ Display Parameters #######################################
plt.rcParams["text.usetex"] = True
save_dir = "/home/amaury/Bureau/LPHYS2265 - Sea ice ocean atmosphere interactions in polar regions/Projet - Multimodel Analysis/Figures/"
figure = plt.figure(figsize=(16, 10))

################################################### Read Data ##############################################################

data_dir = "/home/amaury/Bureau/LPHYS2265 - Sea ice ocean atmosphere interactions in polar regions/Projet - Multimodel Analysis/Ensemble Projections/"
##### Read CTL #####
CTL = loadmat(data_dir + "CTL.mat")
hi = CTL["hi"]
hs = CTL["hs"]
Tsu = CTL["Tsu"]
Tw = CTL["Tw"]
doy = CTL["doy"]
model = CTL["model"]
# one-liner to read a single variable

### Setting NaN value to the h_s variable for the models that don't have snow implemented ###
model_name_without_snow = [
    "ANTIM",
    "BRAILLE_ANE",
    "COLDGATE",
    "DOUGLACE",
    "ICENBERG",
    "MisterFreeze",
]
model_index_without_snow = [0, 1, 2, 3, 6, 8]
for i in model_index_without_snow:
    hs[:, i] = np.NaN


##### Read PR #####

PR = loadmat(data_dir + "PR.mat")
Nmod = PR["Nmod"]
model = PR["model"]
year = PR["year"]
himax = PR["himax"]
himin = PR["himin"]
himean = PR["himean"]
hsmax = PR["hsmax"]
Tsumin = PR["Tsumin"]

############################################################################################################################
###################################################### Data Analysis #######################################################
############################################################################################################################

################################################### Control Simulation #####################################################

# Target seasonal cycle of ice thickness of MU71
hi_MU71 = [
    2.82,
    2.89,
    2.97,
    3.04,
    3.10,
    3.14,
    2.96,
    2.78,
    2.73,
    2.71,
    2.72,
    2.75,
]

time_range_ctl = np.arange(Day_0, N_days_CTL, 1)
time_range_ctl_MU_71 = np.linspace(Day_0, N_days_CTL, 12)
##### Mean computation #####


def mean_all_mod(data, N_day, N_mod):
    """Computes the mean of all models of a given variable for each day and returns it as a list"""
    mod_mean = np.zeros(N_day)
    for day in range(N_day):
        som = 0
        for model in range(N_mod):
            # if data[0, model] == "NaN":
            # new_data = [x for x in data[:, model] if np.isnan(x) == False]
            # print(np.shape(new_data))
            som = som + data[day, model]
        day_mean = som / N_mod
        mod_mean[day] = day_mean

    return mod_mean


##### Display #####


def plot_all_mod(data, data_name, N_mod):
    for model in range(N_mod):
        plt.plot(time_range_ctl, data[:, model])
    if data_name == "hi":
        plt.plot(
            time_range_ctl_MU_71,
            hi_MU71,
            label=r"$h_{i_{MU71}}$",
            linewidth=4,
            color="tab:green",
        )
    mod_mean = mean_all_mod(data=data, N_day=N_days_CTL, N_mod=N_mod)
    # print(mod_mean)
    plt.plot(
        time_range_ctl, mod_mean, linewidth=4, color="tab:blue", label=r"Models Mean"
    )
    plt.title(
        r"Comparison between ensemble members ($N_{mod} = 15$), their averages and the observation series (MU71)",
        size=24,
    )
    plt.xlabel("Days", size=20)
    if data_name == "hi":
        unit = "[m]"
    if data_name == "Tsu":
        unit = "[°K]"
    if data_name == "hs":
        unit = "[m]"
    plt.ylabel(data_name + unit, size=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.legend(fontsize=20)
    plt.savefig(save_dir + data_name + "_all_mod.png", dpi=300)
    # plt.show()
    plt.clf()


def subplot_all_mod(data1, data2, data3, data_name, N_mod):
    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.3, left=0.1, right=0.98)

    plt.figure()
    plt.suptitle(r"Ensemble model output $(h_i, T_{su}, h_s)$ for the CTL simulation")
    ### Figure 1 ###
    ax = plt.subplot(gs[0, 0])  # row 0, col 0
    for model in range(N_mod):
        plt.plot(time_range_ctl, data1[:, model], linewidth=0.5)
    plt.plot(
        time_range_ctl_MU_71,
        hi_MU71,
        label=r"$h_{i_{MU71}}$",
        linewidth=2,
        color="tab:green",
    )
    mod_mean = mean_all_mod(data=data1, N_day=N_days_CTL, N_mod=N_mod)
    plt.plot(
        time_range_ctl, mod_mean, linewidth=2, color="tab:blue", label=r"Models Mean"
    )
    plt.xlabel("Days", size=10)
    plt.ylabel("Ice Thickness [m]", size=10)
    plt.xticks(np.arange(0, N_days_CTL, 50), fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.legend(fontsize=10)

    ### Figure 2 ###
    ax = plt.subplot(gs[0, 1])  # row 0, col 1
    for model in range(N_mod):
        plt.plot(time_range_ctl, data2[:, model], linewidth=0.5)
    mod_mean = mean_all_mod(data=data2, N_day=N_days_CTL, N_mod=N_mod)
    plt.plot(
        time_range_ctl, mod_mean, linewidth=2, color="tab:blue", label=r"Models Mean"
    )
    plt.xlabel("Days", size=10)
    plt.ylabel("Surface Temperature [°K]", size=10)
    plt.xticks(np.arange(0, N_days_CTL, 50), fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.legend(fontsize=10)

    ### Figure 3 ###
    ax = plt.subplot(gs[1, :])  # row 1, span all columns
    for model in range(N_mod):
        plt.plot(time_range_ctl, data3[:, model], linewidth=0.5)
    mod_mean = mean_all_mod(data=data3, N_day=N_days_CTL, N_mod=N_mod)
    plt.plot(
        time_range_ctl, mod_mean, linewidth=2, color="tab:blue", label=r"Models Mean"
    )
    plt.xlabel("Days", size=10)
    plt.ylabel("Snow Thickness [m]", size=10)
    plt.xticks(np.arange(0, N_days_CTL, 25), fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.legend(fontsize=10)

    plt.savefig(save_dir + "subplot_all_mod.png", dpi=300)
    plt.clf()


if __name__ == "__main__":
    plot_all_mod(data=hi, data_name="hi", N_mod=N_mod_CTL)
    plot_all_mod(data=Tsu, data_name="Tsu", N_mod=N_mod_CTL)
    plot_all_mod(data=hs, data_name="hs", N_mod=N_mod_CTL)
    subplot_all_mod(
        data1=hi, data2=Tsu, data3=hs, data_name=["hi,Tsu,hs"], N_mod=N_mod_CTL
    )
