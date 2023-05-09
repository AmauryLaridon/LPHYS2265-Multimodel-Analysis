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
from sklearn.metrics import mean_squared_error

#plt.rcParams["text.usetex"] = True
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
################################################### Parameters #############################################################

################################ Script Parameters #######################################
N_mod_CTL = 15  # number of models at disposal for the CTL run [Adim]
N_mod_PR = 14  # number of models at disposal for the PR run [Adim]
N_years_CTL = 1  # number of years in the CTL simulation [Adim]
N_days_CTL = 365 * N_years_CTL  # number of days in the CTL simulation [Adim]
N_month_CTL = 12 * N_years_CTL  # number of month in the CTL simulation [Adim]
N_years_PR = 100  # number of years in the PR simulation [Adim]
N_days_PR = 365 * N_years_PR  # number of days in the PR simulation [Adim]
Day_0 = 0
Month_0 = 0
################################ Display Parameters #######################################


save_dir = "./Figures/"
figure = plt.figure(figsize=(16, 10))

################################################### Read Data ##############################################################

#data_dir = "/home/amaury/Bureau/LPHYS2265 - Sea ice ocean atmosphere interactions in polar regions/Projet - Multimodel Analysis/Ensemble Projections/"
##### Read CTL #####
CTL = loadmat("./CTL.mat")
hi_ctl = CTL["hi"]
hs_ctl = CTL["hs"]
Tsu_ctl = CTL["Tsu"]
Tw_ctl = CTL["Tw"]
doy_ctl = CTL["doy"]
model_ctl = CTL["model"]
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
    hs_ctl[:, i] = np.NaN


##### Read PR #####

PR = loadmat("./PR.mat")
Nmod_pr = PR["Nmod"]
model_pr = PR["model"]
year_pr = PR["year"]
himax_pr = PR["himax"]
himin_pr = PR["himin"]
himean_pr = PR["himean"]
hsmax_pr = PR["hsmax"]
Tsumin_pr = PR["Tsumin"]

############################################################################################################################
###################################################### Data Analysis #######################################################
############################################################################################################################

##################################################### Tools function #######################################################

##### Mean computation #####


def mean_all_mod(data):
    """Computes the mean of all models of a given variable for each member of the interval N (could be 12 month or 365 days) and returns it as a list"""
    #transform the 0 in Nan
    data = np.where(data==0, np.nan, data)
    return np.mean(ma.masked_invalid(data),axis = 1)

def month_mean(data):
    """Given the array of a daily variable for one year returns the means for every month
    of that value."""

    data_mean_month_ar = np.zeros(12)

    data_sum = 0
    for day in range(0, 31):
        data_sum += data[day]
    data_mean = data_sum / 31
    data_mean_month_ar[0] = data_mean

    data_sum = 0
    for day in range(31, 59):
        data_sum += data[day]
    data_mean = data_sum / 28
    data_mean_month_ar[1] = data_mean

    data_sum = 0
    for day in range(59, 90):
        data_sum += data[day]
    data_mean = data_sum / 31
    data_mean_month_ar[2] = data_mean

    data_sum = 0
    for day in range(90, 120):
        data_sum += data[day]
    data_mean = data_sum / 30
    data_mean_month_ar[3] = data_mean

    data_sum = 0
    for day in range(120, 151):
        data_sum += data[day]
    data_mean = data_sum / 31
    data_mean_month_ar[4] = data_mean

    data_sum = 0
    for day in range(151, 181):
        data_sum += data[day]
    data_mean = data_sum / 30
    data_mean_month_ar[5] = data_mean

    data_sum = 0
    for day in range(181, 212):
        data_sum += data[day]
    data_mean = data_sum / 31
    data_mean_month_ar[6] = data_mean

    data_sum = 0
    for day in range(212, 243):
        data_sum += data[day]
    data_mean = data_sum / 31
    data_mean_month_ar[7] = data_mean

    data_sum = 0
    for day in range(243, 273):
        data_sum += data[day]
    data_mean = data_sum / 30
    data_mean_month_ar[8] = data_mean

    data_sum = 0
    for day in range(273, 304):
        data_sum += data[day]
    data_mean = data_sum / 31
    data_mean_month_ar[9] = data_mean

    data_sum = 0
    for day in range(304, 334):
        data_sum += data[day]
    data_mean = data_sum / 30
    data_mean_month_ar[10] = data_mean

    data_sum = 0
    for day in range(334, 365):
        data_sum += data[day]
    data_mean = data_sum / 31
    data_mean_month_ar[11] = data_mean

    return data_mean_month_ar


##### Error computation #####


def err_annual_mean_thick(data1, data2):
    """Compute the annual mean value of a variable for a given model output and compare it to an other serie. Compute also the absolute error
    and the relative error between the two. The data2 is considered as the reference one for the relative error computation
    """
    data1_annual_mean_thick = sum(data1) / 12
    data2_annual_mean_thick = sum(data2) / 12
    err_abs = data1_annual_mean_thick - data2_annual_mean_thick
    err_rel = (
        (data1_annual_mean_thick - data2_annual_mean_thick) / data2_annual_mean_thick
    ) * 100
    return data1_annual_mean_thick, data2_annual_mean_thick, err_abs, err_rel


def MSE_annual_mean_thick(data1, data2):
    """Compute the Mean Squared Error between two data set. The MSE will be used as a diagnostic tool
    for the efficienty of a model or an ensemble of models."""
    mse = mean_squared_error(ma.masked_invalid(data1), ma.masked_invalid(data2))
    return mse


def cor_annual_mean_thick(data1, data2):
    """Computes the correlation coefficient for a first model output and the second one."""
    corr_matrix = np.corrcoef(data1, data2)
    r = corr_matrix[0, 1]
    return r


def std_var_mean_thick(data):
    """returns the standard deviation of a given data set array"""
    std = np.std(data)
    return std


##### Comparison data series #####


def comp_ENS_MU71():
    ##### Comparison between ENSEMBLE and MU71 #####
    ### Computation of the error on annual mean ice thickness the ENSEMBLE mean with respect to MU71 ###
    mean_ENS_hi = mean_all_mod(data=hi)
    mean_ENS_month_hi = month_mean(
        mean_ENS_hi
    )  # Computation of the month mean of hi ensemble mean #
    mean_ENSEMBLE_hi, mean_MU71_hi, err_abs, err_rel = err_annual_mean_thick(
        mean_ENS_month_hi, hi_MU71
    )
    mse_hi = MSE_annual_mean_thick(data1=mean_ENS_month_hi, data2=hi_MU71)
    std_ENSEMBLE = std_var_mean_thick(data=mean_ENS_month_hi)
    std_MU71 = std_var_mean_thick(data=hi_MU71)
    r = cor_annual_mean_thick(data1=mean_ENS_month_hi, data2=hi_MU71)
    print(
        "------------------------------------------------------------------------------------"
    )
    print(
        "-------------------- ENSEMBLE & MU71 ICE THICKNESS COMPARISON ----------------------"
    )
    print(
        "------------------------------------------------------------------------------------"
    )
    print("Mean ice thickness ENSEMBLE = {:.3f}m".format(mean_ENSEMBLE_hi))
    print("Mean ice thickness MU71 = {:.3f}m".format(mean_MU71_hi))
    print("Absolute Error = {:.4f}m".format(err_abs))
    print("Relative Error = {:.3f}%".format(err_rel))
    print("Standard deviation ENSEMBLE = {:.3f}".format(std_ENSEMBLE))
    print("Standard deviation MU71 = {:.3f}".format(std_MU71))
    print("MSE(ENSEMBLE,MU71) = {:.3f}".format(mse_hi))
    print("r(TSIMAL,MU71) = {:.3f}".format(r))
    print(
        "------------------------------------------------------------------------------------"
    )


def comp_TSIMAL_MU71():
    ##### Comparison between TSIMAL and MU71 #####
    ### Computation of the error on annual mean ice thickness the TSIMAL mean with respect to MU71 ###
    mean_ENS_hi = mean_all_mod(data=hi)
    mean_ENS_month_hi = month_mean(
        mean_ENS_hi
    )  # Computation of the month mean of hi ensemble mean #
    mean_month_TSIMAL_hi = month_mean(
        hi[:, 13]
    )  # Computation of the month mean of hi for TSIMAL #
    mean_TSIMAL_hi, mean_MU71_hi, err_abs, err_rel = err_annual_mean_thick(
        mean_month_TSIMAL_hi, hi_MU71
    )
    mse_hi = MSE_annual_mean_thick(data1=mean_month_TSIMAL_hi, data2=hi_MU71)
    std_TSIMAL = std_var_mean_thick(data=mean_month_TSIMAL_hi)
    std_MU71 = std_var_mean_thick(data=hi_MU71)
    r = cor_annual_mean_thick(data1=mean_month_TSIMAL_hi, data2=hi_MU71)
    print(
        "-------------------- TSIMAL & MU71 ICE THICKNESS COMPARISON ----------------------"
    )
    print(
        "------------------------------------------------------------------------------------"
    )
    print("Mean ice thickness TSIMAL = {:.3f}m".format(mean_TSIMAL_hi))
    print("Mean ice thickness MU71 = {:.3f}m".format(mean_MU71_hi))
    print("Absolute Error = {:.4f}m".format(err_abs))
    print("Relative Error = {:.3f}%".format(err_rel))
    print("Standard deviation TSIMAL = {:.3f}".format(std_TSIMAL))
    print("Standard deviation MU71 = {:.3f}".format(std_MU71))
    print("MSE(TSIMAL,MU71) = {:.3f}".format(mse_hi))
    print("r(TSIMAL,MU71) = {:.3f}".format(r))
    print(
        "------------------------------------------------------------------------------------"
    )


def comp_SIGUS_MU71():
    ##### Comparison between SIGUS and MU71 #####
    ### Computation of the error on annual mean ice thickness the SIGUS mean with respect to MU71 ###
    mean_ENS_hi = mean_all_mod(data=hi)
    mean_ENS_month_hi = month_mean(
        mean_ENS_hi
    )  # Computation of the month mean of hi ensemble mean #
    mean_month_SIGUS_hi = month_mean(
        hi[:, 11]
    )  # Computation of the month mean of hi for SIGUS #
    mean_SIGUS_hi, mean_MU71_hi, err_abs, err_rel = err_annual_mean_thick(
        mean_month_SIGUS_hi, hi_MU71
    )
    mse_hi = MSE_annual_mean_thick(data1=mean_month_SIGUS_hi, data2=hi_MU71)
    std_SIGUS = std_var_mean_thick(data=mean_month_SIGUS_hi)
    std_MU71 = std_var_mean_thick(data=hi_MU71)
    r = cor_annual_mean_thick(data1=mean_month_SIGUS_hi, data2=hi_MU71)
    print(
        "-------------------- SIGUS & MU71 ICE THICKNESS COMPARISON ----------------------"
    )
    print(
        "------------------------------------------------------------------------------------"
    )
    print("Mean ice thickness SIGUS = {:.3f}m".format(mean_SIGUS_hi))
    print("Mean ice thickness MU71 = {:.3f}m".format(mean_MU71_hi))
    print("Absolute Error = {:.4f}m".format(err_abs))
    print("Relative Error = {:.3f}%".format(err_rel))
    print("Standard deviation SIGUS = {:.3f}".format(std_SIGUS))
    print("Standard deviation MU71 = {:.3f}".format(std_MU71))
    print("MSE(SIGUS,MU71) = {:.3f}".format(mse_hi))
    print("r(SIGUS,MU71) = {:.3f}".format(r))
    print(
        "------------------------------------------------------------------------------------"
    )


def comp_TSIMAL_ENS():
    ##### Comparison between ENSEMBLE and TSIMAL #####
    ### Computation of the error on annual mean ice thickness of TSIMAL with respect to the ENSEMBLE ###
    mean_ENS_hi = mean_all_mod(data=hi)
    mean_ENS_month_hi = month_mean(
        mean_ENS_hi
    )  # Computation of the month mean of hi ensemble mean #
    mean_month_TSIMAL_hi = month_mean(
        hi[:, 13]
    )  # Computation of the month mean of hi for TSIMAL #
    mean_TSIMAL_hi, mean_ENS_hi, err_abs, err_rel = err_annual_mean_thick(
        mean_month_TSIMAL_hi, mean_ENS_month_hi
    )
    mse_hi = MSE_annual_mean_thick(data1=mean_month_TSIMAL_hi, data2=mean_ENS_month_hi)
    std_ENS = std_var_mean_thick(data=mean_ENS_month_hi)
    std_TSIMAL = std_var_mean_thick(data=mean_month_TSIMAL_hi)
    r = cor_annual_mean_thick(data1=mean_month_TSIMAL_hi, data2=mean_ENS_month_hi)
    print(
        "-------------------- TSIMAL & ENSEMBLE ICE THICKNESS COMPARISON ----------------------"
    )
    print(
        "------------------------------------------------------------------------------------"
    )
    print("Mean ice thickness TSIMAL = {:.3f}m".format(mean_TSIMAL_hi))
    print("Mean ice thickness ENSEMBLE = {:.3f}m".format(mean_ENS_hi))
    print("Absolute Error = {:.4f}m".format(err_abs))
    print("Relative Error = {:.3f}%".format(err_rel))
    print("Standard deviation TSIMAL = {:.3f}".format(std_TSIMAL))
    print("Standard deviation ENSEMBLE = {:.3f}".format(std_ENS))
    print("MSE(TSIMAL,ENSEMBLE) = {:.3f}".format(mse_hi))
    print("r(TSIMAL,ENSEMBLE) = {:.3f}".format(r))
    print(
        "------------------------------------------------------------------------------------"
    )
    ### Computation of the error on annual mean snow thickness of TSIMAL with respect to the ENSEMBLE ###
    mean_ENS_hs = mean_all_mod(data=hs)
    mean_ENS_month_hs = month_mean(
        mean_ENS_hs
    )  # Computation of the month mean of hi ensemble mean #
    mean_month_TSIMAL_hs = month_mean(
        hs[:, 13]
    )  # Computation of the month mean of hs for TSIMAL #
    mean_TSIMAL_hs, mean_ENS_hs, err_abs, err_rel = err_annual_mean_thick(
        mean_month_TSIMAL_hs, mean_ENS_month_hs
    )
    # mse_hs = MSE_annual_mean_thick(data1=mean_month_TSIMAL_hs, data2=mean_ENS_month_hs)
    std_ENS = std_var_mean_thick(data=mean_ENS_month_hs)
    std_TSIMAL = std_var_mean_thick(data=mean_month_TSIMAL_hs)
    r = cor_annual_mean_thick(data1=mean_month_TSIMAL_hs, data2=mean_ENS_month_hs)
    print(
        "-------------------- TSIMAL & ENSEMBLE SNOW THICKNESS COMPARISON ----------------------"
    )
    print(
        "------------------------------------------------------------------------------------"
    )
    print("Mean snow thickness TSIMAL = {:.3f}m".format(mean_TSIMAL_hs))
    print("Mean snow thickness ENSEMBLE = {:.3f}m".format(mean_ENS_hs))
    print("Absolute Error = {:.4f}m".format(err_abs))
    print("Relative Error = {:.3f}%".format(err_rel))
    print("Standard deviation TSIMAL = {:.3f}".format(std_TSIMAL))
    print("Standard deviation ENSEMBLE = {:.3f}".format(std_ENS))
    # print("MSE(TSIMAL,ENSEMBLE) = {:.3f}".format(mse_hs))
    print("r(TSIMAL,ENSEMBLE) = {:.3f}".format(r))
    print(
        "------------------------------------------------------------------------------------"
    )
    ### Computation of the error on annual mean surface temperature of TSIMAL with respect to the ENSEMBLE ###
    mean_ENS_Tsu = mean_all_mod(data=Tsu)
    mean_ENS_month_Tsu = month_mean(
        mean_ENS_Tsu
    )  # Computation of the month mean of hi ensemble mean #
    mean_month_TSIMAL_Tsu = month_mean(
        Tsu[:, 13]
    )  # Computation of the month mean of hs for TSIMAL #
    mean_TSIMAL_Tsu, mean_ENS_Tsu, err_abs, err_rel = err_annual_mean_thick(
        mean_month_TSIMAL_Tsu, mean_ENS_month_Tsu
    )
    mse_Tsu = MSE_annual_mean_thick(
        data1=mean_month_TSIMAL_Tsu, data2=mean_ENS_month_Tsu
    )
    std_ENS = std_var_mean_thick(data=mean_ENS_month_Tsu)
    std_TSIMAL = std_var_mean_thick(data=mean_month_TSIMAL_Tsu)
    r = cor_annual_mean_thick(data1=mean_month_TSIMAL_Tsu, data2=mean_ENS_month_Tsu)
    print(
        "-------------------- TSIMAL & ENSEMBLE SURFACE TEMPERATURE COMPARISON ----------------------"
    )
    print(
        "------------------------------------------------------------------------------------"
    )
    print("Mean surface temperature TSIMAL = {:.3f}°K".format(mean_TSIMAL_Tsu))
    print("Mean surface temperature ENSEMBLE = {:.3f}°K".format(mean_ENS_Tsu))
    print("Absolute Error = {:.4f}°K".format(err_abs))
    print("Relative Error = {:.3f}%".format(err_rel))
    print("Standard deviation TSIMAL = {:.3f}".format(std_TSIMAL))
    print("Standard deviation ENSEMBLE = {:.3f}".format(std_ENS))
    print("MSE(TSIMAL,ENSEMBLE) = {:.3f}".format(mse_Tsu))
    print("r(TSIMAL,ENSEMBLE) = {:.3f}".format(r))
    print(
        "------------------------------------------------------------------------------------"
    )


def comp_SIGUS_ENS():
    ##### Comparison between ENSEMBLE and SIGUS #####
    ### Computation of the error on annual mean ice thickness of SIGUS with respect to the ENSEMBLE ###
    mean_ENS_hi = mean_all_mod(data=hi)
    mean_ENS_month_hi = month_mean(
        mean_ENS_hi
    )  # Computation of the month mean of hi ensemble mean #
    mean_month_SIGUS_hi = month_mean(
        hi[:, 11]
    )  # Computation of the month mean of hi for SIGUS #
    mean_SIGUS_hi, mean_ENS_hi, err_abs, err_rel = err_annual_mean_thick(
        mean_month_SIGUS_hi, mean_ENS_month_hi
    )
    mse_hi = MSE_annual_mean_thick(data1=mean_month_SIGUS_hi, data2=mean_ENS_month_hi)
    std_ENS = std_var_mean_thick(data=mean_ENS_month_hi)
    std_SIGUS = std_var_mean_thick(data=mean_month_SIGUS_hi)
    r = cor_annual_mean_thick(data1=mean_month_SIGUS_hi, data2=mean_ENS_month_hi)
    print(
        "-------------------- SIGUS & ENSEMBLE ICE THICKNESS COMPARISON ----------------------"
    )
    print(
        "------------------------------------------------------------------------------------"
    )
    print("Mean ice thickness SIGUS = {:.3f}m".format(mean_SIGUS_hi))
    print("Mean ice thickness ENSEMBLE = {:.3f}m".format(mean_ENS_hi))
    print("Absolute Error = {:.4f}m".format(err_abs))
    print("Relative Error = {:.3f}%".format(err_rel))
    print("Standard deviation SIGUS = {:.3f}".format(std_SIGUS))
    print("Standard deviation ENSEMBLE = {:.3f}".format(std_ENS))
    print("MSE(SIGUS,ENSEMBLE) = {:.3f}".format(mse_hi))
    print("r(SIGUS,ENSEMBLE) = {:.3f}".format(r))
    print(
        "------------------------------------------------------------------------------------"
    )
    ### Computation of the error on annual mean snow thickness of SIGUS with respect to the ENSEMBLE ###
    mean_ENS_hs = mean_all_mod(data=hs)
    mean_ENS_month_hs = month_mean(
        mean_ENS_hs
    )  # Computation of the month mean of hi ensemble mean #
    mean_month_SIGUS_hs = month_mean(
        hs[:, 11]
    )  # Computation of the month mean of hs for SIGUS #
    mean_SIGUS_hs, mean_ENS_hs, err_abs, err_rel = err_annual_mean_thick(
        mean_month_SIGUS_hs, mean_ENS_month_hs
    )
    # mse_hs = MSE_annual_mean_thick(data1=mean_month_SIGUS_hs, data2=mean_ENS_month_hs)
    std_ENS = std_var_mean_thick(data=mean_ENS_month_hs)
    std_SIGUS = std_var_mean_thick(data=mean_month_SIGUS_hs)
    r = cor_annual_mean_thick(data1=mean_month_SIGUS_hs, data2=mean_ENS_month_hs)
    print(
        "-------------------- SIGUS & ENSEMBLE SNOW THICKNESS COMPARISON ----------------------"
    )
    print(
        "------------------------------------------------------------------------------------"
    )
    print("Mean snow thickness SIGUS = {:.3f}m".format(mean_SIGUS_hs))
    print("Mean snow thickness ENSEMBLE = {:.3f}m".format(mean_ENS_hs))
    print("Absolute Error = {:.4f}m".format(err_abs))
    print("Relative Error = {:.3f}%".format(err_rel))
    print("Standard deviation SIGUS = {:.3f}".format(std_SIGUS))
    print("Standard deviation ENSEMBLE = {:.3f}".format(std_ENS))
    # print("MSE(SIGUS,ENSEMBLE) = {:.3f}".format(mse_hs))
    print("r(SIGUS,ENSEMBLE) = {:.3f}".format(r))
    print(
        "------------------------------------------------------------------------------------"
    )
    ### Computation of the error on annual mean surface temperature of SIGUS with respect to the ENSEMBLE ###
    mean_ENS_Tsu = mean_all_mod(data=Tsu)
    mean_ENS_month_Tsu = month_mean(
        mean_ENS_Tsu
    )  # Computation of the month mean of hi ensemble mean #
    mean_month_SIGUS_Tsu = month_mean(
        Tsu[:, 11]
    )  # Computation of the month mean of hs for SIGUS #
    mean_SIGUS_Tsu, mean_ENS_Tsu, err_abs, err_rel = err_annual_mean_thick(
        mean_month_SIGUS_Tsu, mean_ENS_month_Tsu
    )
    mse_Tsu = MSE_annual_mean_thick(
        data1=mean_month_SIGUS_Tsu, data2=mean_ENS_month_Tsu
    )
    std_ENS = std_var_mean_thick(data=mean_ENS_month_Tsu)
    std_SIGUS = std_var_mean_thick(data=mean_month_SIGUS_Tsu)
    r = cor_annual_mean_thick(data1=mean_month_SIGUS_Tsu, data2=mean_ENS_month_Tsu)
    print(
        "-------------------- SIGUS & ENSEMBLE SURFACE TEMPERATURE COMPARISON ----------------------"
    )
    print(
        "------------------------------------------------------------------------------------"
    )
    print("Mean surface temperature SIGUS = {:.3f}°K".format(mean_SIGUS_Tsu))
    print("Mean surface temperature ENSEMBLE = {:.3f}°K".format(mean_ENS_Tsu))
    print("Absolute Error = {:.4f}°K".format(err_abs))
    print("Relative Error = {:.3f}%".format(err_rel))
    print("Standard deviation SIGUS = {:.3f}".format(std_SIGUS))
    print("Standard deviation ENSEMBLE = {:.3f}".format(std_ENS))
    print("MSE(SIGUS,ENSEMBLE) = {:.3f}".format(mse_Tsu))
    print("r(SIGUS,ENSEMBLE) = {:.3f}".format(r))
    print(
        "------------------------------------------------------------------------------------"
    )


##### Display #####


def plot_all_mod(data, data_name, N_mod, extra_label):
    figure = plt.figure(figsize=(16, 10))

    if data_name == "hi_mean_month":
        lab_size_fact = 0.5
        lab_size_fact_mod = 0.75
    else:
        lab_size_fact = 1
        lab_size_fact_mod = 1
    if np.shape(data)[0] == 365:
        time_range = time_range_ctl
        time_range_mu = time_range_MU71
        Nbre = N_days_CTL
        label_x = "Days"
    if np.shape(data)[0] == 12:
        time_range = time_range_ctl_month
        time_range_mu = time_range_MU71_month
        Nbre = N_month_CTL
        label_x = "Month"
    elif np.shape(data)[0] == 100:
        time_range = time_range_pr
        Nbre = N_years_PR
        label_x = "Year"
    for model in range(N_mod):
        plt.plot(time_range, data[:, model], linewidth=1 * lab_size_fact_mod)
    if data_name == "hi":
        plt.plot(
            time_range_mu,
            hi_MU71,
            label=r"$h_{i_{MU71}}$",
            linewidth=4 * lab_size_fact,
            color="tab:green",
        )
    if data_name == "hi_mean_month":
        plt.plot(
            time_range_mu,
            hi_MU71,
            label=r"$h_{i_{MU71}}$",
            linewidth=4 * lab_size_fact,
            color="tab:green",
        )
    mod_mean = mean_all_mod(data=data)
    # print(mod_mean)
    plt.plot(
        time_range,
        mod_mean,
        linewidth=4 * lab_size_fact,
        color="tab:blue",
        label=r"Models Mean",
    )
    N_mod_str = str(N_mod)
    plt.title(
        r"Comparison between ensemble members"
        + r"($N_{mod}$ ="
        + N_mod_str
        + ")"
        + r" and their averages "
        + extra_label
        + " simulation",
        size=24 * lab_size_fact,
    )
    plt.xlabel(label_x, size=25 * lab_size_fact)
    if (
        data_name == "hi"
        or data_name == "hi_mean_month"
        or data_name == "himax"
        or data_name == "himean"
        or data_name == "himin"
    ):
        unit = "[m]"
    if data_name == "Tsu" or data_name == "Tsu_mean_month" or data_name == "Tsumin":
        unit = "[°K]"
    if data_name == "hs" or data_name == "hs_mean_month" or data_name == "hsmax":
        unit = "[m]"
    plt.ylabel(data_name + unit, size=25 * lab_size_fact)
    plt.xticks(fontsize=20 * lab_size_fact)
    plt.yticks(fontsize=20 * lab_size_fact)
    plt.grid()
    plt.legend(fontsize=20 * lab_size_fact)
    plt.savefig(save_dir + data_name + "_all_mod_" + extra_label + ".png", dpi=300)
    # plt.show()
    plt.clf()
    plt.legend(fontsize=20)
    #plt.savefig(save_dir + data_name + "_all_mod000.png", dpi=300)
    plt.show()
    #plt.clf()

def summarized_projection(display_single_models = False, save = False):
    """
        Plots all relevant informations about projection.
        Turn displaY_single_models to True if you want to have see all the models results.
    """
    PR03_himean = himean_pr[:,:,0]
    PR03_himax = himax_pr[:,:,0]
    PR03_himin = himin_pr[:,:,0]
    PR03_hsmax = hsmax_pr[:,:,0]
    PR03_Tsumin = Tsumin_pr[:,:,0]
    
    PR06_himean = himean_pr[:,:,1]
    PR06_himax = himax_pr[:,:,1]
    PR06_himin = himin_pr[:,:,1]
    PR06_hsmax = hsmax_pr[:,:,1]
    PR06_Tsumin = Tsumin_pr[:,:,1]

    PR12_himean = himean_pr[:,:,2]
    PR12_himax = himax_pr[:,:,2]
    PR12_himin = himin_pr[:,:,2]
    PR12_hsmax = hsmax_pr[:,:,2]
    PR12_Tsumin = Tsumin_pr[:,:,2]

    # The following arrays have dims (n_proj = 3,n_year = 100, n_mod = 14):   -n_proj for the projection, 
    #                                                                         -n_year for the year
    #                                                                         -n_mod for the model 
    PR_himean = np.array([PR03_himean,PR06_himean,PR12_himean])
    PR_himax = np.array([PR03_himax,PR06_himax,PR12_himax])
    PR_himin = np.array([PR03_himin,PR06_himin,PR12_himin])
    PR_hsmax = np.array([PR03_hsmax,PR06_hsmax,PR12_hsmax])
    PR_Tsumin = np.array([PR03_Tsumin,PR06_Tsumin,PR12_Tsumin])

    #Regrouping all datas in a single dictionnary to allows faster and more flexible plotting.
    Projections = {"himean":PR_himean,"himax":PR_himax,"himin":PR_himin,"hsmax":PR_hsmax,"Tsumin":PR_Tsumin}

    for key in Projections.keys():
        mean = np.mean(Projections[key], axis = 2)
        std = np.std(Projections[key], axis = 2)/2
        plt.title(f'Multi-model analysis for the variable: {key}')

        #### - PR03 - ####
        # Individuals models
        if display_singel_models:
            plt.plot([year for year in range(100)],Projections[key][0,:,:], alpha=0.25,color = "tab:blue")
        # ensemble mean
        plt.plot([year for year in range(100)],mean[0,:],color = "tab:blue", label = 'PR03 mean',linewidth=4)
        # shadow std
        plt.fill_between([year for year in range(100)], mean[0,:] - std[0,:], mean[0,:] + std[0,:],alpha = 0.5,color = "tab:blue", label= '+/- std/2')
        
        #### - PR06 - ####
        # Individuals models
        if display_singel_models:
            plt.plot([year for year in range(100)],Projections[key][1,:,:], alpha=0.25,color = "tab:orange")
        # ensemble mean
        plt.plot([year for year in range(100)],mean[1,:],color = "tab:orange", label = 'PR06 mean',linewidth=4)
        # shadow std
        plt.fill_between([year for year in range(100)], mean[1,:] - std[1,:], mean[1,:] + std[1,:],alpha = 0.5,color = "tab:orange", label= '+/- std/2')
        
        #### - PR12 - ####
        # Individuals models
        if display_single_models:
            plt.plot([year for year in range(100)],Projections[key][2,:,:], alpha=0.25,color = "tab:red")
        # ensemble mean
        plt.plot([year for year in range(100)],mean[2,:],color = "tab:red", label = 'PR12 mean',linewidth=4)
        # shadow std
        plt.fill_between([year for year in range(100)], mean[2,:] - std[2,:], mean[2,:] + std[2,:],alpha = 0.5,color = "tab:red", label= '+/- std/2')

        plt.legend()
        plt.grid()
        if save:
            plt.savefig('fig_ctl/'+str(key)+'.png')
            plt.clf()
        else:    
            plt.show()



summarized_projection(display_single_models=False, save = False)





def subplot_all_mod(data1, data2, data3, data_name, N_mod, extra_label):
    figure = plt.figure(figsize=(16, 10))

    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.3, left=0.1, right=0.98)
    plt.figure()
    plt.suptitle(
        r"Ensemble model output $(h_i, T_{su}, h_s)$ for the "
        + extra_label
        + " simulation"
    )

    if np.shape(data1)[0] == 365:
        time_range = time_range_ctl
        time_range_mu = time_range_MU71
        Nbre = N_days_CTL
        label_x = "Days"
        x_ticks = np.arange(0, N_days_CTL, 50)
    if np.shape(data1)[0] == 12:
        time_range = time_range_ctl_month
        time_range_mu = time_range_MU71_month
        Nbre = N_month_CTL
        label_x = "Month"
        x_ticks = np.arange(0, N_month_CTL, 2)
    ### Figure 1 ###
    ax = plt.subplot(gs[0, 0])  # row 0, col 0
    for model in range(N_mod):
        plt.plot(time_range, data1[:, model], linewidth=0.5)
    plt.plot(
        time_range_mu,
        hi_MU71,
        label=r"$h_{i_{MU71}}$",
        linewidth=2,
        color="tab:green",
    )
    mod_mean = mean_all_mod(data=data1)
    plt.plot(time_range, mod_mean, linewidth=2, color="tab:blue", label=r"Models Mean")
    plt.xlabel(label_x, size=10)
    plt.ylabel("Ice Thickness [m]", size=10)
    plt.xticks(x_ticks, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.legend(fontsize=10)

    ### Figure 2 ###
    ax = plt.subplot(gs[0, 1])  # row 0, col 1
    for model in range(N_mod):
        plt.plot(time_range_ctl, data2[:, model], linewidth=0.5)
    mod_mean = mean_all_mod(data=data2)
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
    mod_mean = mean_all_mod(data=data3)
    plt.plot(
        time_range_ctl, mod_mean, linewidth=2, color="tab:blue", label=r"Models Mean"
    )
    plt.xlabel("Days", size=10)
    plt.ylabel("Snow Thickness [m]", size=10)
    plt.xticks(np.arange(0, N_days_CTL, 25), fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.legend(fontsize=10)

    plt.savefig(save_dir + "subplot_all_mod_" + extra_label + ".png", dpi=300)
    plt.clf()


def summarized_projection(display_single_models=False, save=False):
    """
    Plots all relevant informations about projection.
    Turn displaY_single_models to True if you want to have see all the models results.
    """
    PR03_himean = himean[:, :, 0]
    PR03_himax = himax[:, :, 0]
    PR03_himin = himin[:, :, 0]
    PR03_hsmax = hsmax[:, :, 0]
    PR03_Tsumin = Tsumin[:, :, 0]

    PR06_himean = himean[:, :, 1]
    PR06_himax = himax[:, :, 1]
    PR06_himin = himin[:, :, 1]
    PR06_hsmax = hsmax[:, :, 1]
    PR06_Tsumin = Tsumin[:, :, 1]

    PR12_himean = himean[:, :, 2]
    PR12_himax = himax[:, :, 2]
    PR12_himin = himin[:, :, 2]
    PR12_hsmax = hsmax[:, :, 2]
    PR12_Tsumin = Tsumin[:, :, 2]

    # The following arrays have dims (n_proj = 3,n_year = 100, n_mod = 14):   -n_proj for the projection,
    #                                                                         -n_year for the year
    #                                                                         -n_mod for the model
    PR_himean = np.array([PR03_himean, PR06_himean, PR12_himean])
    PR_himax = np.array([PR03_himax, PR06_himax, PR12_himax])
    PR_himin = np.array([PR03_himin, PR06_himin, PR12_himin])
    PR_hsmax = np.array([PR03_hsmax, PR06_hsmax, PR12_hsmax])
    PR_Tsumin = np.array([PR03_Tsumin, PR06_Tsumin, PR12_Tsumin])

    # Regrouping all datas in a single dictionnary to allows faster and more flexible plotting.
    Projections = {
        "himean": PR_himean,
        "himax": PR_himax,
        "himin": PR_himin,
        "hsmax": PR_hsmax,
        "Tsumin": PR_Tsumin,
    }

    for key in Projections.keys():
        mean = np.mean(Projections[key], axis=2)
        std = np.std(Projections[key], axis=2) / 2
        plt.title(f"Multi-model Analysis for PR Scenarios : {key}", size=28)
        if key == "himean":
            y_label = r"$hi_{mean} [m]$"
        if key == "himax":
            y_label = r"$hi_{max} [m]$"
        if key == "himin":
            y_label = r"$hi_{min} [m]$"
        if key == "hsmax":
            y_label = r"$hs_{max} [m]$"
        if key == "Tsumin":
            y_label = r"$Tsu_{min} [{}^{\circ}K]$"
        #### - PR03 - ####
        # Individuals models
        if display_single_models:
            plt.plot(
                [year for year in range(N_years_PR)],
                Projections[key][0, :, :],
                alpha=0.6,
                color="tab:blue",
            )
        # ensemble mean
        plt.plot(
            [year for year in range(N_years_PR)],
            mean[0, :],
            color="tab:blue",
            label=r"$\mu_{PR03}$",
            linewidth=4,
        )
        # shadow std
        plt.fill_between(
            [year for year in range(N_years_PR)],
            mean[0, :] - std[0, :],
            mean[0, :] + std[0, :],
            alpha=0.7,
            color="tab:blue",
            label=r"$\mu_{PR03} \pm \frac{\sigma_{PR03}}{2}$",
        )

        #### - PR06 - ####
        # Individuals models
        if display_single_models:
            plt.plot(
                [year for year in range(N_years_PR)],
                Projections[key][1, :, :],
                alpha=0.6,
                color="tab:orange",
            )
        # ensemble mean
        plt.plot(
            [year for year in range(N_years_PR)],
            mean[1, :],
            color="tab:orange",
            label=r"$\mu_{PR06}$",
            linewidth=4,
        )
        # shadow std
        plt.fill_between(
            [year for year in range(N_years_PR)],
            mean[1, :] - std[1, :],
            mean[1, :] + std[1, :],
            alpha=0.7,
            color="tab:orange",
            label=r"$\mu_{PR06} \pm \frac{\sigma_{PR06}}{2}$",
        )

        #### - PR12 - ####
        # Individuals models
        if display_single_models:
            plt.plot(
                [year for year in range(N_years_PR)],
                Projections[key][2, :, :],
                alpha=0.6,
                color="tab:red",
            )
        # ensemble mean
        plt.plot(
            [year for year in range(N_years_PR)],
            mean[2, :],
            color="tab:red",
            label=r"$\mu_{PR12}$",
            linewidth=4,
        )
        # shadow std
        plt.fill_between(
            [year for year in range(N_years_PR)],
            mean[2, :] - std[2, :],
            mean[2, :] + std[2, :],
            alpha=0.7,
            color="tab:red",
            label=r"$\mu_{PR12} \pm \frac{\sigma_{PR12}}{2}$",
        )

        plt.legend(fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.ylabel(y_label, size=29)
        plt.xlabel("Year", size=29)
        plt.grid()
        if save:
            plt.savefig(save_dir + "PR_MutliMod_" + str(key) + ".png")
            plt.clf()
        else:
            plt.show()


def subplot_TSIMAL_SIGUS_ENS(data1, data2, data3, N_mod, extra_label):
    figure = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.3, left=0.1, right=0.98)
    plt.figure()
    plt.suptitle(
        r"Ensemble model and TISMAL\&SIGUS "
        + r"output $(h_i, T_{su}, h_s)$ "
        + extra_label
        + " simulation"
    )
    if np.shape(data1)[0] == 365:
        time_range = time_range_ctl
        Nbre = N_days_CTL
        label_x = "Days"
        x_ticks = np.arange(Day_0, 365, 50)
    if np.shape(data1)[0] == 12:
        time_range = time_range_ctl_month
        Nbre = N_month_CTL
        label_x = "Month"
        x_ticks = np.arange(Day_0, 12, 6)
    elif np.shape(data1)[0] == 100:
        time_range = time_range_pr
        Nbre = N_years_PR
        label_x = "Year"
        x_ticks = np.arange(Day_0, 100, 20)
    ### Figure 1 ###
    ax = plt.subplot(gs[0, 0])  # row 0, col 0
    plt.plot(time_range, data1[:, 13], linewidth=1, label="TSIMAL", color="red")
    plt.plot(time_range, data1[:, 11], linewidth=1, label="SIGUS", color="orange")
    mod_mean = mean_all_mod(data=data1)
    plt.plot(time_range, mod_mean, linewidth=2, color="tab:blue", label=r"Models Mean")
    plt.xlabel(label_x, size=10)
    plt.ylabel("Ice Thickness [m]", size=10)
    plt.xticks(x_ticks, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.legend(fontsize=10)

    ### Figure 2 ###
    ax = plt.subplot(gs[0, 1])  # row 0, col 1
    plt.plot(time_range, data2[:, 13], linewidth=1, label="TSIMAL", color="red")
    plt.plot(time_range, data2[:, 11], linewidth=1, label="SIGUS", color="orange")
    mod_mean = mean_all_mod(data=data2)
    plt.plot(time_range, mod_mean, linewidth=2, color="tab:blue", label=r"Models Mean")
    plt.xlabel(label_x, size=10)
    plt.ylabel("Surface Temperature [°K]", size=10)
    plt.xticks(x_ticks, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.legend(fontsize=10)

    ### Figure 3 ###
    ax = plt.subplot(gs[1, :])  # row 1, span all columns
    plt.plot(time_range, data3[:, 13], linewidth=1, label="TSIMAL", color="red")
    plt.plot(time_range, data3[:, 11], linewidth=1, label="SIGUS", color="orange")
    mod_mean = mean_all_mod(data=data3)
    plt.plot(time_range, mod_mean, linewidth=2, color="tab:blue", label=r"Models Mean")
    plt.xlabel(label_x, size=10)
    plt.ylabel("Snow Thickness [m]", size=10)
    plt.xticks(x_ticks, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.legend(fontsize=10)
    plt.savefig(save_dir + "subplot_TSIMAL_SIGUS_ENS_" + extra_label + ".png", dpi=300)
    plt.clf()

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
time_range_MU71 = np.linspace(Day_0, N_days_CTL, 12)
time_range_MU71_month = np.linspace(Day_0, N_month_CTL, 12)
time_range_ctl = np.arange(
    Day_0, N_days_CTL, 1
)  # time range for the CTL simulations in days. Used for plot
time_range_ctl_month = np.arange(
    Month_0, N_month_CTL, 1
)  # time range for the CTL simulations in month. Used for plot
time_range_pr = np.arange(
    Day_0, N_years_PR, 1
)  # time range for the PR simulations in month. Used for plot



""" if __name__ == "__main__":
    ######################################## Control Simulations Analysis #########################################
    ##### Plot of the ensemble simulations with daily resolution #####
    """plot_all_mod(data=hi, data_name="hi", N_mod=N_mod_CTL, extra_label="CTL")
    plot_all_mod(data=Tsu, data_name="Tsu", N_mod=N_mod_CTL, extra_label="CTL")
    plot_all_mod(data=hs, data_name="hs", N_mod=N_mod_CTL, extra_label="CTL")
    subplot_all_mod(
        data1=hi,
        data2=Tsu,
        data3=hs,
        data_name=["hi,Tsu,hs"],
        N_mod=N_mod_CTL,
        extra_label="CTL",
    )

    ### Computation of the month mean of the variables ###
    # hi #
    hi_mean_month = np.zeros((12, N_mod_CTL))
    for model in range(N_mod_CTL):
        hi_mean_month_mod = month_mean(hi[:, model])
        hi_mean_month[:, model] = hi_mean_month_mod
    # print(hi_mean_month)
    # hs #
    hs_mean_month = np.zeros((12, N_mod_CTL))
    for model in range(N_mod_CTL):
        hs_mean_month_mod = month_mean(hs[:, model])
        hs_mean_month[:, model] = hs_mean_month_mod
    # print(hs_mean_month)
    # hi #
    Tsu_mean_month = np.zeros((12, N_mod_CTL))
    for model in range(N_mod_CTL):
        Tsu_mean_month_mod = month_mean(Tsu[:, model])
        Tsu_mean_month[:, model] = Tsu_mean_month_mod
    # print(Tsu_mean_month)
    ### Plot of the ensemble simulations with month resolution ###
    plot_all_mod(
        data=hi_mean_month,
        data_name="hi_mean_month",
        N_mod=N_mod_CTL,
        extra_label="CTL",
    )
    subplot_all_mod(
        data1=hi_mean_month,
        data2=Tsu,
        data3=hs,
        data_name=["hi_mean_month,Tsu,hs"],
        N_mod=N_mod_CTL,
        extra_label="CTL",
    )

    ########## Verification ###########
    comp_ENS_MU71()
    comp_TSIMAL_MU71()
    comp_SIGUS_MU71()
    comp_TSIMAL_ENS()
    comp_SIGUS_ENS()
    subplot_TSIMAL_SIGUS_ENS()
    ######################################## Projection Simulations Analysis ##################################### """
