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

################################################### Parameters #############################################################

################################ Script Parameters #######################################
N_mod_CTL = 15  # number of models at disposal for the CTL run [Adim]
N_mod_PR = 14  # number of models at disposal for the PR run [Adim]
N_years_CTL = 1  # number of years in the CTL simulation [Adim]
N_days_CTL = 365 * N_years_CTL  # number of days in the CTL simulation [Adim]
N_month_CTL = 12 * N_years_CTL  # number of month in the CTL simulation [Adim]
N_years_PR = 100  # number of years in the PR simulation [Adim]
N_days_PR = 365 * N_years_PR  # number of days in the PR simulation [Adim]
N_pr = 3  # number of projection scenario used
Day_0 = 0
Month_0 = 0
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
model_index_without_snow_student = [0, 2, 3, 6, 8]
for i in model_index_without_snow:
    hs[:, i] = np.NaN

model_index_student = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
model_name_vancop_Semt_0_lay = ["SUPERN_ICE"]
model_index_vancop_Semt_0_lay = [12]
model_name_vancop_diff = ["BRAILLE_ANE", "YBLAIRE"]
model_index_vancop_diff = [1, 14]


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

##################################################### Tools function #######################################################

##### Mean computation #####


def mean_all_mod(data):
    # transform the 0 in Nan
    # data = np.where(data == 0, np.nan, data)
    return np.mean(ma.masked_invalid(data), axis=1)


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


def plot_all_mod2(
    data, data_name, N_mod, extra_label, display_single_models=False, model_mean=False
):
    """Almost same plotting function as plot_all_mod() but with a distinction between the models that have snow implemented and those
    who haven't."""

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
    if data_name[0] == "h":
        unit = "[m]"
    if data_name[0] == "T":
        unit = "[°K]"
    N_mod_str = str(N_mod)

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
    ####### - Plots - ######

    ##- Individuals models - ##
    if display_single_models:
        # without snow
        if data_name != "hs":
            plt.plot(
                time_range,
                np.take(data, model_index_without_snow, axis=1),
                alpha=0.6,
                linewidth=1 * lab_size_fact_mod,
                color="tab:red",
            )
        # with snow
        plt.plot(
            time_range,
            np.delete(data, model_index_without_snow, axis=1),
            alpha=0.6,
            linewidth=1 * lab_size_fact_mod,
            color="tab:orange",
        )

    ## - Mean - ##
    # With snow
    plt.plot(
        time_range,
        np.mean(np.delete(data, model_index_without_snow, axis=1), axis=1),
        linewidth=4 * lab_size_fact,
        color="tab:orange",
        label=r"SM",
    )
    # Without snow
    if data_name != "hs":
        plt.plot(
            time_range,
            np.mean(np.take(data, model_index_without_snow, axis=1), axis=1),
            linewidth=4 * lab_size_fact,
            color="tab:red",
            label=r"SFM",
        )

    ## - Std - ##

    # With snow
    mean_std = round(
        np.mean(np.std(np.delete(data, model_index_without_snow, axis=1), axis=1)), 2
    )
    plt.fill_between(
        time_range,
        np.mean(np.delete(data, model_index_without_snow, axis=1), axis=1)
        - np.std(np.delete(data, model_index_without_snow, axis=1), axis=1),
        np.mean(np.delete(data, model_index_without_snow, axis=1), axis=1)
        + np.std(np.delete(data, model_index_without_snow, axis=1), axis=1),
        alpha=0.5,
        color="tab:orange",
        label=r"$\sigma_{SM}$ = " + str(mean_std) + unit,
    )
    if data_name != "hs":
        # Without snow
        mean_std = round(
            np.mean(np.std(np.take(data, model_index_without_snow, axis=1), axis=1)), 2
        )
        plt.fill_between(
            time_range,
            np.mean(np.take(data, model_index_without_snow, axis=1), axis=1)
            - np.std(np.take(data, model_index_without_snow, axis=1), axis=1),
            np.mean(np.take(data, model_index_without_snow, axis=1), axis=1)
            + np.std(np.take(data, model_index_without_snow, axis=1), axis=1),
            alpha=0.5,
            color="tab:red",
            label=r"$\sigma_{SFM}$ = " + str(mean_std) + unit,
        )

    #### - Ensemble model Mean - ####
    if model_mean:
        ## - Mean - ##
        mean_ENS = mean_all_mod(data)
        plt.plot(
            time_range,
            mean_ENS,
            linewidth=4 * lab_size_fact,
            color="tab:blue",
            label=r"ENS",
        )

        ## - Std - ##
        if data_name == "hs":
            # print("mean_ENS", mean_ENS)
            std = np.std(mean_ENS)
            # print(std)
            std_round = round(std, 2)
            std_str = str(std_round)

            plt.fill_between(
                time_range,
                mean_ENS - std,
                mean_ENS + std,
                alpha=0.5,
                color="tab:blue",
                label=r"$\sigma_{ENS}$ = " + std_str + unit,
            )
        else:
            # print("data", data)
            std = np.mean(np.std(data, axis=1))
            # print(std)
            std_round = round(std, 2)
            std_str = str(std_round)
            # mean_std = round(np.mean(np.std(data, axis=1)), 2)
            plt.fill_between(
                time_range,
                np.mean(data, axis=1) - np.std(data, axis=1),
                np.mean(data, axis=1) + np.std(data, axis=1),
                alpha=0.5,
                color="tab:blue",
                label=r"$\sigma_{ENS}$ = " + std_str + unit,
            )

    ## - MU71 - ##
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
        plt.plot(
            time_range_mu,
            hi_MU71,
            label=r"$h_{i_{MU71}}$",
            linewidth=4 * lab_size_fact,
            color="tab:green",
        )

    ###### - plots parameters - ######

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
    plt.ylabel(data_name + unit, size=25 * lab_size_fact)
    plt.xticks(fontsize=20 * lab_size_fact)
    plt.yticks(fontsize=20 * lab_size_fact)
    plt.grid()
    plt.legend(fontsize=20 * lab_size_fact)
    if display_single_models:
        extra_label_2 = "_dsm"
    else:
        extra_label_2 = ""
    if model_mean:
        extra_label_3 = "_mm"
    else:
        extra_label_3 = ""
    plt.savefig(
        save_dir
        + extra_label
        + "_"
        + data_name
        + "_DS"
        + extra_label_2
        + extra_label_3
        + ".png",
        dpi=300,
    )
    # plt.show()
    plt.clf()


def subplot_all_mod(data1, data2, data3, data_name, N_mod, extra_label):
    """Subplot with all models plots for three differents variables"""
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


def subplot_all_mod2(
    data,
    data_name,
    N_mod,
    extra_label,
    display_single_models=False,
    dist_models=False,
    supern_displ=False,
):
    """Subplot with models ENS mean and Std + Distinction for models having snow and those who doesnt. Plot of the mean ice thickness and it's daily variation"""
    fig = plt.figure(figsize=(16, 10))
    # gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.1, left=0.001, right=0.98)

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
    if data_name[0] == "h":
        unit = "[m]"
    if data_name[0] == "T":
        unit = "[°K]"
    N_mod_str = str(N_mod)

    ####### - Plots - ######

    # fig = plt.figure()
    """ ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,1,2)
    axs = [ax1, ax2, ax3] """

    ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=2)
    ax2 = plt.subplot2grid((4, 4), (0, 2), rowspan=3, colspan=2)
    ax3 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
    ax4 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
    axs = [ax1, ax2, ax3, ax4]
    plt.subplots_adjust(bottom=0.08, right=0.97, left=0.09, top=0.9)

    ##### - Left subplots - ####
    ##- Individuals models - ##
    if dist_models == True:
        # model without snow #
        axs[0].plot(
            time_range,
            data[:, model_index_vancop_diff[0]],
            alpha=0.6,
            linewidth=3 * lab_size_fact_mod,
            color="tab:red",
            linestyle="dashed",
            label="BRAILLE_ANE",
        )
        # model with snow #
        axs[0].plot(
            time_range,
            data[:, model_index_vancop_diff[1]],
            alpha=0.6,
            linewidth=3 * lab_size_fact_mod,
            color="tab:orange",
            linestyle="-.",
            label="YBLAIRE",
        )
        if supern_displ == True:
            axs[0].plot(
                time_range,
                data[:, model_index_vancop_Semt_0_lay[0]],
                alpha=0.6,
                linewidth=3 * lab_size_fact_mod,
                color="tab:orange",
                linestyle=":",
                label="SUPERN_ICE",
            )
    if display_single_models:
        # without snow
        axs[0].plot(
            time_range,
            np.take(data, model_index_without_snow_student, axis=1),
            alpha=0.6,
            linewidth=1 * lab_size_fact_mod,
            color="tab:red",
        )
        # with snow
        axs[0].plot(
            time_range,
            np.delete(data, model_index_without_snow_student, axis=1),
            alpha=0.6,
            linewidth=1 * lab_size_fact_mod,
            color="tab:orange",
        )
    ## - Mean - ##
    # With snow
    axs[0].plot(
        time_range,
        np.mean(np.delete(data, model_index_without_snow, axis=1), axis=1),
        linewidth=4 * lab_size_fact,
        color="tab:orange",
        label=r"SM",
    )
    # Without snow
    axs[0].plot(
        time_range,
        np.mean(np.take(data, model_index_without_snow, axis=1), axis=1),
        linewidth=4 * lab_size_fact,
        color="tab:red",
        label=r"SFM",
    )

    ## - Std - ##

    # With snow
    mean_std = round(
        np.mean(np.std(np.delete(data, model_index_without_snow, axis=1), axis=1)), 2
    )
    axs[0].fill_between(
        time_range,
        np.mean(np.delete(data, model_index_without_snow, axis=1), axis=1)
        - np.std(np.delete(data, model_index_without_snow, axis=1), axis=1),
        np.mean(np.delete(data, model_index_without_snow, axis=1), axis=1)
        + np.std(np.delete(data, model_index_without_snow, axis=1), axis=1),
        alpha=0.5,
        color="tab:orange",
        label=r"$\sigma_{SM}$ = " + str(mean_std) + unit,
    )

    # Without snow
    mean_std = round(
        np.mean(np.std(np.take(data, model_index_without_snow, axis=1), axis=1)), 2
    )
    axs[0].fill_between(
        time_range,
        np.mean(np.take(data, model_index_without_snow, axis=1), axis=1)
        - np.std(np.take(data, model_index_without_snow, axis=1), axis=1),
        np.mean(np.take(data, model_index_without_snow, axis=1), axis=1)
        + np.std(np.take(data, model_index_without_snow, axis=1), axis=1),
        alpha=0.5,
        color="tab:red",
        label=r"$\sigma_{SFM}$ = " + str(mean_std) + unit,
    )

    #### - Right Subplots - ####
    if dist_models == True:
        # model without snow #
        axs[1].plot(
            time_range,
            data[:, model_index_vancop_diff[0]],
            alpha=0.6,
            linewidth=3 * lab_size_fact_mod,
            color="tab:red",
            linestyle="dashed",
            label="BRAILLE_ANE",
        )
        # model with snow #
        axs[1].plot(
            time_range,
            data[:, model_index_vancop_diff[1]],
            alpha=0.6,
            linewidth=3 * lab_size_fact_mod,
            color="tab:orange",
            linestyle="-.",
            label="YBLAIRE",
        )
        if supern_displ == True:
            axs[1].plot(
                time_range,
                data[:, model_index_vancop_Semt_0_lay[0]],
                alpha=0.6,
                linewidth=3 * lab_size_fact_mod,
                color="tab:orange",
                linestyle=":",
                label="SUPERN_ICE",
            )
    if display_single_models:
        # without snow
        axs[1].plot(
            time_range,
            np.take(data, model_index_without_snow, axis=1),
            alpha=0.6,
            linewidth=1 * lab_size_fact_mod,
            color="tab:red",
        )
        # with snow
        axs[1].plot(
            time_range,
            np.delete(data, model_index_without_snow, axis=1),
            alpha=0.6,
            linewidth=1 * lab_size_fact_mod,
            color="tab:orange",
        )

    ## - Mean - ##

    axs[1].plot(
        time_range,
        np.mean(data, axis=1),
        linewidth=4 * lab_size_fact,
        color="tab:blue",
        label=r"ENS",
    )

    ## - Std - ##
    mean_std = round(np.mean(np.std(data, axis=1)), 2)
    axs[1].fill_between(
        time_range,
        np.mean(data, axis=1) - np.std(data, axis=1),
        np.mean(data, axis=1) + np.std(data, axis=1),
        alpha=0.5,
        color="tab:blue",
        label=r"$\sigma_{ENS}$ = " + str(mean_std) + unit,
    )

    ## - Variation - ##
    axs[2].plot(
        time_range,
        np.gradient(np.mean(np.take(data, model_index_without_snow, axis=1), axis=1))
        * 100,
        color="tab:red",
        label="SFM",
    )  # Plot the day variation from a day to an other
    axs[2].plot(
        time_range,
        np.gradient(np.mean(np.delete(data, model_index_without_snow, axis=1), axis=1))
        * 100,
        color="tab:orange",
        label="SM",
    )  # Plot the day variation from a day to an other

    axs[3].plot(
        time_range, np.gradient(np.mean(data, axis=1)) * 100, label="ENS"
    )  # Plot the day variation from a day to an other

    ## - MU71 - ##
    if data_name == "hi":
        axs[0].plot(
            time_range_mu,
            hi_MU71,
            label=r"$h_{i_{MU71}}$",
            linewidth=4 * lab_size_fact,
            color="tab:green",
        )
        axs[1].plot(
            time_range_mu,
            hi_MU71,
            label=r"$h_{i_{MU71}}$",
            linewidth=4 * lab_size_fact,
            color="tab:green",
        )
    if data_name == "hi_mean_month":
        axs[0].plot(
            time_range_mu,
            hi_MU71,
            label=r"$h_{i_{MU71}}$",
            linewidth=4 * lab_size_fact,
            color="tab:green",
        )
        axs[1].plot(
            time_range_mu,
            hi_MU71,
            label=r"$h_{i_{MU71}}$",
            linewidth=4 * lab_size_fact,
            color="tab:green",
        )

    ###### - plots parameters - ######

    fig.suptitle(
        r"Comparison between ensemble members "
        + r"($N_{mod}$ ="
        + N_mod_str
        + ")"
        + r" and their averages "
        + extra_label
        + " simulation",
        size=24 * lab_size_fact,
    )
    # axs[0].set_xlabel(label_x, size=25 * lab_size_fact)
    # axs[0].set_xlabel(label_x, fontsize=20)
    axs[0].set_ylabel(data_name + unit, fontsize=22)
    axs[0].tick_params(axis="x", labelsize=20)
    axs[0].tick_params(axis="y", labelsize=20)
    axs[0].grid()
    if data_name == "hi":
        axs[0].set_ylim(2.1, 3.75)
    axs[0].legend(fontsize=21 * lab_size_fact)

    # axs[1].set_xlabel(label_x, fontsize=20)
    # axs[1].set_ylabel(data_name + unit, fontsize=20)
    axs[1].tick_params(axis="x", labelsize=20)
    axs[1].tick_params(axis="y", labelsize=20)
    axs[1].grid()
    if data_name == "hi":
        axs[1].set_ylim(2.1, 3.75)
    axs[1].legend(fontsize=21 * lab_size_fact)

    axs[2].set_xlabel(label_x, fontsize=22)
    if data_name == "hi":
        axs[2].set_ylabel("Variation [cm/day]", fontsize=22)
    else:
        axs[2].set_ylabel("Variation " + unit + "/day", fontsize=22)
    axs[2].tick_params(axis="x", labelsize=20)
    axs[2].tick_params(axis="y", labelsize=20)
    axs[2].grid()
    axs[2].legend(fontsize=21 * lab_size_fact)
    if data_name == "hi":
        axs[2].set_ylim(-2, 0.75)

    axs[3].set_xlabel(label_x, fontsize=22)
    if data_name == "hi":
        axs[3].set_ylim(-2, 0.75)
    # axs[3].set_ylabel("Variation [cm/days]", fontsize=20)
    axs[3].tick_params(axis="x", labelsize=20)
    axs[3].tick_params(axis="y", labelsize=20)
    axs[3].grid()
    axs[3].legend(fontsize=21 * lab_size_fact)

    if display_single_models:
        extra_label_2 = "_dsm"
    else:
        extra_label_2 = ""
    if dist_models:
        extra_label_3 = "_dm"
    else:
        extra_label_3 = ""
    plt.savefig(
        save_dir
        + extra_label
        + "_Subplot_"
        + data_name
        + "_DS"
        + extra_label_2
        + extra_label_3
        + ".png",
        dpi=300,
    )
    # plt.show()
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
        figure = plt.figure(figsize=(16, 10))
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
            label=r"$PR03$",
            linewidth=4,
        )
        # shadow std
        plt.fill_between(
            [year for year in range(N_years_PR)],
            mean[0, :] - std[0, :],
            mean[0, :] + std[0, :],
            alpha=0.7,
            color="tab:blue",
            label=r"$\sigma_{PR03}$",
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
            label=r"$PR06$",
            linewidth=4,
        )
        # shadow std
        plt.fill_between(
            [year for year in range(N_years_PR)],
            mean[1, :] - std[1, :],
            mean[1, :] + std[1, :],
            alpha=0.7,
            color="tab:orange",
            label=r"$\sigma_{PR06}$",
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
            label=r"$PR12$",
            linewidth=4,
        )
        # shadow std
        plt.fill_between(
            [year for year in range(N_years_PR)],
            mean[2, :] - std[2, :],
            mean[2, :] + std[2, :],
            alpha=0.7,
            color="tab:red",
            label=r"$\sigma_{PR12}$",
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


def summarized_projection2(pr_label, pr_index, display_single_models=False):
    """Same as summarized_projection() but used for ploting trend of himax and himean for different projections in order to distinguish where are the most
    important impact of sea ice lost for each projection.
    Turn display_single_models to True if you want to have see all the models results.
    pr_index = 0 for PR03, = 1 for PR06 and = 2 for PR12
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

    figure = plt.figure(figsize=(16, 10))
    mean_max = np.mean(PR_himax[pr_index], axis=1)
    # print(mean_max)
    mean_min = np.mean(PR_himin[pr_index], axis=1)
    # print(mean_min)
    std_max = np.std(PR_himax[pr_index], axis=1)
    # print(std_max)
    std_min = np.std(PR_himin[pr_index], axis=1)
    # print(std_min)

    #### - PRXX - ####
    # Individuals models
    if display_single_models:
        plt.plot(
            [year for year in range(N_years_PR)],
            Projections[himin][pr_index, :, :],
            alpha=0.6,
            color="tab:red",
        )
        plt.plot(
            [year for year in range(N_years_PR)],
            Projections[himax][pr_index, :, :],
            alpha=0.6,
            color="tab:blue",
        )
    # ensemble mean hi_min
    plt.plot(
        [year for year in range(N_years_PR)],
        mean_min,
        color="tab:red",
        label=r"$hi_{min}$",
        linewidth=4,
    )
    # shadow std
    plt.fill_between(
        [year for year in range(N_years_PR)],
        mean_min - std_min,
        mean_min + std_min,
        alpha=0.7,
        color="tab:red",
        label=r"$\sigma_{hi_{min}}$",
    )

    # ensemble mean hi_max
    plt.plot(
        [year for year in range(N_years_PR)],
        mean_max,
        color="tab:blue",
        label=r"$hi_{max}$",
        linewidth=4,
    )
    # shadow std
    plt.fill_between(
        [year for year in range(N_years_PR)],
        mean_max - std_max,
        mean_max + std_max,
        alpha=0.7,
        color="tab:blue",
        label=r"$\sigma_{hi_{max}}$",
    )

    hi_max_diff = mean_max[0] - mean_max[-1]
    hi_min_diff = mean_min[0] - mean_min[-1]
    max_delta = max(hi_max_diff, hi_min_diff)
    min_delta = min(hi_max_diff, hi_min_diff)
    delta_seasonal = max_delta - min_delta

    plt.title(
        f"Multi-model Seasonal Variation Analysis for "
        + pr_label
        + " Scenario\n"
        + r"Seasonal Anomaly = $\Delta hi_{min} - \Delta hi_{max} =$"
        + "{:.3f}".format(delta_seasonal)
        + "m",
        size=28,
    )

    plt.legend(fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel(r"$h_i [m]$", size=29)
    plt.xlabel("Year", size=29)
    plt.grid()
    plt.savefig(save_dir + "Seas_Var_" + pr_label + ".png")
    # plt.show()
    plt.clf()

    print("---------------------------------------------------")
    print("      " + pr_label + " SEASONAL VARIATION ANALYSIS")
    print("---------------------------------------------------")
    print("hi_max[0] - hi_max[-1] = {:.3f}".format(hi_max_diff) + "m")
    print("hi_min[0] - hi_min[-1] = {:.3f}".format(hi_min_diff) + "m")
    print("Seasonal Anomaly = {:.3f}".format(delta_seasonal) + "m")
    print("---------------------------------------------------")


def subplot_sum_proj(display_single_models=False):
    """Same as summarized_projection2() but with the three subplots associated to the three PR scenarios in one figure.
    Used for ploting trend of himax and himean for different projections in order to distinguish where are the most
    important impact of sea ice lost for each projection.
    Turn display_single_models to True if you want to have see all the models results.
    pr_index = 0 for PR03, = 1 for PR06 and = 2 for PR12
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
    figure = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(
        3, 1, wspace=0.6, hspace=0.3, left=0.1, right=0.98, top=0.93, bottom=0.08
    )
    for pr_index in range(N_pr):
        mean_max = np.mean(PR_himax[pr_index], axis=1)
        # print(mean_max)
        mean_min = np.mean(PR_himin[pr_index], axis=1)
        # print(mean_min)
        std_max = np.std(PR_himax[pr_index], axis=1)
        # print(std_max)
        std_min = np.std(PR_himin[pr_index], axis=1)
        # print(std_min)

        # plt.figure()
        ### Figure pr_index ###
        ax = plt.subplot(gs[pr_index])  # row i, col 0
        # Individuals models
        if display_single_models:
            plt.plot(
                [year for year in range(N_years_PR)],
                Projections[himin][pr_index, :, :],
                alpha=0.6,
                color="tab:red",
            )
            plt.plot(
                [year for year in range(N_years_PR)],
                Projections[himax][pr_index, :, :],
                alpha=0.6,
                color="tab:blue",
            )
        # ensemble mean hi_min
        plt.plot(
            [year for year in range(N_years_PR)],
            mean_min,
            color="tab:red",
            label=r"$hi_{min}$",
            linewidth=4,
        )
        # shadow std
        plt.fill_between(
            [year for year in range(N_years_PR)],
            mean_min - std_min,
            mean_min + std_min,
            alpha=0.7,
            color="tab:red",
            label=r"$\sigma_{hi_{min}}$",
        )
        # ensemble mean hi_max
        plt.plot(
            [year for year in range(N_years_PR)],
            mean_max,
            color="tab:blue",
            label=r"$hi_{max}$",
            linewidth=4,
        )
        # shadow std
        plt.fill_between(
            [year for year in range(N_years_PR)],
            mean_max - std_max,
            mean_max + std_max,
            alpha=0.7,
            color="tab:blue",
            label=r"$\sigma_{hi_{max}}$",
        )

        hi_max_diff = mean_max[0] - mean_max[-1]
        hi_min_diff = mean_min[0] - mean_min[-1]
        max_delta = max(hi_max_diff, hi_min_diff)
        min_delta = min(hi_max_diff, hi_min_diff)
        delta_seasonal = max_delta - min_delta

        pr_label_ar = ["PR03", "PR06", "PR12"]
        pr_label = pr_label_ar[pr_index]

        print("---------------------------------------------------")
        print("      " + pr_label + " SEASONAL VARIATION ANALYSIS")
        print("---------------------------------------------------")
        print("hi_max[0] - hi_max[-1] = {:.3f}".format(hi_max_diff) + "m")
        print("hi_min[0] - hi_min[-1] = {:.3f}".format(hi_min_diff) + "m")
        print("Seasonal Anomaly = {:.3f}".format(delta_seasonal) + "m")
        print("---------------------------------------------------")

        plt.title(
            pr_label
            + " Scenario "
            + r"Seasonal Anomaly = $\Delta hi_{min} - \Delta hi_{max} =$"
            + "{:.3f}".format(delta_seasonal)
            + "m",
            size=36,
        )

        plt.legend(fontsize=30)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=32)
        plt.ylabel(r"$h_i [m]$", size=34)
        plt.grid()

    plt.xlabel("Year", size=34)
    plt.savefig(save_dir + "Subplot_Seas_Var_" + pr_label + ".png")
    # plt.show()
    plt.clf()


def subplot_sum_proj2(display_single_models=False):
    """Vertical Subplot of the summarized projections with himean, hsmax,Tsumin
    Turn display_single_models to True if you want to have see all the models results.
    pr_index = 0 for PR03, = 1 for PR06 and = 2 for PR12
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
    figure = plt.figure(figsize=(16, 24))
    gs = gridspec.GridSpec(
        3, 1, wspace=0.6, hspace=0.3, left=0.1, right=0.98, top=0.96, bottom=0.05
    )
    pr_list_name = [r"PR03", r"PR06", r"PR12"]
    pr_list_name_std = [r"$\sigma_{PR03}$", r"$\sigma_{PR06}$", r"$\sigma_{PR12}$"]
    var_list = [PR_himean, PR_hsmax, PR_Tsumin]
    var_list_name = [r"$hi_{mean}$", r"$hs_{max}$", r"$Tsu_{min}$"]
    var_list_name_and_unit = [
        r"$hi_{mean} [m]$",
        r"$hs_{max} [m]$",
        r"$Tsu_{min} [°K]$",
    ]
    color_list = ["tab:blue", "tab:orange", "tab:red"]
    label_list_mean = [r"$hi_{mean}$", r"$hs_{max}$", r"$Tsu_{min}$"]
    label_list_std = [
        r"$\sigma_{hi_{mean}}$",
        r"$\sigma_{hs_{max}}$",
        r"$\sigma_{Tsu_{min}}$",
    ]

    for fig_index in range(3):
        var = var_list[fig_index]
        ### Figure fig_index ###
        ax = plt.subplot(gs[fig_index])  # row i, col 0

        for pr_index in range(N_pr):
            mean = np.mean(var[pr_index], axis=1)
            std_mean = np.std(var[pr_index], axis=1)

            col = color_list[pr_index]
            lab_mean = label_list_mean[pr_index]
            lab_std = label_list_std[pr_index]

            # ensemble mean
            plt.plot(
                [year for year in range(N_years_PR)],
                mean,
                color=col,
                label=pr_list_name[pr_index],
                linewidth=4,
            )
            # shadow std
            if var != 1:
                plt.fill_between(
                    [year for year in range(N_years_PR)],
                    mean - std_mean,
                    mean + std_mean,
                    alpha=0.7,
                    color=col,
                    label=pr_list_name_std[pr_index],
                )
        plt.title(
            "Multi-model " + var_list_name[fig_index] + " Analysis for PR Scenarios",
            size=40,
        )

        plt.xticks(fontsize=34)
        plt.yticks(fontsize=38)
        plt.ylabel(var_list_name_and_unit[fig_index], size=38)
        plt.grid()

    plt.legend(fontsize=34, loc="upper left")
    plt.xlabel("Year", size=38)
    plt.savefig(save_dir + "Subplot_Sum_PR.png")
    # plt.show()
    plt.clf()


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


if __name__ == "__main__":
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
    """
    # hi #
    hi_mean_month = np.zeros((12, N_mod_CTL))
    for model in range(N_mod_CTL):
        hi_mean_month_mod = month_mean(hi[:, model])
        hi_mean_month[:, model] = hi_mean_month_mod
    # print(hi_mean_month)
    """
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
    ##### Plot with distinction betwsen models with and without snow #####
    plot_all_mod(data=hi, data_name="hi", N_mod=N_mod_CTL, extra_label="CTL")
    plot_all_mod(data=Tsu, data_name="Tsu", N_mod=N_mod_CTL, extra_label="CTL")
    plot_all_mod(data=hs, data_name="hs", N_mod=N_mod_CTL, extra_label="CTL")

    ########## Verification ###########
    comp_ENS_MU71()
    comp_TSIMAL_MU71()
    comp_SIGUS_MU71()
    comp_TSIMAL_ENS()
    comp_SIGUS_ENS()
    subplot_TSIMAL_SIGUS_ENS(
        data1=hs, data2=Tsu, data3=hs, N_mod=N_mod_CTL, extra_label="CTL"
    )
    ######################################## Projection Simulations Analysis #####################################
    ##### Plot of the ensemble simulations with daily resolution #####
    ### PR03 ###
    plot_all_mod(
        data=himax[:, :, 0], data_name="himax", N_mod=N_mod_PR, extra_label="PR03"
    )
    plot_all_mod(
        data=himean[:, :, 0], data_name="himean", N_mod=N_mod_PR, extra_label="PR03"
    )
    plot_all_mod(
        data=himin[:, :, 0], data_name="himin", N_mod=N_mod_PR, extra_label="PR03"
    )
    plot_all_mod(
        data=hsmax[:, :, 0], data_name="hsmax", N_mod=N_mod_PR, extra_label="PR03"
    )
    plot_all_mod(
        data=Tsumin[:, :, 0], data_name="Tsumin", N_mod=N_mod_PR, extra_label="PR03"
    )
    subplot_TSIMAL_SIGUS_ENS(
        data1=himean[:, :, 0],
        data2=Tsumin[:, :, 0],
        data3=hsmax[:, :, 0],
        N_mod=N_mod_PR,
        extra_label="PR03",
    )
    ### PR06 ###
    plot_all_mod(
        data=himax[:, :, 1], data_name="himax", N_mod=N_mod_PR, extra_label="PR06"
    )
    plot_all_mod(
        data=himean[:, :, 1], data_name="himean", N_mod=N_mod_PR, extra_label="PR06"
    )
    plot_all_mod(
        data=himin[:, :, 1], data_name="himin", N_mod=N_mod_PR, extra_label="PR06"
    )
    plot_all_mod(
        data=hsmax[:, :, 1], data_name="hsmax", N_mod=N_mod_PR, extra_label="PR06"
    )
    plot_all_mod(
        data=Tsumin[:, :, 1], data_name="Tsumin", N_mod=N_mod_PR, extra_label="PR06"
    )
    subplot_TSIMAL_SIGUS_ENS(
        data1=himean[:, :, 1],
        data2=Tsumin[:, :, 1],
        data3=hsmax[:, :, 1],
        N_mod=N_mod_PR,
        extra_label="PR06",
    )
    ### PR12 ###
    plot_all_mod(
        data=himax[:, :, 2], data_name="himax", N_mod=N_mod_PR, extra_label="PR12"
    )
    plot_all_mod(
        data=himean[:, :, 2], data_name="himean", N_mod=N_mod_PR, extra_label="PR12"
    )
    plot_all_mod(
        data=himin[:, :, 2], data_name="himin", N_mod=N_mod_PR, extra_label="PR12"
    )
    plot_all_mod(
        data=hsmax[:, :, 2], data_name="hsmax", N_mod=N_mod_PR, extra_label="PR12"
    )
    plot_all_mod(
        data=Tsumin[:, :, 2], data_name="Tsumin", N_mod=N_mod_PR, extra_label="PR12"
    )
    subplot_TSIMAL_SIGUS_ENS(
        data1=himean[:, :, 2],
        data2=Tsumin[:, :, 2],
        data3=hsmax[:, :, 2],
        N_mod=N_mod_PR,
        extra_label="PR12",
    )
    """
    ########## FINAL FIGURES ##########
    ##### Plot of the multi-model grouped by PR scenario ######
    # summarized_projection(display_single_models=False, save=True)
    ##### Plot of the multi-model grouped for CTL with DS ######
    """
    plot_all_mod2(
        data=hi,
        data_name="hi",
        N_mod=N_mod_CTL,
        extra_label="CTL",
        display_single_models=False,
        model_mean=False,
    )
    plot_all_mod2(
        data=hs,
        data_name="hs",
        N_mod=N_mod_CTL,
        extra_label="CTL",
        display_single_models=True,
        model_mean=True,
    )
    plot_all_mod2(
        data=Tsu,
        data_name="Tsu",
        N_mod=N_mod_CTL,
        extra_label="CTL",
        display_single_models=False,
        model_mean=True,
    )
    subplot_all_mod2(
        data=hi,
        data_name="hi",
        N_mod=N_mod_CTL,
        extra_label="CTL",
        display_single_models=False,
    )
    subplot_all_mod2(
        data=hi_mean_month,
        data_name="hi_mean_month",
        N_mod=N_mod_CTL,
        extra_label="CTL",
        display_single_models=False,
    )
    subplot_all_mod2(
        data=Tsu,
        data_name="Tsu",
        N_mod=N_mod_CTL,
        extra_label="CTL",
        display_single_models=False,
    )
    subplot_all_mod2(
        data=Tw,
        data_name="Tw",
        N_mod=N_mod_CTL,
        extra_label="CTL",
        display_single_models=False,
    )"""
    """summarized_projection2(pr_label="PR03", pr_index=0)
    summarized_projection2(pr_label="PR06", pr_index=1)
    summarized_projection2(pr_label="PR12", pr_index=2)
    # subplot_sum_proj()
    subplot_all_mod2(
        data=hi,
        data_name="hi",
        N_mod=N_mod_CTL,
        extra_label="CTL",
        display_single_models=False,
        dist_models=True,
    )"""
    subplot_sum_proj2()
