#########################################################################
#
# Ice class (AGF211) by Dirk Notz
#
# Resolved by Martin Vancoppenolle
#
# Rewritten Semtner 0-layer - with ice concentration equation on top of it
#
# and an ocean heat flux that depends on ice concentration
#
# 21 January 2016
#
#########################################################################

import matplotlib.pyplot as plt
import numpy as np
import flux_fit as ff
import math as math
import thickness_data as td

plt.close("all")

# Numerical Parameters
sec_per_day = 86400        # seconds per day
day_per_yr  = 365          # days per year
Ny = 100                   # Number of years
Nd = Ny*day_per_yr         # Number of days
Nit = 20                   # Number of iterations for surface temperature
dt  = 86400.               # Time step

# physical constants 
L           = 334000       # Latent heat of freezing for water [J/kg]
rho_i       = 900          # density of ice [kg/m^3]
k_i         = 2.2          # heat conductivity of ice [W/(m K)]
alb_i       = 0.64         # bare ice albedo
rho_s       = 330.         # density of snow [kg/m^3]
k_s         = 0.3          # heat conductivity of snow [W/(m K)]
alb_s       = 0.83         # dry snow albedo
alb_w       = 0.06         # Water albedo
rho_w       = 1025.        # Water density
cw          = 4000.        # Water specific heat

eps_i       = 0.98
eps_w       = 0.95
eps_sigma   = eps_i * 5.67e-8 # Constant in Boltzman-law
eps_sigma_w = eps_w * 5.67e-8 # Constant in Boltzman-law
Kelvin      = 273.15       # 0C in Kelvins

# model parameters
gam_SM      = 1.500        # Conductivity Semtner factor (1.065 for semtner)
bet_SM      = 0.4          # Albedo Semtner factor
I_0         = 0.25         # penetration of solar radiation

# snow_fall activated or not
i_pre       = 1            # 1 for snow fall, 0 for no snowfall - does not work

# albedo parameterization
i_alb       = 2            # 0 = prescribed to alb_s, 1=read, 2=Semtner param
hs_thr_alb  = 0.1          # threshold snow depth for albedo transition in the albedo param

# active concentration
i_hibler    = 1            # Activate concentration
f_lam       = 1.           # Efficiency of lateral melting
hzero       = 0.01         # New ice thickness in leads

# Ocean heat flux parameterization
Qoce0       = 5.           # Ocean heat flux
i_Qoce      = 1            # 0=prescribed ocean heat flux, 1=depends on ice concentration
f_sol       = 0.20         # fraction of solar radiation transmitted (Maykut and McPhee)

# Longwave perturbation
delta_lw_rcp = 12
i_delta_lw   = 1           # 0=step perturbation / 1=100-yr perturbation

# Initial-boundary conditions
Tb    = -1.8 + Kelvin      # Basal temperature
h_w   = 30.                # Mixed layer depth

# generate 365 days array
days = np.arange(Nd)
years= np.arange(Ny)
Tsu  = np.zeros(Nd)        # Surface Temperature
Tw   = np.zeros(Nd)        # Water temperature
h_i  = np.zeros(Nd)        # Ice thickness
h_s  = np.zeros(Nd)        # Snow depth
a_i  = np.zeros(Nd)        # Ice concentration
v_i  = np.zeros(Nd)        # Ice volume
alb  = np.zeros(Nd)        # Surface albedo
Qoce = np.zeros(Nd)        # Ocean heat flux

#--------------------------------------------------------------------------------------------------
# Initialize model
#--------------------------------------------------------------------------------------------------

Tsu[0] = 244.53
v_i[0] = 0.5
h_s[0] = 0.0
v_i[0] = 2.608
h_s[0] = 0.306
Tw[0]  = Tb
a_i[0] = 1.
h_i[0] = v_i[0] / a_i[0]

#--------------------------------------------------------------------------------------------------
# Main model loop
#--------------------------------------------------------------------------------------------------
for day in range(1,Nd):

   #----------
   # Calendar
   #----------
   day_of_year = np.mod(day,365)

   year = np.modf(day/365)[1]

   #-----------------------
   # Longwave perturbation
   #-----------------------
   if i_delta_lw==1: # gradual (over 100 years)
      delta_lw = delta_lw_rcp * year / 100.
   else:             # step (instantaneous)
      delta_lw = delta_lw_rcp 

   if day_of_year<1:
      print('year:', year, ' delta_lw:', delta_lw)

   #--------------------
   # Water computations
   #--------------------
   # Water energy budget
   Qout_w     = -eps_sigma_w * Tw[day-1]**4.                      # Outgoing LW

   Qin_w      = (1-alb_w)*ff.shortwave(day_of_year) + \
                ff.otherfluxes(day_of_year) + delta_lw            # Incoming SW

   if h_i[day-1] <= 0.:

      # Fix surface temperature to 273.15
      Tsu[day]   = Kelvin

      # Fix ice albedo to water albedo
      alb[day]   = alb_w

      # Fix Ocean heat flux to prescribed value
      Qoce[day]  = Qoce0

      # Fix ice concentration to zero
      a_i[day-1] = 0.
   
      # Water cooling or warming
      dTw     = dt / ( rho_w * cw * h_w ) * ( Qin_w + Qout_w )    # Tw increment

      Tw[day] = max([Tw[day-1] + dTw,Tb])                        # Cap Tw

   # If water starts freezing set temperature to freezing and use excess heat for melting
   if Tw[day] <= Tb: 

      Qused      = rho_w*cw*h_w*(Tw[day-1]-Tb)/dt                 # must be > 0

      Qleft      = - ( Qin_w + Qout_w ) * ( 1. - a_i[day-1] ) - Qused  # >0

      dv_ow      = max([dt / (rho_i * L) * Qleft,0.])             # freezing open water

      if i_hibler == 0:
         da_ow = 0.
         a_i[day-1] = 1.
      else:
         da_ow      = min([1-a_i[day-1],dv_ow/hzero])
         a_i[day-1] = a_i[day-1] + da_ow

      v_i[day-1] = v_i[day-1] + dv_ow

      h_i[day-1] = v_i[day-1] / a_i[day-1]

   #----------------------
   # Sea ice computations
   #----------------------
   if v_i[day-1] > 0.:

      h_i[day-1] = v_i[day-1] / a_i[day-1]

      #--- Ocean heat flux ---
      if i_Qoce==0:
         Qoce[day] = Qoce0
      else:
         Qoce[day] = Qoce0 + ( 1 - a_i[day-1] ) * ff.shortwave(day_of_year) * (1-alb_w)*f_sol

      #--- Albedo ---
      # Prescribed albedo
      if i_alb==0:
         alb[day] = 0.8 # Dirk's prescribed value

      # Functional albedo
      if i_alb==1:
         alb[day] = ff.albedo(day_of_year)

      # Semtner-like albedo
      if i_alb==2:
         zalpha   = min(h_s[day-1],hs_thr_alb) / hs_thr_alb
         alb[day] = alb_s * zalpha + alb_i * ( 1. - zalpha )

      # Semtner trick to further increase albedo
      alb[day] = alb[day] + bet_SM * ( 1 - alb[day] ) * I_0

      # Downwelling atmospheric flux
      Qa = (1-alb[day])*ff.shortwave(day_of_year) + ff.otherfluxes(day_of_year) + delta_lw
      
      # Conductivity times_thickness
      Keff_h     = k_i * k_s / ( k_i * h_s[day-1] + k_s * h_i[day-1] ) * gam_SM
   
      # Surface temperature computation (Newton-Raphson method)
      zTsu = Tsu[day-1]
      for i_iter in range(1,Nit):

         Fc         = Keff_h * ( zTsu - Tb )                 # Conductive heat flux

         Qnet       = Qa - eps_sigma * zTsu**4. - Fc         # Net flux

         dQnet_dTsu = -4.*eps_sigma * zTsu**3 - Keff_h       # Net flux derivative

         delta_Tsu  = - Qnet / dQnet_dTsu                    # Tsu change

         zTsu       = zTsu + delta_Tsu                       # Dummy Tsu
   
      # Surface temperature computation
      Tsu[day]    = min([zTsu,Kelvin])                       # Cap Tsu

      # Heat available for surface ablation
      Fc         = Keff_h * ( Tsu[day] - Tb )                # Conductive heat flux
   
      Qnet       = Qa - eps_sigma * Tsu[day]**4. - Fc        # Net atmospheric heat flux

      # Snow depth computation
      dh_s_pre   = i_pre * ff.snowfall(day_of_year)          # Snow fall

      dh_s_mel   = -Qnet * dt / ( rho_s * L )                # Snow melting

      h_s[day]   = h_s[day-1] + dh_s_pre + dh_s_mel          # Update snow depth

      Qexcess    = - min([h_s[day],0.]) * rho_s * L / dt     # Over-used heat (W/m2)

      dh_i_exc   = - Qexcess * dt / ( rho_i * L )            # Extra ice melting

      h_s[day]   = max([h_s[day],0.])
   
      # Ice thickness computation
      dh_i_bot   = 1. / ( rho_i * L) * ( - Qoce[day] - Fc ) * dt  # Bottom growth / melt

      if h_s[day-1] > 0.:
         dh_i_su = dh_i_exc                                  # Surface melt if snow left
      else:
         dh_i_su = -Qnet * dt / ( rho_i * L)                 # Surface melt

      h_i[day]   = h_i[day-1] + dh_i_bot + dh_i_su           # Update thickness

      dh_i_m_tot = min([ dh_i_bot + dh_i_su , 0. ])          # Total melt

      Qexcess    = - min([h_i[day],0.]) * rho_i * L / dt     # Over-used heat if total melt

      Tw[day]    = Tb + Qexcess * dt / ( rho_w * cw * h_w )  # Water temperature
   
      h_i[day]   = max([h_i[day],0.])                        # Cap thickness

      # Lateral melting
      if i_hibler == 0: # SMOL case
         if ( h_i[day] == 0.):
            a_i[day] = 0.
            v_i[day] = 0.
         else:
            a_i[day] = 1.
            v_i[day] = a_i[day] * h_i[day]

      else:             # HIBLER CASE
         if h_i[day-1] > 0.:
            da_lam     = f_lam* max([-a_i[day-1],a_i[day-1]/( 2 *h_i[day-1])])*dh_i_m_tot

         a_i[day]    = a_i[day-1] + da_lam
         v_i[day]    = a_i[day-1] * h_i[day]
         h_i[day]    = v_i[day] / a_i[day]

#--------------------------------------------------------------------------------------------------
# PLOTS
#--------------------------------------------------------------------------------------------------

# Plot Tsu
fig1=plt.figure
plt.subplot(811)
plt.plot(days,Tsu-Kelvin,'g', label='Tsu', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Tsu')

# Plot h_i
plt.subplot(812)
plt.plot(days,h_i,'g', label='h_i', linewidth=2)
plt.xlabel('Day')
plt.ylabel('h_i')

# Plot h_s
plt.subplot(813)
plt.plot(days,h_s,'g', label='h_s', linewidth=2)
plt.xlabel('Day')
plt.ylabel('h_s')

# Plot Tw 
plt.subplot(814)
plt.plot(days,Tw-Kelvin,'g', label='Tw', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Tw')

# Plot albedo
plt.subplot(815)
plt.plot(days,alb,'g', label='alb', linewidth=2)
plt.xlabel('Day')
plt.ylabel('albedo')

# Plot albedo
plt.subplot(816)
plt.plot(days,a_i,'g', label='a_i', linewidth=2)
plt.xlabel('Day')
plt.ylabel('A')

# Plot albedo
plt.subplot(817)
plt.plot(days,Qoce,'g', label='Qoce', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Qoce')

# Plot albedo
plt.subplot(818)
plt.plot(days,v_i,'g', label='v_i', linewidth=2)
plt.xlabel('Day')
plt.ylabel('v_i')

plt.show()

#fig, axes = plt.subplots(1,2)
#p1=axes[0].plot(days,h_i,'r')
#axes[0].plot(range(15,365,30),td.h_i_MU, 'go')

#--------------------------------------------------------------------------------------------------
# DIAGNOSTICS
#--------------------------------------------------------------------------------------------------

# Full time series of everything
data = np.column_stack((days,Tsu, v_i, h_s, Tw))
#np.savetxt('CTL_YBLAIRE_ICE.txt', data, delimiter=" ", fmt='%s')

# Yearly diagnostics
h_i_max  = np.zeros(Ny)     # Yearly maximum thickness
h_s_max  = np.zeros(Ny)     # Yearly maximum thickness
h_i_min  = np.zeros(Ny)     # Yearly maximum thickness
h_i_mean = np.zeros(Ny)     # Yearly maximum thickness
T_su_min = np.zeros(Ny)

for year in range(0,Ny):
   day00 = year*365
   day99 = day00+364
   h_i_max[year] = max(v_i[day00:day99])
   h_s_max[year] = max(h_s[day00:day99])
   h_i_min[year] = min(v_i[day00:day99])
   h_i_mean[year] = np.mean(v_i[day00:day99])
   T_su_min[year] = min(Tsu[day00:day99])

data = np.column_stack((years, h_i_min, h_i_mean, h_i_max, h_s_max, T_su_min))
np.savetxt('PR12_YBLAIRE.txt', data, delimiter=" ", fmt='%s')

# Figure 2
fig2=plt.figure()
ax1=fig2.add_subplot(211)
plt.plot(h_i_max)
plt.plot(h_i_min)
plt.plot(h_i_mean)
plt.xlabel('Year')
plt.ylabel('h_i (m)')

ax1=fig2.add_subplot(212)
plt.plot(T_su_min-Kelvin)
plt.xlabel('Year')
plt.ylabel('Tsu (degC)')

plt.show()
