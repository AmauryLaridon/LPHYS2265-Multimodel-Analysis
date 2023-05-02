#########################################################################
#
# Ice class (AGF211) by Dirk Notz
#
# Resolved by Martin Vancoppenolle
#
# Brian's (1969) model - no snow
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
rho_i       = 950          # density of ice [kg/m^3]
k_i         = 2.0          # heat conductivity of ice [W/(m K)]
alb_i       = 0.65         # bare ice albedo
rho_s       = 300.         # density of snow [kg/m^3]
k_s         = 0.3          # heat conductivity of snow [W/(m K)]
alb_s       = 0.83         # dry snow albedo
alb_w       = 0.1          # Water albedo
rho_w       = 1025.        # Water density
cw          = 4000.        # Water specific heat

eps         = 0.98
eps_sigma   = eps * 5.67e-8 # Constant in Boltzman-law
Kelvin      = 273.15       # 0C in Kelvins

# model parameters
gam_SM      = 1.400        # Conductivity Semtner factor
bet_SM      = 0.4          # Albedo Semtner factor
I_0         = 0.25         # penetration of solar radiation

# snow_fall activated or not
i_pre       = 1            # 1 for snow fall, 0 for no snowfall - does not work

# albedo parameterization
i_alb       = 2            # 0 = prescribed to alb_s, 1=read, 2=Semtner param
hs_thr_alb  = 0.1          # threshold snow depth for albedo transition in the albedo param

# Ocean heat flux
Qoce0       = 2.           # Ocean heat flux
i_Qoce      = 0            # 0=prescribed ocean heat flux, 1=depends on ice concentration
hi_thr_concentration = 0.25# ice thickness threshold at which ice concentration is 50%
f_sol       = 0.10         # fraction of solar radiation transmitted

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
h_s  = np.zeros(Nd) + 0.0  # Snow depth
alb  = np.zeros(Nd)        # Surface albedo
a_i  = np.zeros(Nd)        # Ice concentration
Qoce = np.zeros(Nd)        # Ocean heat flux

#--------------------------------------------------------------------------------------------------
# Initialize model
#--------------------------------------------------------------------------------------------------

Tsu[0] = 246.86
h_i[0] = 2.650
h_s[0] = 0.0
Tw[0]  = Tb
#h_i[0] = 0.1
#h_s[0] = 0.01

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

   #----------------------
   # Sea ice computations
   #----------------------
   if h_i[day-1] > 0.:

      h_s[day] = 0

      #--- Ice concentration ---
      zcoeff = math.log(1./2.) / hi_thr_concentration
      a_i[day] = 1.0 * ( 1. - math.exp( zcoeff * h_i[day-1] ) )

      #--- Ocean heat flux ---
      if i_Qoce==0:
         Qoce[day] = Qoce0
      else:
         Qoce[day] = Qoce0 + ( 1 - a_i[day] ) * ff.shortwave(day_of_year) * (1-alb_w)*f_sol

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
   
      # Surface temperature computation
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
      dh_i_pre   = i_pre * ff.snowfall(day_of_year)*rho_s/rho_i # Snow fall

      # Ice thickness computation
      dh_i_bot   = 1. / ( rho_i * L) * ( - Qoce[day] - Fc ) * dt  # Bottom growth / melt

      dh_i_su = -Qnet * dt / ( rho_i * L)                    # Surface melt

      h_i[day]   = h_i[day-1] + dh_i_bot + dh_i_su + dh_i_pre # Update thickness
      
      Qexcess    = - min([h_i[day],0.]) * rho_i * L / dt     # Over-used heat if total melt

      Tw[day]    = Tb + Qexcess * dt / ( rho_w * cw * h_w )  # Water temperature
   
      h_i[day]   = max([h_i[day],0.])                        # Cap thickness

   #--------------------
   # Water computations
   #--------------------
   else:

      # Fix surface temperature to 273.15
      Tsu[day]   = Kelvin

      # Fix ice albedo to water albedo
      alb[day]   = alb_w

      # Fix concentration to zero
      a_i[day]   = 0.
      Qoce[day]  = Qoce0

      # Water energy budget
      Qout_w     = -eps_sigma * Tw[day-1]**4.                        # Outgoing LW

      Qin_w      = (1-alb_w)*ff.shortwave(day_of_year) + ff.otherfluxes(day_of_year) + delta_lw # Incoming solar
   
      # Water cooling or warming
      if h_i[day] <= 0.:
         dTw     = dt / ( rho_w * cw * h_w ) * ( Qin_w + Qout_w )    # Tw increment
      else:
         dTw     = 0.
 
      Tw[day]    = max([Tw[day-1] + dTw,Tb])                         # Cap Tw

      # If water starts freezing set temperature to freezing and use excess heat to for mice
      if Tw[day] <= Tb:
         Qused      = rho_w*cw*h_w*(Tw[day-1]-Tb)/dt                 # must be > 0

         Qleft      = - ( Qin_w + Qout_w ) - Qused                   # >0

         dh_ow      = dt / (rho_i * L) * Qleft                       # freezing open water

         h_i[day]   = h_i[day-1] + dh_ow

#--------------------------------------------------------------------------------------------------
# PLOTS
#--------------------------------------------------------------------------------------------------

# Plot Tsu
fig1=plt.figure
plt.subplot(711)
plt.plot(days,Tsu-Kelvin,'g', label='Tsu', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Tsu')

# Plot h_i
plt.subplot(712)
plt.plot(days,h_i,'g', label='h_i', linewidth=2)
plt.xlabel('Day')
plt.ylabel('h_i')

# Plot h_s
plt.subplot(713)
plt.plot(days,h_s,'g', label='h_s', linewidth=2)
plt.xlabel('Day')
plt.ylabel('h_s')

# Plot Tw 
plt.subplot(714)
plt.plot(days,Tw-Kelvin,'g', label='Tw', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Tw')

# Plot albedo
plt.subplot(715)
plt.plot(days,alb,'g', label='alb', linewidth=2)
plt.xlabel('Day')
plt.ylabel('albedo')

# Plot ice concentration
plt.subplot(716)
plt.plot(days,a_i,'g', label='a_i', linewidth=2)
plt.xlabel('Day')
plt.ylabel('a_i')

# Plot ice concentration
plt.subplot(717)
plt.plot(days,Qoce,'g', label='Qoce', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Qoce (W/m2)')

plt.show()
plt.close("all")

#fig, axes = plt.subplots(1,2)
#p1=axes[0].plot(days,h_i,'r')
#axes[0].plot(range(15,365,30),td.h_i_MU, 'go')

#--------------------------------------------------------------------------------------------------
# DIAGNOSTICS
#--------------------------------------------------------------------------------------------------

# Full time series of everything
data = np.column_stack((days,Tsu, h_i, h_s, Tw))
np.savetxt('CTL_BRAILLE_ANE.txt', data, delimiter=" ", fmt='%s')

# Yearly diagnostics
h_i_max  = np.zeros(Ny)     # Yearly maximum thickness
h_s_max  = np.zeros(Ny)     # Yearly maximum thickness
h_i_min  = np.zeros(Ny)     # Yearly maximum thickness
h_i_mean = np.zeros(Ny)     # Yearly maximum thickness
T_su_min = np.zeros(Ny)

for year in range(0,Ny):
   day00 = year*365
   day99 = day00+364
   h_i_max[year] = max(h_i[day00:day99])
   h_s_max[year] = max(h_s[day00:day99])
   h_i_min[year] = min(h_i[day00:day99])
   h_i_mean[year] = np.mean(h_i[day00:day99])
   T_su_min[year] = min(Tsu[day00:day99])

data = np.column_stack((years, h_i_min, h_i_mean, h_i_max, h_s_max, T_su_min))
np.savetxt('PR12_BRAILLE_ANE.txt', data, delimiter=" ", fmt='%s')

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
