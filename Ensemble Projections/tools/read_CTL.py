#!python
#!/usr/bin/env python
from scipy.io import loadmat
CTL = loadmat('CTL.mat')
hi = CTL['hi']
hs = CTL['hs']
Tsu = CTL['Tsu']
Tw = CTL['Tw']
doy = CTL['doy']
model = CTL['model']
# one-liner to read a single variable
