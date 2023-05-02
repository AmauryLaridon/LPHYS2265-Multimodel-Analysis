#!python
#!/usr/bin/env python
from scipy.io import loadmat
PR = loadmat('PR.mat')
Nmod  = PR['Nmod']
model = PR['model']
year = PR['year']
himax = PR['himax']
himin = PR['himin']
himean = PR['himean']
hsmax = PR['hsmax']
Tsumin = PR['Tsumin']
