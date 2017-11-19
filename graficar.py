import csv
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import pandas as pd
import scipy
data_dir="./data/"

#Grafica los datos del canal k

trazas=[]
k=1
for t in range(10,42,2):
    with open(data_dir+'br_calib_'+str(t),'r') as data:
        reader = csv.reader(data)
        cuentas = []
        dacVal = []
        for ResComparador in reader:
            cuentas.append(float(ResComparador[k+1]))
            dacVal.append(float(ResComparador[0]))
        trazas.append(go.Scatter(
        x=dacVal,
        y=cuentas,
        mode='line',
	name='Canal '+str(k)+' temp '+str(t)
        ))
        #plt.semilogy(dacVal,cuentas)

layout = go.Layout(width=1920,height=1080,xaxis=dict(title='Valor de comparacion DAC10'),yaxis=dict(title='Cuentas',type='log',autorange=True))
#layout = go.Layout(yaxis=dict(type='log',autorange=True))
fig = go.Figure(data=trazas, layout=layout)
plotly.offline.plot(fig)
#plt.show()
#plt.savefig('sal.png',dpi=700)
