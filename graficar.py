import csv
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import pandas as pd
import scipy
data_dir="./"

#Grafica los datos del canal k

trazas=[]
for k in range(0,64):
    with open('br_calibrado,'r') as data:
        reader = csv.reader(data)
        cuentas = []
        dacVal = []
        for ResComparador in reader:
            cuentas.append(float(ResComparador[k+1]))
            dacVal.append(float(ResComparador[0]))
        trazas.append(go.Scatter(
        x=dacVal,
        y=cuentas,
        mode='line'
        ))
        #plt.semilogy(dacVal,cuentas)

#layout = go.Layout(yaxis=dict(type='log',autorange=True),width=1920,height=1080)
layout = go.Layout(yaxis=dict(type='log',autorange=True))
fig = go.Figure(data=trazas, layout=layout)
plotly.offline.plot(fig)
#plt.show()
#plt.savefig('sal.png',dpi=700)
