import csv
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import pandas as pd
import scipy
data_dir="C:/Users/Agustin/Dropbox/Exactas/ITeDA/AMIGA/Barrido en temperaturas con correccion HV/Corrida2/"

#Grafica los datos del canal k

trazas=[]
for k in range(0,64):
	for t in range(0,1,1):
	    with open(data_dir+'br_calib_NOS_24','r') as data:
	        reader = csv.reader(data)
	        cuentas = []
	        dacVal = []
	        for ResComparador in reader:
	            cuentas.append(float(ResComparador[k+1])/60)
	            dacVal.append(float(ResComparador[0]))
	        trazas.append(go.Scatter(
	        x=dacVal,
	        y=cuentas,
	        mode='line',
		name='Canal '+str(k)))
	        #plt.semilogy(dacVal,cuentas)

layout = go.Layout(
			xaxis=dict(title='Valor de comparacion DAC10',titlefont=dict(
            size=30,
            color='#000000'
        ),tickfont=dict(
            size=24,
            color='black'
        )
		),
			yaxis=dict(title='Cuentas [Hz]',titlefont=dict(
            size=30,
            color='#000000'),
			type='log'
        ,tickfont=dict(
            size=24,
            color='black'
        ))
			)
#layout = go.Layout(yaxis=dict(type='log',autorange=True))
fig = go.Figure(data=trazas, layout=layout)
plotly.offline.plot(fig)
#plt.show()
#plt.savefig('sal.png',dpi=700)
