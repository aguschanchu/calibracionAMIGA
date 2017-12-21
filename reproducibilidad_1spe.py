import csv
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import pandas as pd
from scipy.special import erf
from scipy.optimize import curve_fit
import time
import warnings
from scipy.optimize import OptimizeWarning
from scipy.stats import linregress
import peakutils


def erfunc(x, a, b, c, d):
    return a*erf(b*(x-c))+d

def ajustar_erf(datos,graficar=False,debug=True):
	'''
	Dado un diccionario con el formato {dacVal:cuentas}, ajusta una erf a la bajada de 1SPE y devuelve los parametros del ajuste.
	El mismo, lo hace mediante la siguiente rutina:
	1) Realiza un smooth de los datos mediante el metodo Savitzky Golay, y deriva el resultado.
	Sobre este, aplica la funcion findpeaks para encontrar el maximo (y por lo tanto,
	el punto de inflexion de los datos originales). Ademas, calcula un ancho aproximado de esta bajada
	2) A partir del punto del maximo y del ancho calculados, recorta los datos originales (sin smooth)
	a los que le ajusta una funcion error
	FYI: El punto de inflexion de erfunc se alcanza en x=c
	'''
	x_data=list(datos.keys())
	y_data=list(datos.values())

	#plt.semilogy(x_data,y_data)
	#plt.show()

	#Antes de comenzar el paso 1, recortamos los ultimos datos, para evitar tener ceros
	#La razon, es que el cuentapicos trabaja en escala logaritmica (para hacer mas facil la busqueda de picos)
	x_data=[i for i in x_data if y_data[x_data.index(i)]>0]
	y_data=[i for i in y_data if i>0]
	if len(y_data)<20:
		print("Unicamente hay linea de base")
		return False


	#Comenzamos con el paso 1
	y=savitzky_golay(-np.log(np.asarray(y_data)), window_size=7, order=2,deriv=1)
	#Buscamos picos dentro del smooth
	indices = peakutils.indexes(y,min_dist=len(y_data)//25,thres=0.4)
	#Filtramos los picos hallados
	indicesf=[]
	for l in indices:
		if y[l] > 0.2*max(y):
			indicesf.append(l)

	#Vamos a necesitar tambien buscar el valor de DAC10 de la linea de base. Para ello, me fijo cuando y cae por debajo de la mitad
	bajada_base=y_data.index(max(y_data))
	subida_base=y_data.index(max(y_data))
	thres=0.2*max(y_data)
	try:
		while y_data[subida_base]>thres:
			subida_base+=1
		while y_data[bajada_base]>thres:
			bajada_base-=1
	except:
		print("Error al buscar ancho linea de base")
		if debug:
			plt.semilogy(x_data,y_data)
			plt.show()
		return False
	indice10_linea_de_base = (x_data[subida_base] + x_data[bajada_base])/2

	#Buscamos el ancho de la bajada de 1SPE. La razon por la cual busco el ancho en torno al primer pico, es que el primero,
	#corresponde a la 'bajada' del rectangulo del ruido

	##Busco los de inflexion contiguos al pico, correspondientes al doblamiento en el Plateua
	indicesl = peakutils.indexes(-y,min_dist=len(y_data)//25,thres=0.1)
	try:
		for j in range(0,len(indicesl)):
			if indicesl[j] < indicesf[1] and indicesl[j+1] > indicesf[1]:
				cotasup=indicesl[j+1]
				cotainf=indicesl[j]
				break
	except:
		print("Error al buscar ancho linea de base")
		if debug:
			plt.semilogy(x_data,y_data)
			plt.show()
			plt.plot(x_data,-y)
			plt.plot([x_data[j] for j in indicesl],[-y_data[j] for j in indicesl])
			plt.show()
		return False

	#Una vez hallados los anchos, los abrimos un poco mas
	cotainf=cotainf-4
	cotasup=cotasup+4
	#Encontramos algunos casos donde el tamaño de la meseta entre ruido y 1SPE era muy pequeña, lo que hacia que la
	#deteccion de anchos sea erronea. Por ello, imponemos los siguientes limites:
	if cotainf <= indicesf[0]+3:
		cotainf=indicesf[0]+3

	if graficar:
		trazas=[]
		trazas.append(go.Scatter(
					x=x_data,
					y=y_data,
					line = dict(color = ('rgb(204, 0, 0)')),
					name='Cuentas'
					))
		trazas.append(go.Scatter(
			x=[x_data[j] for j in indicesf],
			y=[y_data[j] for j in indicesf],
			mode='markers',
			marker=dict(
				size=8,
				color='rgb(200,100,0)',
				symbol='cross'
			),
			name='Picos'
			))
		trazas.append(go.Scatter(
			x=[x_data[j] for j in [bajada_base,subida_base]],
			y=[y_data[j] for j in [bajada_base,subida_base]],
			mode='markers',
			marker=dict(
				size=8,
				color='rgb(250,250,250)',
				symbol='cross'
			),
			name='Exteremos linea de base'
			))
		trazas.append(go.Scatter(
			x=[x_data[cotainf],x_data[cotasup]],
			y=[y_data[cotainf],y_data[cotasup]],
			mode='markers',
			marker=dict(
				size=8,
				color='rgb(0,0,255)',
				symbol='cross'
			),
			name='Limites de ajuste'
			))

	#Recortamos los datos en funcion a lo hallado anteriormente. Unicamente nos interesa el punto de 1SPE, no el resto de la escalera
	x_data=x_data[cotainf:cotasup+1]
	y_data=y_data[cotainf:cotasup+1]

	#Inicializamos valores de ajute de acuerdo a sugerencia hallada en https://mathematica.stackexchange.com/a/42166
	p0=[-(max(y_data)-min(y_data))/2,2/len(x_data),np.mean(x_data),(max(y_data)-min(y_data))/2]

	#Fiteamos una erf. La razon del with es para que el fiteo levante un error, si tiene problemas para ajustar
	with warnings.catch_warnings():
		warnings.simplefilter("error", OptimizeWarning)
		try:
			params, extras = curve_fit(erfunc, x_data, y_data, p0,ftol=1.49012e-14, xtol=1.49012e-14, maxfev=10**8)
			#Es bueno el fiteo? Calculemos r^2
			residuals = y_data - erfunc(x_data, *params)
			ss_res = np.sum(residuals**2)
			r_squared = 1 - ss_res/np.sum((y_data-np.mean(y_data))**2)
			if r_squared < 0.95:
				print("Fiteo malo")
				print(r_squared)
				if debug:
					plt.semilogy(x_data,y_data)
					plt.semilogy(x_data,erfunc(x_data, *params))
					plt.show()
				return False
		except OptimizeWarning:
			print("Error en fiteo")
			return False


	if graficar:
		y_adj=[]
		for x in x_data:
			y_adj.append(erfunc(x,*params))
		trazas.append(go.Scatter(
			x=x_data,
			y=y_adj,
			line = dict(color = ('rgb(0, 250, 0)')),
			name='Ajuste erf'
			))
		return params,extras,trazas,indice10_linea_de_base
	else:
		return params,extras,indice10_linea_de_base


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savicotatzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    """
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def convertir_indice(k):
	'''
	El programa utiliza otra indexacion de los SiPM, que la indexacion dada el archivo de configuracion del CITIROC.
	Por tal motivo, a partir de la regla que nos paso Mati, hacemos la conversion.
	Devuelve una tupla (i,j) donde i es el numero de CITIROC, y j el del SiPM
	'''
	if (k//4)%2 == 0:
		i=2
	else:
		i=1
	#Es la secuencia A244158 de OEIS. No, chiste
	orden_orig=[[20,21,22,23,16,17,18,19,0,1,2,3,4,5,6,7,15,14,13,12,11,10,9,8,27,26,25,24,31,30,29,28],[7,6,5,4,3,2,1,0,19,18,17,16,23,22,21,20,28,29,30,31,24,25,26,27,8,9,10,11,12,13,14,15]]
	#Lo verifique. A mano. Esta bien, posta
	j=orden_orig[(k//4)%2][((k//4)//2)*4+k%4]
	return i,j

data_dir="C:/Users/Agustin/Dropbox/Exactas/ITeDA/AMIGA/Temperatura fija caracterizacion 1SPE/"
#data_dir="C:/Users/Agustin/Dropbox/Exactas/ITeDA/AMIGA/Calibracion DAC8/data/"

cantidad_de_pasos=100

descartados=0
trazasgcalib=[]
picos=[]
v_br={}
#Iteramos sobre el numero de SiPM
for k in range(0,64):
	picos.append([])
	#Iteramos sobre el paso de la barrida de HV
	##Guardamos el grafico de Cuentas(NivelDeDisc) en caso que querramos verlo
	trazasg=[]
	##Guardamos ValorDeDAC1SPE(HV)
	curva_calib={}
	for j in range(0,cantidad_de_pasos):
		with open(data_dir+'br_calib_T25_num'+str(j),'r') as data:
			reader = csv.reader(data)
			datos= {}
			for ResComparador in reader:
				#Importamos la escalera de esta configuracion en el formato usual
				datos[float(ResComparador[0])]=float(ResComparador[k+1])/10**6
			#Ejecutamos la rutina de ajuste
			try:
				res=ajustar_erf(datos,True,False)
			except:
				traceback.print_exc()
				res=False
			if res!=False:
				params,extras,trazas,linea_de_base=res
				for traza in trazas:
					trazasg.append(traza)
				picos[k].append(params[2]-linea_de_base)
			else:
				descartados+=1
	#Descomenta para graficar Cuentas(NivelDeDisc)
	'''
	layout = go.Layout(yaxis=dict(type='log',autorange=True),width=1920,height=1080)
	plotly.offline.plot(go.Figure(data=trazasg,layout=layout))
	'''


#Descomenta para graficar un histograma de V_br
data= []
datos = []
for k in range(0,1):
	if k not in [5,6]:
		for j in picos[k]:
			datos.append(j)
data.append(go.Histogram(x=[j for j in list(datos) if j>0],
xbins=dict(
        start=22.5,
        end=27.5,
		size=(max(datos)-min(datos))/50),
	    opacity=0.75,
		histnorm='probability'
    ))
layout = go.Layout(barmode='stack')
plotly.offline.plot(go.Figure(data=data, layout=layout))
