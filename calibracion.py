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
import peakutils


def erfunc(x, a, b, c, d):
    return a*erf(b*(x-c))+d

def ajustar_erf(datos,graficar=False):
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

	#Antes de comenzar el paso 1, recortamos los ultimos datos, para evitar tener ceros
	#La razon, es que el cuentapicos trabaja en escala logaritmica (para hacer mas facil la busqueda de picos)
	i=0
	while y_data[i] != 0 and i < len(y_data)-1:
		i+=1
	y_data=y_data[0:i]
	x_data=x_data[0:i]

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
	thres=0.1*max(y_data)
	try:
		while y_data[subida_base]>thres:
			subida_base+=1
		while y_data[bajada_base]>thres:
			bajada_base-=1
	except:
		print("Error al buscar ancho linea de base")
		return False
	indice10_linea_de_base = (x_data[subida_base] + x_data[bajada_base])/2

	#Buscamos el ancho de la bajada de 1SPE. La razon por la cual busco el ancho en torno al primer pico, es que el primero,
	#corresponde a la 'bajada' del rectangulo del ruido
	cotainf=indicesf[1]
	cotasup=indicesf[1]
	thres=0.1*max(y)
	try:
		while y[cotasup]>thres:
			cotasup+=1
		while y[cotainf]>thres:
			cotainf-=1
	except:
		print("Error al buscar ancho")
		return False
	#Una vez hallados los anchos, los abrimos un poco mas
	cotainf=cotainf-6
	cotasup=cotasup+6
	#Encontramos algunos casos donde el tamaño de la meseta entre ruido y 1SPE era muy pequeña, lo que hacia que la
	#deteccion de anchos sea erronea. Por ello, imponemos los siguientes limites:
	if cotainf <= indicesf[0]:
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
			if abs(np.mean(extras)) > 1:
				#La matriz de covarianza de los paramatros es muy grande, de modo que posiblemente fallo el fiteo
				if graficar:
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

#layout = go.Layout(width=1920,height=1080)

data_dir="./data/"
#Estos numeros salen de la medicion realizada
HV_BASE=32525
HV_STEP=55
cantidad_de_pasos=34

descartados=0
trazasgcalib=[]
v_br={}
#Iteramos sobre el numero de SiPM
for k in range(0,64):
	#Iteramos sobre el paso de la barrida de HV
	##Guardamos el grafico de Cuentas(NivelDeDisc) en caso que querramos verlo
	trazasg=[]
	##Guardamos ValorDeDAC1SPE(HV)
	curva_calib={}
	for j in range(0,cantidad_de_pasos):
		with open(data_dir+'br_calib_'+str(j),'r') as data:
			reader = csv.reader(data)
			datos= {}
			for ResComparador in reader:
				#Importamos la escalera de esta configuracion en el formato usual
				datos[float(ResComparador[0])]=float(ResComparador[k+1])/10**6
			#Ejecutamos la rutina de ajuste
			res=ajustar_erf(datos,True)
			if res!=False:
				params,extras,trazas,linea_de_base=res
				for traza in trazas:
					trazasg.append(traza)
				curva_calib[(HV_BASE-j*HV_STEP)*0.001812]=params[2]-linea_de_base
			else:
				descartados+=1
	#Una vez con la curva de PeakSPE(HV), realizamos un ajuste lineal para obtener el voltaje de Breakdown
	linfit = np.polyfit(np.asarray(list(curva_calib.keys())),np.asarray(list(curva_calib.values())),1)
	p = np.poly1d(linfit)
	trazasgcalib.append(go.Scatter(
			x=list(curva_calib.keys()),
			y=list(curva_calib.values()),
			mode = 'lines+markers',
			name = 'SiPM '+str(k)
			))
	trazasgcalib.append(go.Scatter(
			x=list(curva_calib.keys()),
			y=p(np.asarray(list(curva_calib.keys()))),
			mode = 'lines',
			name = 'SiPM ajuste'+str(k),
			))
	#Almacenamos V_br en un diccionario
	v_br[k]=-linfit[1]/linfit[0]


	#Descomenta para graficar Cuentas(NivelDeDisc)
	'''
	layout = go.Layout(yaxis=dict(type='log',autorange=True),width=1920,height=1080)
	plotly.offline.plot(go.Figure(data=trazasg,layout=layout))
	'''


#Descomenta para graficar ValorDeDAC1SPE(HV)

layout = go.Layout(
		xaxis=dict(title='HV (V)'),
		yaxis=dict(title='Pico 1SPE (CuentasDAC10)'),width=1920,height=1080
		)
plotly.offline.plot(go.Figure(data=trazasgcalib,layout=layout))


#Filtramos canales recortamos
for j in v_br.keys():
	if abs(v_br[j]-np.mean(list(v_br.values())))>2*np.std(list(v_br.values())):
		v_br[j]=-1
		print("CANAL FALLADO "+str(j))


#Descomenta para graficar un histograma de V_br
'''
data = [go.Histogram(x=[j for j in list(v_br.values()) if j>0],
xbins=dict(
        start=min(v_br.values()),
        end=max(v_br.values()),
        size=(max(v_br.values())-min(v_br.values()))/100
    ))]
plotly.offline.plot(data)
'''
#Antes de continuar, necesitamos elevar el voltaje al punto de operacion, que es V_br+3,5
v_op={}
for i in v_br.keys():
	v_op[i]=v_br[i]+3.5

#El valor de HV_seteamos en la fuente, va a ser el maximo. Ya que, desde ahí, bajamos con el DAC de 8bits
HV_fuente=max(v_op.values())
for k in range(0,64):
	with open('vbr.txt','a') as file:
		file.write(str(convertir_indice(k))+','+str(k)+','+str(v_br[k])+'\n')
#Escribimos la configuracion de los CITIROC
with open('salida.txt','w') as salida:
	salida.write("Voltaje de BR maximo "+str(HV_fuente)+'\n')
	salida.write("En unidades de hamamatsu "+str(HV_fuente/0.001812)+'\n')
	for p in range(1,3):
		salida.write("/*CITIROC "+str(p)+" CONF*/"+'\n')
		for l in range(0,32):
			#A que indice se corresponde?
			for s in range(0,63):
				if convertir_indice(s)==(p,l):
					break
			#Tenemos que reducir el voltaje en
			v_diff=HV_fuente-v_op[s]
			#Que tenemos que pasarla a unidades de DAC
			dacunit=240-round(v_diff*(256/2.5))
			if dacunit<0:
				dacunit=240
			salida.write("Input 8-bit DAC"+str(l)+"="+str(int(dacunit))+";"+'\n')
			salida.write("Input 8-bit DAC"+str(l)+"_ON=1;"+'\n')



#Filtramos canales recortamos
for j in v_br.keys():
	if abs(v_br[j]-np.mean(list(v_br.values())))>2*np.std(list(v_br.values())):
		v_br[j]=-1
		print("CANAL FALLADO "+str(j))


#Descomenta para graficar un histograma de V_br
'''
data = [go.Histogram(x=[j for j in list(v_br.values()) if j>0],
xbins=dict(
        start=min(v_br.values()),
        end=max(v_br.values()),
        size=(max(v_br.values())-min(v_br.values()))/100
    ))]
plotly.offline.plot(data)
'''
#Antes de continuar, necesitamos elevar el voltaje al punto de operacion, que es V_br+3,5
v_op={}
for i in v_br.keys():
	v_op[i]=v_br[i]+3.5

#El valor de HV_seteamos en la fuente, va a ser el maximo. Ya que, desde ahí, bajamos con el DAC de 8bits
HV_fuente=max(v_op.values())
for k in range(0,64):
	with open('vbr.txt','a') as file:
		file.write(str(convertir_indice(k))+','+str(k)+','+str(v_br[k])+'\n')
#Escribimos la configuracion de los CITIROC
with open('salida.txt','w') as salida:
	salida.write("Voltaje de BR maximo "+str(HV_fuente)+'\n')
	salida.write("En unidades de hamamatsu "+str(HV_fuente/0.001812)+'\n')
	for p in range(1,3):
		salida.write("/*CITIROC "+str(p)+" CONF*/"+'\n')
		for l in range(0,32):
			#A que indice se corresponde?
			for s in range(0,63):
				if convertir_indice(s)==(p,l):
					break
			#Tenemos que reducir el voltaje en
			v_diff=HV_fuente-v_op[s]
			#Que tenemos que pasarla a unidades de DAC
			dacunit=240-round(v_diff*(256/2.5))
			if dacunit<0:
				dacunit=240
			salida.write("Input 8-bit DAC"+str(l)+"="+str(int(dacunit))+";"+'\n')
			salida.write("Input 8-bit DAC"+str(l)+"_ON=1;"+'\n')
