# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:14:52 2024

@author: Juan
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib.patches import Rectangle
from puystats import linregress_w as linr

"""
Parámetros útiles
"""
tau=2043#ns
dtau=3#ns

"""
Actividad 3: Regresion lineal para respuesta osciloscopio
"""
X=np.array([20,16, 8])#Voltaje mV
Y=np.array([8.7,15, 30.4])#tiempo mu s
dX=np.ones(len(X))
dY=np.ones(len(Y))
lin_reg = linr(X,Y,dY)

f= lambda x: lin_reg.slope()*x+lin_reg.intercept()




x=np.linspace(0,22,1000)

fig, ax= plt.subplots(2,1, figsize=(6,8))

ax[0].set_title("Tiempo entre pulsos contra ancho del pulso")
ax[0].set_xlabel("$\Delta$V (mV)")
ax[0].set_ylabel("t ($\mu$s)")
ax[0].grid()
ax[0].scatter(X, Y, c="tomato")
ax[0].plot(x,f(x),c="k",linestyle="--",label="y = ("+str(round(lin_reg.slope(),1))+"$\pm$ "
         +str(round(lin_reg.err_slope(),1))+") + ("+str(round(lin_reg.intercept(),1))+"$\pm$"+
         str(round(lin_reg.err_intercept(),1)) +")")
ax[0].legend(loc="upper left")
ax[0].errorbar(X, Y, yerr= dY, xerr= dX, ecolor="k",capsize=2,linestyle="")
print("**************************")
print("Ajuste Actividad 3:")
print("m: ", lin_reg.slope())
print("Error m:",lin_reg.err_slope())
print("b: ", lin_reg.intercept())
print("Error b:",lin_reg.err_intercept())

res=((Y)-f(X))/dY



ax[1].scatter(X,res,c="red")
ax[1].grid()
ax[1].set_xlabel("$\Delta$V (mV)")
ax[1].set_ylabel("Residuales $R_i$ ")
ax[1].set_title("Gráfica de residuales normalizados")
ax[1].legend()
plt.tight_layout()

"""
Drubin-Watson
"""
Ri=Y-f(X)
Drb_Wts=np.sum((Ri[1:]-Ri[:-1])**2)/np.sum(Ri**2)



print("=========================")
print("Durbin-Watson="+str(Drb_Wts))

print("**************************")

"""
Actividad 5:
"""
"""
i) Histograma
"""

# Define the input CSV file path
input_csv_file = 'muon.csv'

# Read the CSV file
dataf = pd.read_csv(input_csv_file)


column1 = '40000'
column2 = '2'
df=dataf[(dataf[column1] < 40000) & (dataf[column1] != 10000) & (dataf[column1] != 10020)
         & (dataf[column1] != 9980) & (dataf[column1] != 4980)]
histo=np.histogram(df[column1],bins=35)

plt.figure(figsize=(8, 6))

#Histograma
frecu,lims_tau= np.histogram(df[column1], bins=35)

#puntos medios de las clases del histograma
t_m=(lims_tau[1:]-lims_tau[:-1])/2+lims_tau[:-1]
"""
ii) Ajuste de la ley y= Aexp(-t/C)+B 
para estimar la radiación de fondo
"""


def exp(x,A,B,C):
    return A*np.exp(-x/C)+B

guess=[len(df),0,tau]
parameters, covariance = curve_fit(exp,t_m, frecu,p0=guess)

fit_A,fit_B,fit_C = parameters[0],parameters[1],parameters[2]
dfit_A,dfit_B,dfit_C=np.sqrt(covariance[0][0]),np.sqrt(covariance[1][1]),np.sqrt(covariance[2][2])

t=np.linspace(10,max(t_m),1000)


#Gráfica
fig2, ax2= plt.subplots(2,1, figsize=(7,10))

ax2[0].hist(df[column1], bins=35, alpha=0.7, color='blue', edgecolor='black')
ax2[0].scatter(t_m,frecu,c="r")
ax2[0].plot(t,exp(t,fit_A,fit_B,fit_C),c="k",linestyle="--",label="y= ("+str(round(fit_A,-2))
         +"$\pm$"+str(round(dfit_A,-2))+")exp(-x/{"+str(round(fit_C,-1))
         +"$\pm$"+str(round(dfit_C,-1))+"}) + ("+str(round(fit_B,-2))
         +"$\pm$"+str(round(dfit_B,-2))+")")
ax2[0].set_title("Histograma de tiempos de decaimiento para \n 37254 llegadas con ajuste exponencial")
ax2[0].set_xlabel("tiempo (ns)")
ax2[0].set_ylabel('Número de muones N')
ax2[0].legend(loc="upper right")


res=((frecu)-exp(t_m,fit_A,fit_B,fit_C))/1



ax2[1].scatter(t_m,res,c="red")
ax2[1].grid()
ax2[1].set_xlabel("$\Delta$V (mV)")
ax2[1].set_ylabel("Residuales $R_i$ ")
ax2[1].set_title("Gráfica de residuales")
ax2[1].legend()

plt.tight_layout()
plt.show()

"""
iii) Ajuste de la ley y=Aexp(-t/C)) 
    usando regresión lineal sobre ln(y)= -t/C+ln(A)
"""
ln_N=np.log(frecu)

dN=400*np.ones(len(frecu)) #se toma la incertidumbre como la radiación de fondo

dln_N=dN/frecu
lin_reg1 = linr(t_m,ln_N,dln_N)

f1= lambda x: lin_reg1.slope()*x+lin_reg1.intercept()


x=np.linspace(min(t_m),max(t_m),1000)

fig1, ax1= plt.subplots(2,1, figsize=(6,8))

ax1[0].set_title("Logarítmo natural de N contra ancho del pulso")
ax1[0].set_xlabel("tiempo (ns)")
ax1[0].set_ylabel("ln(N)")
ax1[0].grid()
ax1[0].scatter(t_m, ln_N, c="tomato")
ax1[0].plot(x,f1(x),c="k",linestyle="--",label="y = ("+str(round(lin_reg1.slope(),5))+"$\pm$ "
         +str(round(lin_reg1.err_slope(),5))+") + ("+str(round(lin_reg1.intercept(),1))+"$\pm$"+
         str(round(lin_reg1.err_intercept(),1)) +")")
ax1[0].legend(loc="upper left")
ax1[0].errorbar(t_m, ln_N, yerr= dln_N, xerr= 1, ecolor="k",capsize=2,linestyle="")

tau_obs=-1/lin_reg1.slope()
dtau_obs=lin_reg1.err_slope()/lin_reg1.slope()**2

N0=np.exp(lin_reg1.intercept())
dN0=np.exp(lin_reg1.intercept())*lin_reg1.err_intercept()

print("**************************")
print("Ajuste Actividad 5:")
print("m: ", lin_reg1.slope())
print("Error m:",lin_reg1.err_slope())
print("b: ", lin_reg1.intercept())
print("Error b:",lin_reg1.err_intercept())
print("tau_obs: ", tau_obs)
print("Error tau_obs:",dtau_obs)
print("N0: ", N0)
print("Error N0:",dN0)

res=((ln_N)-f1(t_m))/dln_N



ax1[1].scatter(t_m,res,c="red")
ax1[1].grid()
ax1[1].set_xlabel("$tiempo (ns)")
ax1[1].set_ylabel("Residuales $R_i$ ")
ax1[1].set_title("Gráfica de residuales normalizados")
ax1[1].legend()
plt.tight_layout()

"""
Drubin-Watson
"""
Ri1=ln_N-f(t_m)
Drb_Wts1=np.sum((Ri1[1:]-Ri1[:-1])**2)/np.sum(Ri1**2)



print("=========================")
print("Durbin-Watson="+str(Drb_Wts1))

print("**************************")

"""
Histograma usando los nuevos parámetros del ajuste  

"""

fi3, ax3= plt.subplots(2,1, figsize=(7,10))

ax3[0].hist(df[column1], bins=35, alpha=0.7, color='green', edgecolor='black')
ax3[0].scatter(t_m,frecu,c="r")
ax3[0].plot(t,exp(t,N0,0,tau_obs),c="k",linestyle="--",label="y= ("+str(round(N0,-2))
         +"$\pm$"+str(round(dN0,-2))+")exp(-x/{"+str(round(tau_obs,-1))
         +"$\pm$"+str(round(dtau_obs,-1))+"})")
ax3[0].set_title("Histograma de tiempos de decaimiento para \n 37254 llegadas con ajuste exponencial")
ax3[0].set_xlabel("tiempo (ns)")
ax3[0].set_ylabel('Número de muones N')
ax3[0].legend(loc="upper right")
ax3[0].errorbar(t_m, frecu, yerr= dN, xerr= 1, ecolor="k",capsize=2,linestyle="")


res=((frecu)-exp(t_m,N0,0,tau_obs))/1



ax3[1].scatter(t_m,res,c="red")
ax3[1].grid()
ax3[1].set_xlabel("$\Delta$V (mV)")
ax3[1].set_ylabel("Residuales $R_i$ ")
ax3[1].set_title("Gráfica de residuales")
ax3[1].legend()

plt.tight_layout()
plt.show()

