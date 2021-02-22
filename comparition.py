# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:00:17 2021

@author: f.aguilar.santos
"""

#obtenemos el fichero para crear el df
import pandas as pd
import numpy as np
import re
import math
import matplotlib.pyplot as plt

##Leemos el csv y creamos el df

raw_data = "./players19-20.v.2.0.csv"

df = pd.read_csv(raw_data, sep=";")

#Limpiamos los datos del dataset con basura
#Limpiamos los datos que tienen %

df['% de penaltis marcados90min'] = df['% de penaltis marcados90min'].apply(lambda x: x.replace("%",""))
df['% de penaltis marcados90min'] = df['% de penaltis marcados90min'].apply(lambda x: float(x[0:4]))

df['% de efectividad de pases90min'] = df['% de efectividad de pases90min'].apply(lambda x: x.replace("%",""))
df['% de efectividad de pases90min'] = df['% de efectividad de pases90min'].apply(lambda x: float(x[0:4]))

df['% de efectividad de los centros90min'] = df['% de efectividad de los centros90min'].apply(lambda x: x.replace("%",""))
df['% de efectividad de los centros90min'] = df['% de efectividad de los centros90min'].apply(lambda x: float(x[0:4]))

df['% disputas ganadas90min'] = df['% disputas ganadas90min'].apply(lambda x: x.replace("%",""))
df['% disputas ganadas90min'] = df['% disputas ganadas90min'].apply(lambda x: float(x[0:4]))

df['% disputas por arriba ganadas90min'] = df['% disputas por arriba ganadas90min'].apply(lambda x: x.replace("%",""))
df['% disputas por arriba ganadas90min'] = df['% disputas por arriba ganadas90min'].apply(lambda x: float(x[0:4]))
#Eliminamos la columna dorsal
df=df.drop(['Dorsal_x'], axis=1)

##Elegimos el jugador que queremos comparar

jugador = "Cristiano Ronaldo"

df_jugador = df.loc[df["Jugador"] == jugador]

#obtengo laposición en la que juega el jugador seleccionado
pos = df_jugador.iloc[0]["Posicion"]

#Dependiendo el jugador que se quiera comparar se tendrán en cuenta unas variables u otras.
#Para ello cogemos los valores por 90 minutos
colsDef = ['Balones recuperados90min','Recuperaciones en campo rival90min','Disputas90min','% disputas ganadas90min',
           'Disputas aéreas90min','% disputas por arriba ganadas90min','Entradas90min','Interceptaciones90min',
           'Rechaces90min']

colsMc = ['Goles90Min','Asistencias90min','Tiros90min','Tiros a portería90min','Pases90min',
          '% de efectividad de pases90min','Pases de finalización90min','Centros90min',
          '% de efectividad de los centros90min','Balones perdidos90min','Pérdidas en campo propio90min',
           'Balones recuperados90min','Recuperaciones en campo rival90min','Goles esperados90min',
           'Disputas90min','% disputas ganadas90min','Disputas en ataque90min','Disputas aéreas90min',
           '% disputas por arriba ganadas90min','Regates90min','Entradas90min','Interceptaciones90min','Rechaces90min']

colsDel = ['Goles90Min','Asistencias90min','Tiros90min','Tiros a portería90min','% de penaltis marcados90min',
           'Pases90min','% de efectividad de pases90min','Pases de finalización90min','Centros90min',
           '% de efectividad de los centros90min','Balones perdidos90min','Pérdidas en campo propio90min',
           'Balones recuperados90min','Recuperaciones en campo rival90min','Goles esperados90min','Disputas90min',
           'Disputas en ataque90min','Regates90min']

viewcolsDef = ['Balones recuperados90min','Recuperaciones en campo rival90min','Disputas90min',
               'Disputas aéreas90min','Entradas90min','Interceptaciones90min','Rechaces90min']

viewcolsMc = ['Goles90Min','Asistencias90min','Tiros a portería90min','Pases90min',
          'Pases de finalización90min','Balones perdidos90min',
           'Balones recuperados90min','Regates90min','Interceptaciones90min']

viewcolsDel = ['Goles90Min','Asistencias90min','Tiros90min','Tiros a portería90min',
               'Pases de finalización90min','Balones perdidos90min','Goles esperados90min','Regates90min']

def select_cols(pos):
    if (pos=="DEF"):
        cols = colsDef
    elif(pos=="MC"):
        cols = colsMc
    elif(pos=="DEL"):
        cols = colsDel
   
    return cols

def select_viewcols(pos):
    if (pos=="DEF"):
        cols = viewcolsDef
    elif(pos=="MC"):
        cols = viewcolsMc
    elif(pos=="DEL"):
        cols = viewcolsDel
   
    return cols

#Creamos la función para obtener una distancia casera
def euclidean_distance(row):
    """
    A simple euclidean distance function
    """
    cols = select_cols(pos)
    inner_value = 0
    for k in cols:
        inner_value += (row[k] - df_jugador[k]) ** 2
    return math.sqrt(inner_value)

#Encontramos la distancia de cada jugador con el seleccionado
distance = df.apply(euclidean_distance, axis=1)

#obtenemos un dataframe solo con las columnas seleccionadas
if (pos=="DEF"):
        df_compare = df[colsDef]
        cols = colsDef
elif(pos=="MC"):
        df_compare = df[colsMc]
        cols = colsMc
elif(pos=="DEL"):
        df_compare = df[colsDel]  
        cols = colsDel

#Normalizamos

df_normalized = (df_compare - df_compare.mean()) / df_compare.std()

#KNN
from scipy.spatial import distance

jugador_normalizado = df_normalized[df["Jugador"] == jugador]

#Encontramos la distancia entre el jugador y todos los demás

euclidean_distances = df_normalized.apply(lambda row: distance.euclidean(row, jugador_normalizado), axis=1)

#Creamos un df con las distancias

distance_frame = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
distance_frame.sort_values("dist", inplace=True)

#Obtenemos el jugador más parecido
values = np.array([])
for k in distance_frame["idx"]:
    values = np.append(values, k)
    
cont = 1;
jugadores_similares = np.array([])
index_jugadores_similares = np.array([])

for k in values[1:6]:
    second_smallest = distance_frame.iloc[cont]["idx"]
    jugador_similar = df.loc[int(second_smallest)]["Jugador"]
    
    jugadores_similares = np.append(jugadores_similares,jugador_similar)
    index_jugadores_similares = np.append(index_jugadores_similares,distance_frame.iloc[cont]["idx"])
    cont = cont + 1;
    

#obtenemos el df del jugador a comparar
cols = select_viewcols(pos)#Obtengo las columnas que se quieren representar 
df_jugador_compare = df_jugador[cols]

#obtenemos el df del jugador más similar
jugador_similar = jugadores_similares[0]

df_jsimilar = df.loc[df["Jugador"] == jugador_similar]

df_jsimilar = df_jsimilar[cols]

#Obtenemos los datos del jugador a comparar y el jugador similar
datos_jugador = df.loc[df["Jugador"] == jugador]

jugador_similar = jugadores_similares[0]
datos_jsimilar = df.loc[df["Jugador"] == jugador_similar]

##################Representacion polar###########################
index_jugador = df_jugador.iloc[0].name
index_jugador_similar = index_jugadores_similares[0]

values_jugador = df_jugador_compare.loc[index_jugador, :].values.tolist()
values_jugador += values_jugador [:1]
values_jsimilar = df_jsimilar.loc[index_jugador_similar, :].values.tolist()
values_jsimilar += values_jsimilar [:1]
atributos =list(df_jugador_compare)
#ranges = [(0.1, 2.3), (1.5, 0.3), (1.3, 0.5),
         #(1.7, 4.5), (1.5, 3.7), (70, 87), (100, 10)]  
   
angles = [n / len(cols) * 2 * math.pi for n in range(len(cols))]
angles += angles [:1]
    
angles2 = [n / len(cols) * 2 * math.pi for n in range(len(cols))]
angles2 += angles2 [:1]
    
#ax = plt.subplot(111, polar=True)

#Create the chart as before, but with both Ronaldo's and Messi's angles/values
ax = plt.subplot(111, polar=True)
plt.xticks(angles[:-1],atributos)
ax.plot(angles,values_jugador)
ax.fill(angles, values_jugador, 'teal', alpha=0.1)

ax.plot(angles2,values_jsimilar)
ax.fill(angles2, values_jsimilar, 'red', alpha=0.1)

#Rather than use a title, individual text points are added
plt.figtext(0.2,1.10,jugador,color="teal")
plt.figtext(0.2,1.05,"vs")
plt.figtext(0.2,1,jugador_similar,color="red")
plt.show()