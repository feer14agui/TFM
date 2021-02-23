# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:00:23 2021

@author: f.aguilar.santos
"""

#obtenemos el fichero para crear el df
import pandas as pd
import numpy as np
import re
import math
import matplotlib.pyplot as plt
import sys
from streamlit import cli as stcli
import streamlit

def main():
# Your streamlit code
    st.title('Búsqueda de jugadores')
    st.write("Vamos a buscar el jugador similar.")
    x = st.text_input('Introduzca el jugador:')
    st.write('El jugador similar es:', x)

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
#Eliminamos los jugadores con menos de 5 partidos
df = df.drop(df[df['Partidos jugados_x']<10].index)


#############################CHART UPGRADE#####################################

def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    """if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1"""
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        """if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1"""
        sdata.append((d-y1) / (y2-y1)
                     * (x2 - x1) + x1)
    return sdata

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

def rangecalc(pos):
    if (pos=="DEF"):
        rang = [(0, round(df[atributos[0]].max())), (0, round(df[atributos[1]].max())),
          (0, round(df[atributos[2]].max())),(0, round(df[atributos[3]].max())),
         (0, round(df[atributos[4]].max())), (0, round(df[atributos[5]].max())),
         (0, round(df[atributos[6]].max()))]
    elif(pos=="MC"):
        rang = [(0, round(df[atributos[0]].max())), (0, round(df[atributos[1]].max())),
          (0, round(df[atributos[2]].max())),(0, round(df[atributos[3]].max())),
         (0, round(df[atributos[4]].max())), (0, round(df[atributos[5]].max())),
         (0, round(df[atributos[6]].max())), (0, round(df[atributos[7]].max())),
         (0, round(df[atributos[8]].max()))]
    elif(pos=="DEL"):
        rang = [(0, round(df[atributos[0]].max())), (0, round(df[atributos[1]].max())),
          (0, round(df[atributos[2]].max())),(0, round(df[atributos[3]].max())),
         (0, round(df[atributos[4]].max())), (0, round(df[atributos[5]].max())),
         (0, round(df[atributos[6]].max())), (0, round(df[atributos[7]].max()))]

    return rang

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

nombre_jugador = datos_jugador.iloc[0]["Jugador"]
nacionalidad_jugador = datos_jugador.iloc[0]["Nacionalidad"]
liga_jugador = datos_jugador.iloc[0]["Liga"]
equipo_jugador = datos_jugador.iloc[0]["Equipo"]
pos_jugador = datos_jugador.iloc[0]["Posicion"]
edad_jugador = datos_jugador.iloc[0]["Edad"]


nombre_jugador_sim = datos_jsimilar.iloc[0]["Jugador"]
nacionalidad_jugador_sim = datos_jsimilar.iloc[0]["Nacionalidad"]
liga_jugador_sim = datos_jsimilar.iloc[0]["Liga"]
equipo_jugador_sim = datos_jsimilar.iloc[0]["Equipo"]
pos_jugador_sim = datos_jsimilar.iloc[0]["Posicion"]
edad_jugador_sim = datos_jsimilar.iloc[0]["Edad"]


##################Representacion polar###########################
index_jugador = df_jugador.iloc[0].name
index_jugador_similar = index_jugadores_similares[0]

values_jugador = df_jugador_compare.loc[index_jugador, :].values.tolist()
#values_jugador += values_jugador [:1]
values_jsimilar = df_jsimilar.loc[index_jugador_similar, :].values.tolist()
#values_jsimilar += values_jsimilar [:1]
atributos =list(df_jugador_compare)

angles = [n / len(cols) * 2 * math.pi for n in range(len(cols))]
angles += angles [:1]

angles2 = [n / len(cols) * 2 * math.pi for n in range(len(cols))]
angles2 += angles2 [:1]

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=len(atributos)):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i))
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles,
                                         labels=variables,color='black',weight="bold")
        [txt.set_rotation(angle-90) for txt, angle
             in zip(text, angles)]

        for txt, angle in zip(text,angles):
                if angle == 0 or angle == 45 or angle == 315:
                    txt.set_horizontalalignment("left")
                elif angle == 180 or angle == 225 or angle == 135:
                    txt.set_horizontalalignment("right")

        [txt.set_rotation_mode("anchor") for txt, angle
             in zip(text, angles)]

        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,1))
                         for x in grid]
            '''if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid'''
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

print(atributos[0],atributos[1],atributos[2],atributos[3],atributos[4],atributos[5],atributos[6])
# Data
variables = atributos
data = values_jugador
data2 = values_jsimilar
ranges = rangecalc(pos)
# plotting
fig1 = plt.figure(figsize=(8, 8))
radar = ComplexRadar(fig1, variables, ranges)
radar.plot(data)
radar.plot(data2)
radar.fill(data, alpha=0.2)
radar.fill(data2, alpha=0.2)
plt.figtext(0.05,1.1,(nombre_jugador + "    " + nacionalidad_jugador),color="teal",size='large',weight='black')
plt.figtext(0.05,1.05,(liga_jugador + "    " + equipo_jugador),color="teal",size='large',weight='black')

plt.figtext(0.7,1.1,(nombre_jugador_sim + "    " + nacionalidad_jugador_sim),color="red",size='large',weight='black')
plt.figtext(0.7,1.05,(liga_jugador_sim + "    " + equipo_jugador_sim),color="red",size='large',weight='black')

plt.show()

if __name__ == "__main__":

    if streamlit._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
