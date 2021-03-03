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
import streamlit as st
from streamlit import cli as stcli
import spacy
import time
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
##Elegimos el jugador que queremos comparar

def main():
    ##Leemos el csv y creamos el df
    raw_data = "./players19-20.v.2.0.csv"

    df = pd.read_csv(raw_data, sep=";")

# Your streamlit code
    st.title('Búsqueda de jugadores')
    st.write("Vamos a buscar el jugador similar.")
    jugador = st.multiselect('Introduzca el jugador que desea comparar:', [c for c in df["Jugador"]])
    jugador = (jugador[0])

    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

    df_jugador = df.loc[df["Jugador"] == jugador]

    #obtengo laposición en la que juega el jugador seleccionado
    pos = df_jugador.iloc[0]["Posicion"]

    #Dependiendo el jugador que se quiera comparar se tendrán en cuenta unas variables u otras.
    #Para ello cogemos los valores por 90 minutos
    colsDef = ['Balones recuperados90min','Recuperaciones en campo rival90min','Disputas90min','% disputas ganadas90min',
               'Disputas aéreas90min','% disputas por arriba ganadas90min','Entradas90min','Interceptaciones90min',
               'Rechaces90min','pos_num']

    colsMc = ['Goles90Min','Asistencias90min','Tiros90min','Tiros a portería90min','Pases90min',
              '% de efectividad de pases90min','Pases de finalización90min','Centros90min',
              '% de efectividad de los centros90min','Balones perdidos90min','Pérdidas en campo propio90min',
               'Balones recuperados90min','Recuperaciones en campo rival90min','Goles esperados90min',
               'Disputas90min','% disputas ganadas90min','Disputas en ataque90min','Disputas aéreas90min',
               '% disputas por arriba ganadas90min','Regates90min','Entradas90min','Interceptaciones90min','Rechaces90min','pos_num']

    colsDel = ['Goles90Min','Asistencias90min','Tiros90min','Tiros a portería90min','% de penaltis marcados90min',
               'Pases90min','% de efectividad de pases90min','Pases de finalización90min','Centros90min',
               '% de efectividad de los centros90min','Balones perdidos90min','Pérdidas en campo propio90min',
               'Balones recuperados90min','Recuperaciones en campo rival90min','Goles esperados90min','Disputas90min',
               'Disputas en ataque90min','Regates90min','pos_num']

    viewcolsDef = ['Balones recuperados90min','Recuperaciones en campo rival90min','Disputas90min',
                   'Disputas aéreas90min','Entradas90min','Interceptaciones90min','Rechaces90min']

    viewcolsMc = ['Goles90Min','Asistencias90min','Tiros a portería90min','Pases90min',
              'Pases de finalización90min','Balones perdidos90min',
               'Balones recuperados90min','Regates90min','Interceptaciones90min']

    viewcolsDel = ['Goles90Min','Asistencias90min','Tiros90min','Tiros a portería90min',
                   'Pases de finalización90min','Balones perdidos90min','Goles esperados90min','Regates90min']

    with st.beta_container():
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


    #############################CHART UPGRADE#####################################


    def _scale_data(data, ranges):
        """scales data[1:] to ranges[0],
        inverts if the scale is reversed"""
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            assert (y1 <= d <= y2) or (y2 <= d <= y1)
        x1, x2 = ranges[0]
        d = data[0]

        sdata = [d]
        for d, (y1, y2) in zip(data[1:], ranges[1:]):

            sdata.append((d-y1) / (y2-y1)
                         * (x2 - x1) + x1)
        return sdata

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
                    if angle < 90 or angle > 300:
                        txt.set_horizontalalignment("left")

                    elif angle > 100 and angle < 260:
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

        def legend(self, data, *args, **kw):
            sdata = _scale_data(data, self.ranges)
            self.ax.legend(labels =(list_jug),bbox_to_anchor=(1.1, 1.2),prop={'size': 16})

    st.write('El jugador similar es:', nombre_jugador_sim)
    #df_jugador = df_jugador.set_index('Jugador', inplace=True)
    df_comun = pd.concat([df_jugador, datos_jsimilar])
    df_comun = df_comun.set_index('Jugador')
    st.table(df_comun[["Nacionalidad", "Edad", "Equipo","Liga"]])
    # Data
    variables = atributos
    data = values_jugador
    data2 = values_jsimilar
    ranges = rangecalc(pos)

    # plotting

    check_jugador = st.sidebar.checkbox(jugador, value = True)
    check_jugador_sim = st.sidebar.checkbox(jugador_similar, value = True)

    with _lock:
        if check_jugador and not check_jugador_sim:
            list_jug = [jugador]
            data_show = data
            fig1 = plt.figure(figsize=(8, 8))

        elif check_jugador_sim and not check_jugador:
            list_jug = [jugador_similar]
            data_show = data2
            fig1 = plt.figure(figsize=(8, 8))

        elif check_jugador_sim and check_jugador:
            list_jug = [jugador,jugador_similar]
            data_show = data2
            fig1 = plt.figure(figsize=(8, 8))
            radar = ComplexRadar(fig1, variables, ranges)
            radar.plot(data)
            radar.fill(data, alpha=0.2)
        else:
            list_jug = []
            data_show = []
            fig1 = plt.figure(figsize=(8, 8))
            #radar.legend(data)

        #fig1 = plt.figure(figsize=(8, 8))
        radar = ComplexRadar(fig1, variables, ranges)
        radar.plot(data_show)
        radar.fill(data_show, alpha=0.2)
        radar.legend(data_show)
        st.pyplot(fig1)


if __name__ == "__main__":
    try:
        if st._is_running_with_streamlit:
            main()
        else:
            sys.argv = ["streamlit", "run", sys.argv[0]]
            sys.exit(stcli.main())
    except:
  # Prevent the error from propagating into your Streamlit app.
        pass
