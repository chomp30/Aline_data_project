import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

df=pd.read_csv("C:\\Users\\Aline\\Documents\\perso\\Gender Inequality Index.csv")
df2=pd.read_csv("D:\\Work\\Aline\\Aline_data_project\\df_gii2.csv")


st.title("Gender Inequality Index : analyse et prédictions")
st.sidebar.title("Sommaire")
pages=["Exploration","Data Visualisation","Modélisation"]
page=st.sidebar.radio("Aller vers", pages)
st.write("Le but de cette exploration de données est de comprendre les critères déterminant le score et le ranking du Gender Inequality Index pour les différents pays du monde.")
st.write("Il s'agit aussi de mettre en avant les facteurs importants de calcul du GII ainsi que leurs biais et limitations.")

if st.checkbox("Afficher"):
  st.write("Suite du Streamlit")

if page==pages[0]:
  st.write("### Introduction")
  st.write("Mon dataset comprend les éléments suivants:")
  st.dataframe(df.head(15))

  if st.checkbox("Afficher les valeurs manquantes"):
    st.dataframe(df.isna().sum())

if page==pages[1]:

  st.write("### Data Visualisation")
  st.write("Dans cette partie, je vais étudier les différentes variables à disposition du dataset.")

  fig=go.Figure()
  fig.add_trace(go.Scatter(x=df2.Country, y=df2.GII_rank_2021, name="GII rank"))
  fig.add_trace(go.Scatter(x=df2.Country, y=df2.HDI_rank_2021, name="HDI rank"))
  fig.update_layout(showlegend=True, title="Ranking GII et HDI par pays")
  st.plotly_chart(fig, theme=None, use_container_width=True)

  fig=go.Figure()
  fig.add_trace(go.Histogram(x=df2.Continent, nbinsx=20, marker_color="green", marker_line=dict(width=1,color="black"), name="Continents"))
  fig.add_trace(go.Histogram(x=df2.Hemisphere, nbinsx=20, marker_color="purple", marker_line=dict(width=1,color="black"), name="Hémisphères"))
  fig.update_layout(showlegend=True, title="Répartition par continent et hémisphère")
  st.plotly_chart(fig, theme=None, use_container_width=True)

  country_gii = df2.groupby(['Country']).agg({'GII_rank_2021':"median"})
  country_gii = country_gii.sort_values(by='GII_rank_2021', ascending=True).head(50)
  fig = go.Figure()
  fig.add_traces([go.Bar(name='GII Rank',
                       x=country_gii.index,
                        y=country_gii['GII_rank_2021'], marker_color="#76b6ec")]),
  fig.update_layout(title="Ranking du GII 2021 des 50 pays les mieux classés")
  st.plotly_chart(fig, theme=None, use_container_width=True)

  country_gii2 = df2.groupby(['Country']).agg({'GII_rank_2021':"median"})
  country_gii2 = country_gii2.sort_values(by='GII_rank_2021', ascending=False).head(50)
  fig = go.Figure()
  fig.add_traces([go.Bar(name='GII Rank',
                       x=country_gii2.index,
                        y=country_gii2['GII_rank_2021'], marker_color="#6a9d74")]),
  fig.update_layout(title="Ranking du GII 2021 des 50 pays les moins bien classés")
  st.plotly_chart(fig, theme=None, use_container_width=True)


  fig = go.Figure()
  continent_gii = df2.groupby(['Continent']).agg({'GII_rank_2021':"mean"})
  continent_gii = continent_gii.sort_values(by='GII_rank_2021', ascending=True).head(50)
  fig.add_traces([go.Bar(name='GII Rank',
                       x=continent_gii.index,
                        y=continent_gii['GII_rank_2021'], marker_color="#00a9ae")]),
  fig.update_layout(title="Ranking du GII 2021 par continent, du mieux au moins bien classé")
  st.plotly_chart(fig, theme=None, use_container_width=True)

  df_num=df2.select_dtypes(include=['int','float'])
  cor = df_num.corr() 
  fig, ax = plt.subplots(figsize = (15,15))
  sns.heatmap(cor, annot = True, ax = ax, cmap = "coolwarm")
  plt.title("Heatmap de corrélation des variables")
  st.pyplot(fig)


