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
st.image("D:\\Work\\Aline\\Aline_data_project\\Reports\\gender_inequality.jpg")
st.sidebar.title("Sommaire")
pages=["Exploration","Data Visualisation","Modélisation"]
page=st.sidebar.radio("Aller vers", pages)
st.write("Le but de cette exploration de données est de comprendre les critères déterminant le score et le ranking du Gender Inequality Index pour les différents pays du monde.")
st.write("Il s'agit aussi de mettre en avant les facteurs importants de calcul du GII ainsi que leurs biais et limitations.")

if st.checkbox("Afficher"):
  st.write("Suite du Streamlit")

if page==pages[0]:
  st.write("### Exploration")
  st.write("Mon dataset comprend les éléments suivants:")
  st.dataframe(df.head(15))
  st.write("Le DataFrame comprend des colonnes contenant des informations sur les noms des pays ainsi que leur code (colonne IS03), les continents, les hémisphères. On a également les différents groupes de développement humain répartis en 4 catégories : low, medium, high et very high. Elles correspondent au HDI (Human Development Index) de 2021, qui mesure les différentes dimensions du développement humain : la durée de vie, la santé, l'éducation et la qualité de vie. On retrouve le classement des pays en fonction de leur HDI dans la colonne HDI ranking.")
  st.write("Ensuite nous avons le ranking de chanque pays selon le Gender Inequality Index en 2021 et enfin les scores du GII de 1990 à 2021.")

  if st.checkbox("Afficher les valeurs manquantes"):
    st.dataframe(df.isna().sum())
  if st.checkbox("Afficher les informations sur le DataFrame"):
    st.dataframe(df.info())    



if page==pages[1]:

  st.write("### Data Visualisation")
  st.write("Dans cette partie, je vais étudier les différentes variables à disposition du dataset.")

  fig=go.Figure()
  fig.add_trace(go.Scatter(x=df2.Country, y=df2.GII_rank_2021, name="GII rank"))
  fig.add_trace(go.Scatter(x=df2.Country, y=df2.HDI_rank_2021, name="HDI rank"))
  fig.update_layout(showlegend=True, title="Ranking GII et HDI par pays")
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)

  fig=go.Figure()
  fig.add_trace(go.Histogram(x=df2.Continent, nbinsx=20, marker_color="green", marker_line=dict(width=1,color="black"), name="Continents"))
  fig.add_trace(go.Histogram(x=df2.Hemisphere, nbinsx=20, marker_color="purple", marker_line=dict(width=1,color="black"), name="Hémisphères"))
  fig.update_layout(showlegend=True, title="Répartition par continent et hémisphère")
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)

  country_gii = df2.groupby(['Country']).agg({'GII_rank_2021':"median"})
  country_gii = country_gii.sort_values(by='GII_rank_2021', ascending=True).head(50)
  fig2 = go.Figure()
  fig2.add_traces(go.Bar(name='GII Rank', x=country_gii.index, y=country_gii['GII_rank_2021'],marker_color="#76b6ec")),
  fig2.update_layout(title="Ranking du GII 2021 des 50 pays les mieux classés")
  st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

  country_gii2 = df2.groupby(['Country']).agg({'GII_rank_2021':"median"})
  country_gii2 = country_gii2.sort_values(by='GII_rank_2021', ascending=False).head(50)
  fig3 = go.Figure()
  fig3.add_traces(go.Bar(name='GII Rank', x=country_gii2.index, y=country_gii2['GII_rank_2021'], marker_color="#6a9d74")),
  fig3.update_layout(title="Ranking du GII 2021 des 50 pays les moins bien classés")
  st.plotly_chart(fig3, theme="streamlit", use_container_width=True)


  fig4 = go.Figure()
  continent_gii = df2.groupby(['Continent']).agg({'GII_rank_2021':"mean"})
  continent_gii = continent_gii.sort_values(by='GII_rank_2021', ascending=True).head(50)
  fig4.add_traces(go.Bar(name='GII Rank', x=continent_gii.index, y=continent_gii['GII_rank_2021'], marker_color="#00a9ae")),
  fig4.update_layout(title="Ranking du GII 2021 par continent, du mieux au moins bien classé")
  st.plotly_chart(fig4, theme="streamlit", use_container_width=True)

  st.write("Lorsque l'on regarde les graphiques, il est évident que les meilleurs places (les plus bas scores, entre 1 et 50) sont trustés par les pays européens tandis que les moins bonnes places (les plus hauts scores) sont sur-représentées par les pays d'Afrique et d'Océanie. S'il est évident et prouvé que la pauvreté favorise l'inégalité de genre, on peut se poser la question du biais de cet index et du fait qu'il ne se base uniquement que sur des réalités économiques et non de représentativité. Une sur-représentation d'un continent que ce soit en positif ou en négatif est sujet à question.")

  st.write("Dans une prochaine étape, je vais effectuer quelques tests statistiques pour la corrélation des variables sur le ranking.")

  df_num=df2.select_dtypes(include=['int','float'])
  cor = df_num.corr() 
  fig, ax = plt.subplots(figsize = (15,15))
  sns.heatmap(cor, annot = True, ax = ax, cmap = "coolwarm")
  plt.title("Heatmap de corrélation des variables")
  st.pyplot(fig)


