import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
df=pd.read_csv("C:\\Users\\Aline\\Documents\\perso\\Gender Inequality Index.csv")
#Le but de cette exploration de données est de comprendre l'évolution du GII au fil des années pour les différents pays
# Il s'agit aussi de déterminer quels sont les facteurs importants de calcul du GII et s'ils sont limitants
df.head()
df.isna().sum()
df.info()
df.duplicated().sum()
df.value_counts("Continent")
df.value_counts("Hemisphere")
df['Country'].unique()
dictionnaire={"Gender Inequality Index (1994)":"GII_1994",
             "Gender Inequality Index (1995)":"GII_1995",
             "Gender Inequality Index (1996)":"GII_1996",
             "Gender Inequality Index (1997)":"GII_1997",
             "Gender Inequality Index (1998)":"GII_1998",
             "Gender Inequality Index (1999)":"GII_1999",
             "Gender Inequality Index (2000)":"GII_2000",
             "Gender Inequality Index (2001)":"GII_2001",
             "Gender Inequality Index (2002)":"GII_2002",
             "Gender Inequality Index (2003)":"GII_2003",
             "Gender Inequality Index (2004)":"GII_2004",
             "Gender Inequality Index (2005)":"GII_2005",
             "Gender Inequality Index (2006)":"GII_2006",
             "Gender Inequality Index (2007)":"GII_2007",
             "Gender Inequality Index (2008)":"GII_2008",
             "Gender Inequality Index (2009)":"GII_2009",
             "Gender Inequality Index (2010)":"GII_2010",
             "Gender Inequality Index (2011)":"GII_2011",
             "Gender Inequality Index (2012)":"GII_2012",
             "Gender Inequality Index (2013)":"GII_2013",
             "Gender Inequality Index (2014)":"GII_2014",
             "Gender Inequality Index (2015)":"GII_2015",
             "Gender Inequality Index (2016)":"GII_2016",
             "Gender Inequality Index (2017)":"GII_2017",
             "Gender Inequality Index (2018)":"GII_2018",
             "Gender Inequality Index (2019)":"GII_2019",
             "Gender Inequality Index (2020)":"GII_2020",
             "Gender Inequality Index (2021)":"GII_2021",}
df=df.rename(dictionnaire, axis=1)
#Il y a trop de valeurs manquantes pour les GII entre 1990 et 2000, cela fausserait les observations, je vais donc supprimer
#ces colonnes
df.drop(["Gender Inequality Index (1990)","Gender Inequality Index (1991)","Gender Inequality Index (1992)","Gender Inequality Index (1993)","GII_1994","GII_1995","GII_1996","GII_1997","GII_1998","GII_1999","GII_2000"], axis=1, inplace=True)
#Les colonnes ISO3 et UNDP Developing Region n'apportent pas plus d'informations, je les supprime
df.drop(["ISO3","UNDP Developing Regions"], axis=1, inplace=True)
df.head()
df[df.isna().any(axis=1)]
#Pour les valeurs manquantes dans les colonnes GII, je décide de les remplacer par la médiane de chaque colonne, cela permet
# d'éviter de sur-évaluer un GII.
median_GII_2001 = df['GII_2001'].median()
df['GII_2001'].fillna(median_GII_2001, inplace=True)
median_GII_2002 = df['GII_2002'].median()
df['GII_2002'].fillna(median_GII_2002, inplace=True)

median_GII_2003 = df['GII_2003'].median()
df['GII_2003'].fillna(median_GII_2003, inplace=True)

median_GII_2004 = df['GII_2004'].median()
df['GII_2004'].fillna(median_GII_2004, inplace=True)

median_GII_2005 = df['GII_2005'].median()
df['GII_2005'].fillna(median_GII_2005, inplace=True)

median_GII_2006 = df['GII_2006'].median()
df['GII_2006'].fillna(median_GII_2006, inplace=True)

median_GII_2007 = df['GII_2007'].median()
df['GII_2007'].fillna(median_GII_2007, inplace=True)

median_GII_2008 = df['GII_2008'].median()
df['GII_2008'].fillna(median_GII_2008, inplace=True)

median_GII_2009 = df['GII_2009'].median()
df['GII_2009'].fillna(median_GII_2009, inplace=True)

median_GII_2010 = df['GII_2010'].median()
df['GII_2010'].fillna(median_GII_2010, inplace=True)

median_GII_2011 = df['GII_2011'].median()
df['GII_2011'].fillna(median_GII_2011, inplace=True)

median_GII_2012 = df['GII_2012'].median()
df['GII_2012'].fillna(median_GII_2012, inplace=True)

median_GII_2013 = df['GII_2013'].median()
df['GII_2013'].fillna(median_GII_2013, inplace=True)

median_GII_2014 = df['GII_2014'].median()
df['GII_2014'].fillna(median_GII_2014, inplace=True)

median_GII_2015 = df['GII_2015'].median()
df['GII_2015'].fillna(median_GII_2015, inplace=True)

median_GII_2016 = df['GII_2016'].median()
df['GII_2016'].fillna(median_GII_2016, inplace=True)

median_GII_2017 = df['GII_2017'].median()
df['GII_2017'].fillna(median_GII_2017, inplace=True)

median_GII_2018 = df['GII_2018'].median()
df['GII_2018'].fillna(median_GII_2018, inplace=True)

median_GII_2019 = df['GII_2019'].median()
df['GII_2019'].fillna(median_GII_2019, inplace=True)

median_GII_2020 = df['GII_2020'].median()
df['GII_2020'].fillna(median_GII_2020, inplace=True)

median_GII_2021 = df['GII_2021'].median()
df['GII_2021'].fillna(median_GII_2021, inplace=True)
df[df["GII Rank (2021)"].isna()]
#Pour les valeurs manquantes du GII Rank 2021, je vais les remplacer par les HDI Rank 2021 du même pays. Même si cela varie,
#la valeur reste plus proche qu'une moyenne ou une médiane des autres GII Rank.
df.loc[3,'GII Rank (2021)'] = 40.0
df.loc[7,'GII Rank (2021)'] = 71.0
df.loc[38,'GII Rank (2021)'] = 156.0
df.loc[45,'GII Rank (2021)'] = 171.0
df.loc[46,'GII Rank (2021)'] = 102.0
df.loc[52,'GII Rank (2021)'] = 176.0
df.loc[59,'GII Rank (2021)'] = 134.0
df.loc[67,'GII Rank (2021)'] = 145.0
df.loc[69,'GII Rank (2021)'] = 68.0
df.loc[72,'GII Rank (2021)'] = 4.0
df.loc[92,'GII Rank (2021)'] = 136.0
df.loc[93,'GII Rank (2021)'] = 75.0
df.loc[101,'GII Rank (2021)'] = 16.0
df.loc[113,'GII Rank (2021)'] = 131.0
df.loc[139,'GII Rank (2021)'] = 80.0
df.loc[145,'GII Rank (2021)'] = 106.0
df.loc[154,'GII Rank (2021)'] = 155.0
df.loc[157,'GII Rank (2021)'] = 44.0
df.loc[167,'GII Rank (2021)'] = 72.0
df.loc[179,'GII Rank (2021)'] = 130.0
df.loc[189,'GII Rank (2021)'] = 140.0
df[df["GII Rank (2021)"].isna()]
#Pour Monaco, Nauru, la Corée du Nord et la Somalie, il manque trop d'info pour qu'il y ait une quelconque pertinence, je
#vais supprimer ces pays
df.drop([108,132,142,158], inplace=True)
dictionnaire2={"Human Development Groups":"Human_development",
             "HDI Rank (2021)":"HDI_rank_2021",
              "GII Rank (2021)":"GII_rank_2021"}
df=df.rename(dictionnaire2, axis=1)

plt.figure(figsize=(10,10))
sns.displot(df['GII_2021'],kde=True,rug=True,bins=15, color="pink")
sns.displot(df['GII_2020'],kde=True,rug=True,bins=15, color="orange");
sns.displot(df['GII_2021'],kind="ecdf");

fig=go.Figure()
fig.add_trace(go.Scatter(x=df.Country, y=df.GII_rank_2021, name="GII rank"))
fig.add_trace(go.Scatter(x=df.Country, y=df.HDI_rank_2021, name="HDI rank"))
fig.update_layout(showlegend=True, legend_title="Ranking GII et HDI par pays")
fig.show()

fig=go.Figure()
fig.add_trace(go.Histogram(x=df.Continent, nbinsx=20, marker_color="green", marker_line=dict(width=1,color="black"), name="Continents"))
fig.add_trace(go.Histogram(x=df.Hemisphere, nbinsx=20, marker_color="purple", marker_line=dict(width=1,color="black"), name="Hémisphères"))
fig.update_layout(showlegend=True, legend_title="Répartition par continent et hémisphère")
fig.show()

country_gii = df.groupby(['Country']).agg({'GII_rank_2021':"median"})
country_gii = country_gii.sort_values(by='GII_rank_2021', ascending=True).head(50)

fig5 = go.Figure()
fig5.add_traces([go.Bar(name='GII Rank',
                       x=country_gii.index,
                        y=country_gii['GII_rank_2021'], marker_color="#76b6ec")]),
fig5.update_layout(title="Ranking du GII 2021 des 50 pays les mieux classés")
fig5.show();

country_gii2 = df.groupby(['Country']).agg({'GII_rank_2021':"median"})
country_gii2 = country_gii2.sort_values(by='GII_rank_2021', ascending=False).head(50)

fig5 = go.Figure()
fig5.add_traces([go.Bar(name='GII Rank',
                       x=country_gii2.index,
                        y=country_gii2['GII_rank_2021'], marker_color="#6a9d74")]),
fig5.update_layout(title="Ranking du GII 2021 des 50 pays les moins bien classés")
fig5.show();

continent_gii = df.groupby(['Continent']).agg({'GII_rank_2021':"mean"})
continent_gii = continent_gii.sort_values(by='GII_rank_2021', ascending=True).head(50)

fig5 = go.Figure()
fig5.add_traces([go.Bar(name='GII Rank',
                       x=continent_gii.index,
                        y=continent_gii['GII_rank_2021'], marker_color="#00a9ae")]),
fig5.update_layout(title="Ranking du GII 2021 par continent, du mieux au moins bien classé")
fig5.show();

#Lorsque l'on regarde les graphiques, il est évident que les meilleurs places (entre 1 et 50) sont trustés par les pays
#européens tandis que les moins bonnes places sont sur-représentées par les pays d'Afrique et d'Océanie. S'il est évident et
#prouvé que la pauvreté favorise l'inégalité de genre, on peut se poser la question du biais de cet index et de ce qu'il ne prend pas
#en compte pour établir ce ranking. Une sur-représentation d'un continent que ce soit en positif ou en négatif est sujet à question.
#Dans une prochaine étape, je vais effectuer quelques tests statistiques pour la corrélation des variables sur le ranking.

df['Human_development'].unique()
df=df.replace(to_replace=["Low","Medium","High","Very High"], value=[0,1,2,3])

df_num=df.select_dtypes(include=['int','float'])

cor = df_num.corr() 
fig, ax = plt.subplots(figsize = (12,12))
sns.heatmap(cor, annot = True, ax = ax, cmap = "coolwarm");

#Ici, le GII rank est fortement corrélé avec les différents GI Index, plus ils sont proches de 2021, plus ils sont corrélés,
#ce qui est logique. A noter que le human_development est corrélé négativement et que le HD Index est au contraire très
#corrélé. En effectuant le pre-processing et la normalisation, je vais voir si les variables Continent et Hemisphere sont
#importantes.

#Pour le pre-processing, je vais retirer la variable "Country" et ne laisser que l'hémisphère et le continent dans mon jeu
#de données.

df_preprocess=df.drop("Country", axis=1)
df_preprocess.head()

from sklearn.model_selection import train_test_split
X=df_preprocess.drop("GII_rank_2021", axis=1)
y=df_preprocess['GII_rank_2021']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import OneHotEncoder
cat=["Continent","Hemisphere"]
ohe=OneHotEncoder(drop="first",sparse=False)
X_train_ohe=ohe.fit_transform(X_train[cat])
X_test_ohe=ohe.transform(X_test[cat])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train_ohe)
X_test_scaler=scaler.transform(X_test_ohe)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train_scaled,y_train)

predictions=model.predict(X_test_scaler)
erreurs=predictions-y_test
print(erreurs)

def metrics_scikit_learn(y_test, predictions):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mse=mean_squared_error(y_test,predictions)
    rmse=np.sqrt(mse)
    mae=mean_absolute_error(y_test, predictions)
    return mse, rmse,mae
print(metrics_scikit_learn(y_test, predictions))

from sklearn.tree import DecisionTreeRegressor
model_dtr=DecisionTreeRegressor()
model_dtr.fit(X_train_scaled,y_train)

print("score train:", model_dtr.score(X_train_scaled,y_train))
print("score test:", model_dtr.score(X_test_scaler, y_test))

model_min_samples=DecisionTreeRegressor(max_depth=5,min_samples_leaf=24,random_state=42)
model_min_samples.fit(X_train_scaled, y_train)

print("score train:", model_min_samples.score(X_train_scaled,y_train))
print("score test:", model_min_samples.score(X_test_scaler, y_test))

from sklearn import ensemble
rfr=ensemble.RandomForestRegressor()
rfr.fit(X_train_scaled,y_train)
y_pred=rfr.predict(X_test_scaler)

pd.crosstab(y_test,y_pred,
           rownames=['Réel'],
           colnames=['Pred'])

print("score train:", rfr.score(X_train_scaled,y_train))
print("score test:", rfr.score(X_test_scaler, y_test))

#Des 3 modèles, le random forest est le plus performant pour trouver le ranking du GII à partir des autres scores GII.
#Néanmoins le score reste assez faible. Je vais donc explorer un autre dataset, avec d'autres variables afin de voir si
#le ranking du GII peut être déterminé par d'autres variables.



df2=pd.read_csv("C:\\Users\\Aline\\Documents\\perso\\Gender_Inequality_Index2.csv")

#Dans ce deuxième DataFrame, on retrouve les pays, le GII et le ranking 2021 ainsi que l'indice de développement humain.
#A la place des indices des index passés, on a le taux de mortalité maternelle, le taux de naissances de grossesses adolescentes
#le pourcentage de sièges tenus par des femmes aux parlements ainsi que la proportion d'hommes et de femmes dans l'éducation secondaire et
#la force de travail.


