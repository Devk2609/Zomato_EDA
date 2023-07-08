import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn  as sns
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_csv('C:/Users/Admin/Desktop/Python_For_DevOps/Projects/Zomato_EDA/zomato1.csv', encoding=('ISO-8859-1'))
df.head()
df.info()
df.isnull().sum()
df.info()
df['Country Code'].value_counts()
df_country = pd.read_excel('C:/Users/Admin/Desktop/Python_For_DevOps/Projects/Zomato_EDA/Country-Code.xlsx')
df = pd.merge(df , df_country , how = 'left' , on = 'Country Code')
fig = px.scatter_mapbox(df,
                        lat="Latitude",
                        lon="Longitude",
                        hover_name="Restaurant Name" ,
                        hover_data=['Country','Aggregate rating'],
                        size= 'Votes')

fig.update_layout(mapbox_style="open-street-map")
fig.show()
df.Country.value_counts()
Countries = df.Country.value_counts().index
Country_brc = df.Country.value_counts().values
plt.figure(figsize = (12,9))
palette_color = sns.color_palette('hls', 8)
plt.pie(Country_brc[:5], labels=Countries[:5], colors=palette_color, autopct='%.0f%%')
plt.show()
df.columns
print(df['Country'].loc[df['Aggregate rating'] == 0 ].unique())
print(df.Country.unique())
df_rating_0 = df[['Restaurant Name', 'Country', 'City']].loc[df['Aggregate rating'] == 0 ]
df_rating_0_city =df_rating_0.loc[df_rating_0.Country == 'India']
df_rating_0_city.groupby('City').count()
df.columns
ratings=df.groupby(['Aggregate rating', 'Rating color', 'Rating text']).size().reset_index().rename(columns={0:'Rating Count'})
ratings.head()
plt.figure(figsize= (12,9))
sns.barplot(x= 'Aggregate rating', y = 'Rating Count',hue= 'Rating color' , data= ratings)
plt.show()
df[['Country', 'Has Online delivery']].loc[df['Has Online delivery']== 'Yes'].groupby('Country').count()
df_cities = df.City.value_counts
counttt = df.City.value_counts().index
cities = df.City.value_counts().values
plt.figure(figsize = (12,7))
ax = sns.barplot(x =counttt[:5], y = cities[:5])

for i in ax.containers:
    ax.bar_label(i,)
df_cuisines  = df.Cuisines.value_counts().reset_index()
df = df[['Restaurant Name', 'Cuisines']]
df.dropna(inplace=True)
unique_Cuisines = set()
for Cuisine in df['Cuisines']:
    unique_Cuisines.update(Cuisine.split(', '))
unique_Cuisines = sorted(list(unique_Cuisines))
Cuisine_count = {}
for Cuisine in df['Cuisines']:
    for c in Cuisine.split(', '):
        if c in Cuisine_count:
            Cuisine_count[c] += 1
        else:
            Cuisine_count[c] = 1
top_Cuisines = sorted(Cuisine_count.items(), key=lambda x: x[1], reverse=True)[:10]
top_Cuisine_names = [Cuisine[0] for Cuisine in top_Cuisines]
top_Cuisine_counts = [Cuisine[1] for Cuisine in top_Cuisines]
print(top_Cuisine_names)
print(top_Cuisine_counts)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_Cuisine_counts, y=top_Cuisine_names)
plt.title('Top 10 Cuisines by Count')
plt.xlabel('Count')
plt.ylabel('Cuisine')
plt.show()
vectorizer = CountVectorizer(binary=True)
Cuisine_binary_features = vectorizer.fit_transform(df['Cuisines'])

cosine_similarities = cosine_similarity(Cuisine_binary_features)

def get_similar_Cuisines(Cuisine_name, n=10):
    if Cuisine_name not in unique_Cuisines:
        print(f'{Cuisine_name} is not present in the dataset.')
        return [], []
    
    cosine_similarities_Cuisine = cosine_similarities[unique_Cuisines.index(Cuisine_name)]
    
    if n > len(unique_Cuisines) - 1:
        print('Value of n is greater than the number of unique cuisines.')
        return [], []
    
    similar_Cuisines_indices = np.argsort(cosine_similarities_Cuisine)[-n-1:-1][::-1]
    similar_Cuisines_indices = [index for index in similar_Cuisines_indices if index < len(unique_Cuisines)]
    
    similar_Cuisines = [unique_Cuisines[index] for index in similar_Cuisines_indices]
    similar_restaurants = df.loc[df['Cuisines'].str.contains('|'.join(similar_Cuisines)), 'Restaurant Name'].unique().tolist()
    
    return similar_Cuisines, similar_restaurants

similar_Cuisines, similar_restaurants = get_similar_Cuisines('', n=5)
print("Similar Cuisines:", similar_Cuisines)
print("Related Restaurants:", similar_restaurants)
