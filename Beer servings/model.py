import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

data1=pd.read_csv("beer-servings.csv")

df1= data1.drop(['Unnamed: 0'],axis = 1)
df1 = df1.dropna(subset=['total_litres_of_pure_alcohol'])

for i in ['beer_servings','spirit_servings', 'wine_servings']:
    df1[i] = df1[i].fillna(df1[i].median())

from sklearn.preprocessing import LabelEncoder
le_country = LabelEncoder()
df1['country'] = le_country.fit_transform(df1['country'])

le_continent = LabelEncoder()
df1['continent'] = le_continent.fit_transform(df1['continent'])


x= df1.drop(['total_litres_of_pure_alcohol'], axis=1)
y=df1['total_litres_of_pure_alcohol']

from sklearn.preprocessing import StandardScaler
num_cols = ['beer_servings', 'spirit_servings', 'wine_servings']
std_scaler = StandardScaler()
x[num_cols] = std_scaler.fit_transform(x[num_cols])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state = 42)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor=regressor.fit(x_train,y_train)

pickle.dump(regressor,open('model.pkl','wb'))
pickle.dump(std_scaler, open('scaler.pkl', 'wb'))
pickle.dump(le_country, open('le_country.pkl', 'wb'))
pickle.dump(le_continent, open('le_continent.pkl', 'wb'))
