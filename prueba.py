import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import seaborn as sns

df = pd.read_csv('Internet_Fijo_Penetraci_n_Municipio_20250804.csv')
df.dropna(inplace=True)
df.info()

print(df)

# Datos de salida
y = df['No. ACCESOS FIJOS A INTERNET']

# Datos de entrada 
x = df[['AÑO', 'TRIMESTRE', 'COD_MUNICIPIO','POBLACIÓN DANE']]

#se entrena el modela
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=19)
rfr = RandomForestRegressor(random_state=13)

#Se entrena el modelo
rfr.fit(x_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

y_pred = rfr.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R²:", r2)

rfr_pickle = open('tecnologias_Colombia.pickle', 'wb')
pickle.dump(rfr, rfr_pickle)
rfr_pickle.close()

nuevo = pd.DataFrame([{
    'AÑO': 2025,
    'TRIMESTRE': 3,
    'COD_MUNICIPIO': 5001,
    'POBLACIÓN DANE':3800000
}])

prediccion = rfr.predict(nuevo)
print("🔮 Predicción:", int(prediccion[0]))