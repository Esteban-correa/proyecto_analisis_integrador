import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np # Necesario para reemplazar infinitos

print("Iniciando el análisis de clustering de municipios...")

# --- 1. Cargar y Preparar Datos ---
df = pd.read_csv('Internet_Fijo_Penetraci_n_Municipio_20250804.csv')
df.dropna(inplace=True)
df = df[df['POBLACIÓN DANE'] > 0]
df.sort_values(by=['COD_MUNICIPIO', 'AÑO', 'TRIMESTRE'], inplace=True)

df['INDICE_PENETRACION'] = (df['No. ACCESOS FIJOS A INTERNET'] / df['POBLACIÓN DANE']) * 100

# --- 2. Ingeniería de Características ---
print("Calculando características para cada municipio...")
df_reciente = df.loc[df.groupby('COD_MUNICIPIO')[['AÑO', 'TRIMESTRE']].idxmax().iloc[:, 1]]
df_caracteristicas = df_reciente[['COD_MUNICIPIO', 'MUNICIPIO', 'DEPARTAMENTO', 'POBLACIÓN DANE', 'INDICE_PENETRACION']].copy()
df_caracteristicas.rename(columns={'INDICE_PENETRACION': 'INDICE_ACTUAL'}, inplace=True)

df['INDICE_AÑO_ANTERIOR'] = df.groupby('COD_MUNICIPIO')['INDICE_PENETRACION'].shift(4)
df_crecimiento = df.dropna(subset=['INDICE_AÑO_ANTERIOR'])

# Evitar división por cero o por números muy pequeños
df_crecimiento = df_crecimiento[df_crecimiento['INDICE_AÑO_ANTERIOR'] > 0.01] 
df_crecimiento['TASA_CRECIMIENTO'] = (df_crecimiento['INDICE_PENETRACION'] - df_crecimiento['INDICE_AÑO_ANTERIOR']) / df_crecimiento['INDICE_AÑO_ANTERIOR']

tasa_promedio = df_crecimiento.groupby('COD_MUNICIPIO')['TASA_CRECIMIENTO'].mean().reset_index()
df_caracteristicas = pd.merge(df_caracteristicas, tasa_promedio, on='COD_MUNICIPIO', how='left')

# --- INICIO DE LA CORRECCIÓN ---
# Limpiar valores infinitos y nulos, y podar los outliers
df_caracteristicas.replace([np.inf, -np.inf], np.nan, inplace=True)
df_caracteristicas.fillna(0, inplace=True)

# Podamos los valores extremos. Consideramos cualquier cosa por encima del percentil 95 como un outlier.
# Esto significa que una tasa de crecimiento de más del 500% (5.0) será limitada a ese valor.
limite_superior = df_caracteristicas['TASA_CRECIMIENTO'].quantile(0.95)
df_caracteristicas['TASA_CRECIMIENTO'] = df_caracteristicas['TASA_CRECIMIENTO'].clip(upper=limite_superior)
# --- FIN DE LA CORRECCIÓN ---

# --- 3. Modelo de Clustering (K-Means) ---
print("Entrenando el modelo K-Means...")
features_para_cluster = df_caracteristicas[['INDICE_ACTUAL', 'TASA_CRECIMIENTO', 'POBLACIÓN DANE']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_para_cluster)
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
kmeans.fit(features_scaled)
df_caracteristicas['CLUSTER'] = kmeans.labels_

# --- 4. Guardar Resultados ---
df_caracteristicas.to_csv('municipios_con_clusters.csv', index=False)
print("Análisis completado. Resultados guardados en 'municipios_con_clusters.csv'")