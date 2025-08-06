import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression # Importado aquí para la función

# --- 0. Configuración de la Página y Estilo ---
st.set_page_config(
    page_title="Análisis de Internet en Colombia",
    page_icon="📊",
    layout="wide"
)
sns.set_theme(style="whitegrid")

# --- 1. Carga de Datos (Cacheada) ---
@st.cache_data
def cargar_datos():
    """Carga y prepara el DataFrame una sola vez."""
    df = pd.read_csv('https://raw.githubusercontent.com/Esteban-correa/proyecto_analisis_integrador/refs/heads/main/Internet_Fijo_Penetraci_n_Municipio_20250804.csv')
    # Limpieza de datos
    df['INDICE'] = df['INDICE'].str.replace(',', '.').astype(float)
    df.info()
    df.isnull().sum()
    df.isna().sum()
    df['No. ACCESOS FIJOS A INTERNET'] = df['No. ACCESOS FIJOS A INTERNET'].astype(int)
    return df

# --- 2. Definiciones de las Funciones para Gráficas y Lógica ---

def generar_grafico_evolucion(df, departamento_seleccionado):
    """Genera un gráfico de líneas para la evolución de accesos en un departamento."""
    df_dep = df[df['DEPARTAMENTO'].str.upper() == departamento_seleccionado.upper()].copy()
    df_grouped = df_dep.groupby(['AÑO', 'TRIMESTRE'])['No. ACCESOS FIJOS A INTERNET'].sum().reset_index()
    df_grouped.sort_values(by=['AÑO', 'TRIMESTRE'], inplace=True)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    x = range(len(df_grouped))
    y = df_grouped['No. ACCESOS FIJOS A INTERNET']
    trimestres_labels = [f"T{t}" for t in df_grouped['TRIMESTRE']]
    ax.plot(x, y, marker='o', linestyle='-', color='#007ACC', linewidth=2, markersize=8, label='No. de Accesos')
    
    ax.set_title(f'Evolución de Accesos Fijos a Internet en {departamento_seleccionado.title()}', fontsize=18, weight='bold')
    ax.set_xlabel('Año y Trimestre', fontsize=12)
    ax.set_ylabel('Número de Accesos Fijos', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(trimestres_labels, rotation=45, ha="right")
    
    y_pos_anio = ax.get_ylim()[0]
    for i, row in df_grouped.iterrows():
        if row['TRIMESTRE'] == 1:
            ax.text(i, y_pos_anio, row['AÑO'], ha='center', va='bottom', fontsize=11, weight='bold')
            
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
    fig.tight_layout()
    return fig, df_grouped

def crear_grafico_ranking(df, año, trimestre, nivel_geografico, tipo_ranking, n=10):
    """Función flexible para generar gráficos de ranking."""
    df_periodo = df[(df['AÑO'] == año) & (df['TRIMESTRE'] == trimestre)].copy()
    
    if nivel_geografico == 'DEPARTAMENTO':
        data_proc = df_periodo.groupby('DEPARTAMENTO').agg({'No. ACCESOS FIJOS A INTERNET': 'sum', 'POBLACIÓN DANE': 'sum'}).reset_index()
    else:
        data_proc = df_periodo.copy()
        
    data_proc = data_proc[data_proc['POBLACIÓN DANE'] > 0]
    data_proc['INDICE_CALCULADO'] = (data_proc['No. ACCESOS FIJOS A INTERNET'] / data_proc['POBLACIÓN DANE']) * 100
    
    ascendente = (tipo_ranking == 'peores')
    if ascendente:
        data_proc = data_proc[data_proc['No. ACCESOS FIJOS A INTERNET'] > 0]
    
    data_ranked = data_proc.sort_values('INDICE_CALCULADO', ascending=ascendente).head(n)
    
    if data_ranked.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No hay datos para esta selección", ha='center')
        return fig, pd.DataFrame()
        
    titulo_ranking = "Mejores" if tipo_ranking == 'mejores' else "Peores"
    titulo_geo = "Departamentos" if nivel_geografico == 'DEPARTAMENTO' else "Municipios"
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=data_ranked['INDICE_CALCULADO'], y=data_ranked[nivel_geografico], palette='viridis' if tipo_ranking == 'mejores' else 'rocket_r', ax=ax)
    ax.set_title(f'{titulo_ranking} {n} {titulo_geo} por Penetración ({año}-T{trimestre})', fontsize=16, weight='bold')
    ax.set_xlabel('Accesos por cada 100 Habitantes (%)', fontsize=12)
    ax.set_ylabel(titulo_geo[:-1] if nivel_geografico.endswith('s') else titulo_geo, fontsize=12)
    fig.tight_layout()
    return fig, data_ranked

def crear_torta_accesos_por_departamento(df, top_n=10):
    """Genera un gráfico de torta con la distribución de accesos por departamento."""
    accesos_por_depto = df.groupby('DEPARTAMENTO')['No. ACCESOS FIJOS A INTERNET'].sum().sort_values(ascending=False)
    
    if len(accesos_por_depto) > top_n:
        top_deptos = accesos_por_depto.head(top_n)
        otros_sum = accesos_por_depto.iloc[top_n:].sum()
        data_to_plot = pd.concat([top_deptos, pd.Series({'Otros': otros_sum})])
    else:
        data_to_plot = accesos_por_depto
        
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = sns.color_palette('viridis_r', len(data_to_plot))
    wedges, texts, autotexts = ax.pie(data_to_plot, autopct='%1.1f%%', startangle=90, colors=colors, pctdistance=0.85, wedgeprops=dict(width=0.4))
    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax.set_title('Distribución Porcentual de Accesos a Internet por Departamento (Histórico)', fontsize=16, weight='bold')
    ax.legend(wedges, data_to_plot.index, title="Departamentos", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    return fig, data_to_plot.reset_index(name='Total Accesos')

def estimar_poblacion_futura(df, cod_municipio, año_objetivo):
    """Estima la población de un municipio para un año futuro."""
    historial_mun = df[df['COD_MUNICIPIO'] == cod_municipio][['AÑO', 'POBLACIÓN DANE']].drop_duplicates().sort_values('AÑO')
    
    if historial_mun.empty: return 0
    
    last_known_year = historial_mun['AÑO'].iloc[-1]
    last_known_pop = historial_mun['POBLACIÓN DANE'].iloc[-1]

    if año_objetivo <= last_known_year: return last_known_pop

    if len(historial_mun) < 2:
        avg_growth_rate = 0.005
    else:
        avg_growth_rate = historial_mun['POBLACIÓN DANE'].pct_change().mean()

    growth_rate_to_use = max(avg_growth_rate, 0.002)
    num_years_to_project = año_objetivo - last_known_year
    poblacion_predicha = last_known_pop * ((1 + growth_rate_to_use) ** num_years_to_project)
    
    return int(poblacion_predicha)

# --- 3. Construcción de la Aplicación en Streamlit ---
df_main = cargar_datos()
st.title("Análisis de Acceso a Internet Fijo en Colombia")
st.write("Este análisis examina en profundidad el estado del acceso a Internet fijo en Colombia, evaluando su cobertura, penetración y evolución en los últimos años. Incluye la identificación de regiones con mayores y menores niveles de conectividad, las brechas digitales existentes entre zonas urbanas y rurales, y los factores que influyen en la calidad y disponibilidad del servicio. A través de datos estadísticos y tendencias, se busca ofrecer una visión completa del panorama actual y de las oportunidades para fortalecer la infraestructura y garantizar un acceso más equitativo a nivel nacional.")

# --- CORRECCIÓN DE LA LÍNEA DEL ERROR ---
# Se crean 3 variables para 3 pestañas
tab1, tab2, tab3 = st.tabs([" Análisis Exploratorio", " Modelo Predictivo", "Sobre Nosotros"])

# --- Pestaña 1: Análisis Exploratorio ---
with tab1:
    st.header("Evolución Histórica por Departamento")

    depto_sel_evolucion = st.selectbox("Departamento", sorted(df_main['DEPARTAMENTO'].unique()))
    fig_evolucion, datos_evolucion = generar_grafico_evolucion(df_main, depto_sel_evolucion)

    col_e1, col_e2 = st.columns([2, 1])
    with col_e1:
        st.pyplot(fig_evolucion)
    with col_e2:
        st.info("Pregunta: ¿Cómo ha crecido el acceso a internet entre trimestres?")
        crecimiento = datos_evolucion['No. ACCESOS FIJOS A INTERNET'].pct_change().mean() * 100
        st.markdown(f"Respuesta: En promedio, ha crecido un {crecimiento:.2f}% por trimestre.")
        st.dataframe(datos_evolucion)

    st.divider()


    st.header("Ranking de Conectividad por Periodo")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        año_sel = st.selectbox("Año", sorted(df_main['AÑO'].unique(), reverse=True), key='ranking_año')
    with col2:
        trim_sel = st.selectbox("Trimestre", sorted(df_main['TRIMESTRE'].unique()), key='ranking_trim')
    with col3:
        geo_sel = st.selectbox("Nivel Geográfico", ["DEPARTAMENTO", "MUNICIPIO"], key='ranking_geo')
    with col4:
        ranking_sel = st.selectbox("Tipo de Ranking", ["Mejores 10", "Peores 10"], key='ranking_tipo')

    tipo_ranking_map = 'mejores' if ranking_sel == "Mejores 10" else 'peores'
    fig_ranking, datos_ranking = crear_grafico_ranking(df_main, año_sel, trim_sel, geo_sel, tipo_ranking_map)

    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        st.pyplot(fig_ranking)
    with col_r2:    
        st.success("Pregunta: ¿Qué municipios tienen más/menos accesos por cada 100 habitantes?")
        if not datos_ranking.empty:
            st.markdown(f"Respuesta: El primero del ranking es {datos_ranking[geo_sel].iloc[0]} con {datos_ranking['INDICE_CALCULADO'].iloc[0]:.2f}%.")
        st.dataframe(datos_ranking)

    st.divider()

    st.header("Distribución General de Accesos por Departamento")
    fig_torta, datos_torta = crear_torta_accesos_por_departamento(df_main)
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.pyplot(fig_torta)
    with col_t2:
        st.warning("Pregunta: ¿Existe desigualdad regional dentro de un departamento?")
        top = datos_torta['Total Accesos'].max()
        bottom = datos_torta['Total Accesos'].min()
        st.markdown(f"Respuesta: Sí, existe una marcada desigualdad regional dentro de los departamentos colombianos. Aunque la gráfica muestra la concentración del acceso a internet en unos pocos departamentos, este mismo patrón de centralización se repite a escala interna.")
        st.dataframe(datos_torta)

    st.divider()
    st.header("Conclusiones y Recomendaciones")

    col_conc, col_rec = st.columns(2)

    with col_conc:
        st.subheader("Conclusiones")
        
        st.success(
            "1. A nivel nacional, el acceso fijo a internet ha mostrado una tendencia de crecimiento constante, "
            "especialmente en departamentos con mayor urbanización. Esto sugiere un avance progresivo en la cobertura."
        )

        st.success(
            "2. Persiste una brecha importante entre departamentos y municipios. En varios casos, aunque el total de accesos "
            "es alto, la penetración relativa (accesos por cada 100 habitantes) sigue siendo baja, reflejando desigualdad regional."
        )

        st.success(
            "3. Algunos municipios pequeños superan a otros más grandes en términos de penetración, lo cual indica que el tamaño "
            "de la población no siempre está alineado con el nivel de conectividad. Esto puede reflejar políticas locales más efectivas."
        )

    with col_rec:
        st.subheader("Recomendaciones")
        
        st.info("Filtrar por municipios y no solo por departamentos permite identificar zonas con alto desempeño relativo, incluso si son pequeñas.")
        
        st.info("Comparar trimestres consecutivos ayuda a detectar caídas inesperadas en la conectividad y evaluar el impacto de nuevas políticas.")
        
        st.info("Observar la relación entre población y accesos permite priorizar acciones en territorios donde hay mayor déficit relativo de conectividad.")
# --- Pestaña 2: Modelo Predictivo (con controles completos) ---
# --- Pestaña 2: Modelo Predictivo (con controles completos) ---

with tab2:
    st.title("Predicción de Accesos a Internet")
    st.write("Este análisis se enfoca en la predicción del número de accesos fijos a Internet en Colombia, empleando el modelo de aprendizaje automático Random Forest para generar estimaciones precisas a partir de datos históricos y tendencias de conectividad. El estudio permite proyectar escenarios futuros que facilitan la planificación de infraestructura, la optimización de recursos y la formulación de estrategias para cerrar brechas digitales. Su enfoque combina técnicas de análisis predictivo con datos reales, brindando una herramienta útil para empresas, instituciones y entidades gubernamentales que buscan mejorar el acceso y la calidad del servicio.")
    st.header("Random Forest Regressor")
    st.write("Random Forest es un algoritmo de aprendizaje automático basado en la creación de múltiples árboles de decisión que trabajan en conjunto para realizar predicciones más precisas y confiables. Cada árbol analiza una parte de los datos y aporta su resultado; luego, el modelo combina todas las respuestas para obtener una estimación final. Esta técnica es ampliamente utilizada por su capacidad para manejar grandes volúmenes de información, reducir el riesgo de sobreajuste y ofrecer resultados sólidos tanto en tareas de clasificación como de regresión.")
    st.markdown("---")
    st.markdown("##### Estima el número de accesos fijos para un municipio en un periodo específico.")
    st.write("") # Espacio vertical

    # --- Inicializar el estado de la sesión ---
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
        st.session_state.prediction_params = None

    try:
        # --- Cargar modelo (se hace una sola vez gracias al caché) ---
        @st.cache_resource
        def load_model():
            with open('tecnologias_Colombia.pickle', 'rb') as file:
                return pickle.load(file)
        
        modelo_cargado = load_model()

        # --- SECCIÓN 1: PARÁMETROS DE ENTRADA ---
        with st.container(border=True):
            st.subheader("Parámetros de Simulación")
            st.markdown("Selecciona un municipio y el periodo que deseas estimar.")
            
            col1, col2 = st.columns(2)
            with col1:
                max_year = df_main['AÑO'].max()
                años_disponibles = sorted([a for a in df_main['AÑO'].unique() if a <= max_year], reverse=True)
                pred_año = st.selectbox("Año de referencia", años_disponibles, key="pred_año")
                pred_trimestre = st.selectbox("Trimestre", sorted(df_main['TRIMESTRE'].unique()), key="pred_trim")
            
            with col2:
                municipios = sorted(df_main['MUNICIPIO'].unique())
                # Asignar un índice por defecto seguro
                default_index = municipios.index("MEDELLÍN") if "MEDELLÍN" in municipios else 0
                pred_municipio = st.selectbox("Municipio", municipios, key="pred_municipio", index=default_index)

        # --- LÓGICA DE INVALIDACIÓN (LA CORRECCIÓN CLAVE) ---
        # Si los parámetros actuales no coinciden con los de la última predicción guardada,
        # significa que el usuario cambió algo pero no ha recalculado. Borramos el resultado viejo.
        if st.session_state.prediction_params and (
            pred_municipio != st.session_state.prediction_params['municipio'] or
            pred_año != st.session_state.prediction_params['año'] or
            pred_trimestre != st.session_state.prediction_params['trimestre']
        ):
            st.session_state.prediction_result = None
            st.session_state.prediction_params = None
            # Opcional: st.rerun() para forzar un refresco visual inmediato si fuera necesario.

        st.write("") # Espacio

        # --- SECCIÓN 2: BOTÓN DE CÁLCULO ---
        col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 1.5, 1])
        with col_btn_2:
            if st.button("Estimar Número de Accesos", type="primary", use_container_width=True):
                try:
                    # --- Preparar datos para la predicción ---
                    municipio_info = df_main[df_main['MUNICIPIO'] == pred_municipio].iloc[0]
                    cod_municipio = municipio_info['COD_MUNICIPIO']
                    
                    poblacion_data = df_main[
                        (df_main['COD_MUNICIPIO'] == cod_municipio) & (df_main['AÑO'] == pred_año)
                    ]['POBLACIÓN DANE']
                    
                    if poblacion_data.empty:
                        st.warning(f"No se encontraron datos de población para {pred_municipio} en el año {pred_año}. Por favor, intente con otro periodo.", icon="⚠️")
                        st.session_state.prediction_result = None # Limpiar resultado
                    else:
                        poblacion = poblacion_data.iloc[0]
                        datos_input = pd.DataFrame([{
                            'AÑO': pred_año,
                            'TRIMESTRE': pred_trimestre,
                            'COD_MUNICIPIO': cod_municipio,
                            'POBLACIÓN DANE': poblacion
                        }])

                        # --- Predicción y almacenamiento en el estado ---
                        prediccion = modelo_cargado.predict(datos_input)
                        st.session_state.prediction_result = int(prediccion[0])
                        # Guardamos los parámetros CON los que se hizo esta predicción
                        st.session_state.prediction_params = {
                            "municipio": pred_municipio,
                            "departamento": municipio_info['DEPARTAMENTO'],
                            "año": pred_año,
                            "trimestre": pred_trimestre
                        }

                except Exception as e:
                    st.error(f"Ocurrió un error al preparar los datos: {e}", icon="🚨")
                    st.session_state.prediction_result = None

        st.write("---")

        # --- SECCIÓN 3: MOSTRAR RESULTADO (si existe y es válido) ---
        if st.session_state.prediction_result is not None:
            params = st.session_state.prediction_params
            st.subheader("Resultado de la Estimación")

            with st.container(border=True):
                st.success(
                    f"Estimación para *{params['municipio']}* ({params['departamento']}) en el periodo *{params['año']}-T{params['trimestre']}*:",
                    icon="✅"
                )
                
                # Usar columnas para una mejor alineación de la métrica
                met_col1, met_col2 = st.columns([1, 2])
                with met_col1:
                    st.metric(
                        label="Accesos Fijos Estimados",
                        value=f"{st.session_state.prediction_result:,}".replace(",", ".")
                    )
                
                st.info("Nota: Esta es una estimación generada por un modelo de Machine Learning y puede variar respecto a los datos reales.", icon="ℹ️")

    except FileNotFoundError:
        st.error("Error Crítico: No se encontró el archivo del modelo tecnologias_Colombia.pickle. Asegúrese de que el archivo esté en la carpeta raíz del proyecto.", icon="🚨")
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al cargar la aplicación: {e}", icon="🚨")





# --- Pestaña 3: Segmentación de Municipios ---
# Reemplaza el código de tu 'tab3' con esto
# Reemplaza el código de tu 'tab3' con esto
with tab3:
    st.title("Un Proyecto Presentado Por")
    st.markdown("##### Conoce al equipo y las herramientas detrás de esta aplicación de análisis de datos.")
    st.write("---")

    # --- Misión del Proyecto ---
    st.info(
        "Nuestra Misión: Utilizar la ciencia de datos para visibilizar la brecha digital en Colombia, "
        "transformando datos públicos en información accesible y accionable para todos."
    )

    # --- Perfiles de los Creadores en un layout para 5 personas ---
    
    # Fila 1
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            # Para agregar una foto, sube el archivo a la carpeta y descomenta la siguiente línea:
            # st.image("foto_alejandro.png") 
            st.subheader("Alejandro Orrego Roldan")
            st.write("**Rol:** Integrante del Equipo de Desarrollo")
            st.markdown("*Aportes en el análisis de datos y diseño del modelo.*")
            st.markdown("---")

    with col2:
        with st.container(border=True):
            # st.image("foto_esteban.png") 
            st.subheader("Esteban Correa Roldan")
            st.write("**Rol:** Integrante del Equipo de Desarrollo")
            st.markdown("*Aportes en el análisis de datos y desarrollo del proyecto.*")
            st.markdown("---")

    with col3:
        with st.container(border=True):
            # st.image("foto_camilo.png") 
            st.subheader("Camilo Zúñiga Morelo")
            st.write("**Rol:** Integrante del Equipo de Desarrollo")
            st.markdown("*Aportes en el análisis de datos y documentación del proyecto.*")
            st.markdown("---")

    # Fila 2 (centrada usando columnas vacías como espaciadores)
    spacer1, col4, col5, spacer2 = st.columns([0.5, 1, 1, 0.5])
    with col4:
        with st.container(border=True):
            # st.image("foto_farley.png") 
            st.subheader("Farley Barrera López (Líder)")
            st.write("**Rol:** Líder Supremo")
            st.markdown("*Aportes en el análisis de datos.*")
            st.markdown("---")
            
    with col5:
        with st.container(border=True):
            # st.image("foto_leandro.png") 
            st.subheader("Leandro Toro López")
            st.write("**Rol:** Integrante del Equipo de Desarrollo")
            st.markdown("*Aportes en el análisis de datos y desarrollo del modelo.*")
            st.markdown("---")
            
    st.write("---")

    # --- Tecnologías Utilizadas ---
    st.subheader("Herramientas y Plataformas")
    st.markdown("Este proyecto fue construido con las siguientes tecnologías:")
    
    st.code(
        """
 Python         (Lenguaje principal de programación)
 Streamlit      (Framework para la interfaz web interactiva)
 Pandas         (Para la manipulación y limpieza de datos)
 Scikit-learn   (Para el modelo de Machine Learning)
 Matplotlib     (Para las gráficas dinámicas)
 SQL Workbench  (Para la gestión y consulta de bases de datos)
 GitHub         (Para el control de versiones y colaboración)
        """,
        language='markdown'
    )


