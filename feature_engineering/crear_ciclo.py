import pandas as pd
import numpy as np

# --- CONSTANTES ---
CREDITOS_REQUERIDOS = 279
TOTAL_CURSOS = 71


def crear_features_acumuladas(df_original):
    """
    Crea un DataFrame de "snapshots" por estudiante y período con métricas
    académicas acumuladas HASTA EL CICLO ANTERIOR.
    
    Argumentos:
    df_original -- El DataFrame 'df_clean' completo.
    
    Devuelve:
    df_semestre -- El DataFrame agregado por (COD_PERSONA, PER_MATRICULA).
    """
    creditos_requeridos = CREDITOS_REQUERIDOS
    total_cursos = TOTAL_CURSOS
    
    print("Iniciando procesamiento...")
    df = df_original.copy()

    # --- 0. Generar columna APROBO si no existe ---
    df['APROBO'] = df['NOTA'].apply(lambda x: 'S' if x >= 11.5 else 'N')    
    
    # --- 1. Ordenar los datos (CRÍTICO para cálculos acumulados) ---
    df = df.sort_values(by=['COD_PERSONA', 'PER_MATRICULA'])

    # --- 2. Crear Columnas de Ayuda (a nivel de curso) ---
    valores_aprobados = ['APROBADO', 'S', 'Si'] 
    df['APROBO_CURSO_NUM'] = df['APROBO'].apply(lambda x: 1 if x in valores_aprobados else 0)
    df['CREDITOS_APROBADOS'] = df['CREDITOS'] * df['APROBO_CURSO_NUM']
    df['PONDERADO_CURSO'] = df['NOTA'] * df['CREDITOS']

    # --- 3. Calcular Métricas Acumuladas (por estudiante) ---
    g = df.groupby('COD_PERSONA')
    
    df['CUM_CREDITOS_LLEVADOS'] = g['CREDITOS'].cumsum()
    df['CUM_CURSOS_LLEVADOS'] = g['COD_CURSO'].cumcount() + 1
    df['CUM_CREDITOS_APROBADOS'] = g['CREDITOS_APROBADOS'].cumsum()
    df['CUM_CURSOS_APROBADOS'] = g['APROBO_CURSO_NUM'].cumsum()
    df['CUM_PONDERADO_TOTAL'] = g['PONDERADO_CURSO'].cumsum()

    # --- 4. Agregar a Nivel Semestre ---
    df_semestre = df.groupby(['COD_PERSONA', 'PER_MATRICULA']).last()

    # --- 5. Calcular Features Finales (a nivel de semestre) ---
    
    # a. Promedio acumulado (temporal, al final del ciclo)
    df_semestre['PROM_ACUMULADO'] = df_semestre['CUM_PONDERADO_TOTAL'] / df_semestre['CUM_CREDITOS_LLEVADOS']

    # b. Porcentajes de avance (temporal, al final del ciclo)
    df_semestre['PCT_CREDITOS_APROBADOS'] = df_semestre['CUM_CREDITOS_APROBADOS'] / creditos_requeridos
    df_semestre['PCT_CURSOS_APROBADOS'] = df_semestre['CUM_CURSOS_APROBADOS'] / total_cursos
    df_semestre['PCT_CREDITOS_LLEVADOS'] = df_semestre['CUM_CREDITOS_LLEVADOS'] / creditos_requeridos
    df_semestre['PCT_CURSOS_LLEVADOS'] = df_semestre['CUM_CURSOS_LLEVADOS'] / total_cursos

    # -----------------------------------------------------------------
    # ⬇️ ⬇️ ⬇️ Generar SEM_CURSADOS en df_semestre ⬇️ ⬇️ ⬇️
    # -----------------------------------------------------------------

    # Convertimos PER_MATRICULA ('2020-01') a formato numérico ordenable
    df_semestre['PER_MATRICULA_INT'] = df_semestre.index.get_level_values('PER_MATRICULA').str.replace('-', '').astype(int)

    # Calculamos cuántos semestres distintos cursó ANTES del actual
    df_semestre['SEM_CURSADOS'] = (
        df_semestre.groupby('COD_PERSONA')['PER_MATRICULA_INT']
        .rank(method='dense')
        .astype(int) - 1
    )


    # -----------------------------------------------------------------
    # ⬇️ ⬇️ ⬇️ CAMBIO PRINCIPAL: APLICAR SHIFT ⬇️ ⬇️ ⬇️
    # -----------------------------------------------------------------
    #
    # Aplicamos .shift(1) agrupado por persona para obtener el valor
    # del ciclo anterior. Rellenamos con 0 los NaN (que son los 
    # estudiantes en su primer ciclo).
    
    print("Aplicando shift para obtener datos del ciclo anterior...")
    
    cols_para_shift = [
        'PROM_ACUMULADO',
        'PCT_CREDITOS_APROBADOS',
        'PCT_CURSOS_APROBADOS',
        'PCT_CREDITOS_LLEVADOS',
        'PCT_CURSOS_LLEVADOS'
    ]
    
    # Agrupamos por estudiante ANTES de hacer el shift
    g_semestre = df_semestre.groupby('COD_PERSONA')
    
    for col in cols_para_shift:
        # Sobreescribimos la columna con su valor anterior
        df_semestre[col] = g_semestre[col].shift(1).fillna(0)
    
    # -----------------------------------------------------------------
    # ⬆️ ⬆️ ⬆️ FIN DEL CAMBIO ⬆️ ⬆️ ⬆️
    # -----------------------------------------------------------------

    
    # c. Edad al momento del período (Esta se mantiene igual)
    periodo_str = df_semestre.index.get_level_values('PER_MATRICULA').astype(str)
    fechas_periodo = pd.to_datetime(
        periodo_str.str.replace('-01', '-03-01')
                       .str.replace('-02', '-08-01')
                       .str.replace('-00', '-01-01'),
        errors='coerce'
    )
    df_semestre['EDAD'] = (fechas_periodo - pd.to_datetime(df_semestre['FECHA_NACIMIENTO'], errors='coerce')).dt.days / 365.25

    # d. Ranking (basado en el promedio del ciclo anterior)
    
    # 1. Reemplazamos 0.0 con NaN temporalmente para que .rank() los ignore
    #    (Un alumno nuevo con 0.0 no debe ser rankeado)
    prom_previo = df_semestre['PROM_ACUMULADO'].replace(0, np.nan)
    
    df_semestre['RANK_PCT'] = df_semestre.groupby('PER_MATRICULA') \
                                     ['PROM_ACUMULADO'].rank(pct=True, ascending=False)
    
    # 2. Definimos cortes y etiquetas
    bins = [0, 0.1, 0.2, 1/3, 1.0]
    labels = ['10mo Superior', '5to Superior', '3cio Superior', 'General']
    
    df_semestre['RANKING'] = pd.cut(
        df_semestre['RANK_PCT'],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )
    
    # 3. Asignamos 'General' a los que tenían 0 (alumnos nuevos)
    df_semestre['RANKING'] = df_semestre['RANKING'].fillna('General')


    # --- 6. Limpieza Final ---
    df_semestre = df_semestre.reset_index()
    
    columnas_finales = [
        'COD_PERSONA', 
        'PER_MATRICULA',
        'PROM_ACUMULADO',
        'PCT_CREDITOS_APROBADOS',
        'PCT_CURSOS_APROBADOS',
        'PCT_CREDITOS_LLEVADOS',
        'PCT_CURSOS_LLEVADOS',
        'SEM_CURSADOS', 
        'EDAD',
        'RANKING'
    ]
    
    # Seleccionar solo las columnas solicitadas
    df_semestre_final = df_semestre[columnas_finales].copy()
    
    # Renombramos la columna de promedio para claridad
    df_semestre_final = df_semestre_final.rename(columns={
        'PROM_ACUMULADO': 'Promedio acumulado (Ciclo Anterior)'
    })
    
    print(f"✅ Procesamiento completado. DataFrame final con {df_semestre_final.shape[0]} filas.")

    return df_semestre_final

# --- Cómo usar la función ---

# (Asumiendo que tu DataFrame original se llama 'df_clean')
# df_snapshots = crear_features_acumuladas(df_clean)

# (Opcional) Ver la estructura y una muestra de los resultados
# print(df_snapshots.info())
# print(df_snapshots.head())


import pandas as pd
import numpy as np

# --- CONSTANTES ---
CREDITOS_REQUERIDOS = 279
TOTAL_CURSOS = 71


def crear_features_acumuladas_2(df_original):
    """
    Crea un DataFrame de "snapshots" por estudiante y período con métricas
    académicas acumuladas HASTA EL CICLO ANTERIOR.
    
    MODIFICADO: Ahora incluye tanto el Promedio Acumulado (NG) como el 
    Promedio del Ciclo Pasado (PC).
    
    Argumentos:
    df_original -- El DataFrame 'df_clean' completo.
    
    Devuelve:
    df_semestre -- El DataFrame agregado por (COD_PERSONA, PER_MATRICULA).
    """
    creditos_requeridos = CREDITOS_REQUERIDOS
    total_cursos = TOTAL_CURSOS
    
    print("Iniciando procesamiento de features acumuladas (NG y PC)...")
    df = df_original.copy()

    # --- 0. Generar columna APROBO si no existe ---
    df['APROBO'] = df['NOTA'].apply(lambda x: 'S' if x >= 11.5 else 'N')    
    
    # --- 1. Ordenar los datos (CRÍTICO para cálculos acumulados) ---
    df = df.sort_values(by=['COD_PERSONA', 'PER_MATRICULA'])

    # --- 2. Crear Columnas de Ayuda (a nivel de curso) ---
    valores_aprobados = ['APROBADO', 'S', 'Si'] 
    df['APROBO_CURSO_NUM'] = df['APROBO'].apply(lambda x: 1 if x in valores_aprobados else 0)
    df['CREDITOS_APROBADOS'] = df['CREDITOS'] * df['APROBO_CURSO_NUM']
    df['PONDERADO_CURSO'] = df['NOTA'] * df['CREDITOS']

    # --- 3. Calcular Métricas Acumuladas (por estudiante) ---
    g = df.groupby('COD_PERSONA')
    
    df['CUM_CREDITOS_LLEVADOS'] = g['CREDITOS'].cumsum()
    df['CUM_CURSOS_LLEVADOS'] = g['COD_CURSO'].cumcount() + 1
    df['CUM_CREDITOS_APROBADOS'] = g['CREDITOS_APROBADOS'].cumsum()
    df['CUM_CURSOS_APROBADOS'] = g['APROBO_CURSO_NUM'].cumsum()
    df['CUM_PONDERADO_TOTAL'] = g['PONDERADO_CURSO'].cumsum()

    # --- 4. Agregar a Nivel Semestre (MODIFICADO) ---
    # Ahora agregamos tanto las sumas específicas del ciclo (para PC)
    # como los valores acumulados (para NG).
    print("Agregando a nivel semestre...")
    df_semestre = df.groupby(['COD_PERSONA', 'PER_MATRICULA']).agg(
        # Para PROMEDIO_CICLO_PASADO (Sumas del ciclo actual)
        SUM_PONDERADO_CICLO=('PONDERADO_CURSO', 'sum'),
        SUM_CREDITOS_CICLO=('CREDITOS', 'sum'),
        
        # Para PROM_ACUMULADO (Valores acumulados al final del ciclo)
        CUM_CREDITOS_LLEVADOS=('CUM_CREDITOS_LLEVADOS', 'last'),
        CUM_CURSOS_LLEVADOS=('CUM_CURSOS_LLEVADOS', 'last'),
        CUM_CREDITOS_APROBADOS=('CUM_CREDITOS_APROBADOS', 'last'),
        CUM_CURSOS_APROBADOS=('CUM_CURSOS_APROBADOS', 'last'),
        CUM_PONDERADO_TOTAL=('CUM_PONDERADO_TOTAL', 'last'),
        
        # Metadata
        FECHA_NACIMIENTO=('FECHA_NACIMIENTO', 'last') 
    )

    # --- 5. Calcular Features Finales (a nivel de semestre) ---
    
    # a. Promedio acumulado (NG) (temporal, al final del ciclo)
    df_semestre['PROM_ACUMULADO'] = df_semestre['CUM_PONDERADO_TOTAL'] / df_semestre['CUM_CREDITOS_LLEVADOS']

    # b. Promedio del ciclo (PC) (temporal, al final del ciclo) (NUEVO)
    df_semestre['PROMEDIO_DEL_CICLO'] = df_semestre['SUM_PONDERADO_CICLO'] / df_semestre['SUM_CREDITOS_CICLO']

    # c. Porcentajes de avance (temporal, al final del ciclo)
    df_semestre['PCT_CREDITOS_APROBADOS'] = df_semestre['CUM_CREDITOS_APROBADOS'] / creditos_requeridos
    df_semestre['PCT_CURSOS_APROBADOS'] = df_semestre['CUM_CURSOS_APROBADOS'] / total_cursos
    df_semestre['PCT_CREDITOS_LLEVADOS'] = df_semestre['CUM_CREDITOS_LLEVADOS'] / creditos_requeridos
    df_semestre['PCT_CURSOS_LLEVADOS'] = df_semestre['CUM_CURSOS_LLEVADOS'] / total_cursos

    # d. Generar SEM_CURSADOS
    df_semestre['PER_MATRICULA_INT'] = df_semestre.index.get_level_values('PER_MATRICULA').str.replace('-', '').astype(int)
    df_semestre['SEM_CURSADOS'] = (
        df_semestre.groupby('COD_PERSONA')['PER_MATRICULA_INT']
        .rank(method='dense')
        .astype(int) - 1
    )


    # --- APLICAR SHIFT (MODIFICADO) ---
    # Aplicamos .shift(1) a TODAS las métricas (NG y PC)
    
    print("Aplicando shift para obtener datos del ciclo anterior...")
    
    cols_para_shift = [
        'PROM_ACUMULADO',
        'PROMEDIO_DEL_CICLO', # <-- AÑADIDO
        'PCT_CREDITOS_APROBADOS',
        'PCT_CURSOS_APROBADOS',
        'PCT_CREDITOS_LLEVADOS',
        'PCT_CURSOS_LLEVADOS'
    ]
    
    g_semestre = df_semestre.groupby('COD_PERSONA')
    
    for col in cols_para_shift:
        # Sobreescribimos la columna con su valor anterior
        df_semestre[col] = g_semestre[col].shift(1).fillna(0)

    print("Calculando feature de 'shock' (PC vs NG)...")
    df_semestre['DIF_PROM_SHOCK'] = df_semestre['PROMEDIO_DEL_CICLO'] - df_semestre['PROM_ACUMULADO']
    
    
    # e. Edad al momento del período (Se mantiene igual)
    periodo_str = df_semestre.index.get_level_values('PER_MATRICULA').astype(str)
    fechas_periodo = pd.to_datetime(
        periodo_str.str.replace('-01', '-03-01')
                       .str.replace('-02', '-08-01')
                       .str.replace('-00', '-01-01'),
        errors='coerce'
    )
    df_semestre['EDAD'] = (fechas_periodo - pd.to_datetime(df_semestre['FECHA_NACIMIENTO'], errors='coerce')).dt.days / 365.25

    # f. Ranking (basado en el promedio acumulado del ciclo anterior)
    df_semestre['PROM_ACUMULADO'] = df_semestre['PROM_ACUMULADO'].replace(0, np.nan)
    
    df_semestre['RANK_PCT'] = df_semestre.groupby('PER_MATRICULA') \
                                     ['PROM_ACUMULADO'].rank(pct=True, ascending=False)
    
    bins = [0, 0.1, 0.2, 1/3, 1.0]
    labels = ['10mo Superior', '5to Superior', '3cio Superior', 'General']
    
    df_semestre['RANKING'] = pd.cut(
        df_semestre['RANK_PCT'],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )
    df_semestre['RANKING'] = df_semestre['RANKING'].fillna('General')


    # --- 6. Limpieza Final ---
    df_semestre = df_semestre.reset_index()
    
    columnas_finales = [
        'COD_PERSONA', 
        'PER_MATRICULA',
        'PROM_ACUMULADO',
        'PROMEDIO_DEL_CICLO', # <-- Columna nueva añadida
        'DIF_PROM_SHOCK',     # <-- AÑADIDO
        'PCT_CREDITOS_APROBADOS',
        'PCT_CURSOS_APROBADOS',
        'PCT_CREDITOS_LLEVADOS',
        'PCT_CURSOS_LLEVADOS',
        'SEM_CURSADOS', 
        'EDAD',
        'RANKING'
    ]
    
    df_semestre_final = df_semestre[columnas_finales].copy()
    
    # Renombramos las columnas para claridad final
    df_semestre_final = df_semestre_final.rename(columns={
        'PROM_ACUMULADO': 'PROM_ACUMULADO_HIST', # NG
        'PROMEDIO_DEL_CICLO': 'PROMEDIO_CICLO_PASADO'             # PC
    })
    
    print(f"✅ Procesamiento completado. DataFrame final con {df_semestre_final.shape[0]} filas.")

    return df_semestre_final

def crear_features_acumuladas_3(df_original):
    """
    Crea un DataFrame de "snapshots" por estudiante y período.
    
    CONTIENE:
    1. Métricas HASTA EL CICLO ANTERIOR (NG, PC, Shock, Avance, Edad, Ranking).
    2. (NUEVO) Métricas de CARGA DEL CICLO ACTUAL (N_Cursos, N_Creditos, 
       N_Familia_*, N_Cluster_*).
    
    Argumentos:
    df_original -- El DataFrame 'df_clean' completo.
    
    Devuelve:
    df_semestre -- El DataFrame agregado por (COD_PERSONA, PER_MATRICULA).
    """
    creditos_requeridos = CREDITOS_REQUERIDOS
    total_cursos = TOTAL_CURSOS
    
    print("Iniciando procesamiento de features (Acumuladas + Carga Actual)...")
    df = df_original.copy()

    # --- 0. Generar columna APROBO si no existe ---
    df['APROBO'] = df['NOTA'].apply(lambda x: 'S' if x >= 11.5 else 'N')    
    
    # --- 1. Ordenar los datos ---
    df = df.sort_values(by=['COD_PERSONA', 'PER_MATRICULA'])

    # --- 2. Crear Columnas de Ayuda (a nivel de curso) ---
    valores_aprobados = ['APROBADO', 'S', 'Si'] 
    df['APROBO_CURSO_NUM'] = df['APROBO'].apply(lambda x: 1 if x in valores_aprobados else 0)
    df['CREDITOS_APROBADOS'] = df['CREDITOS'] * df['APROBO_CURSO_NUM']
    df['PONDERADO_CURSO'] = df['NOTA'] * df['CREDITOS']

    # --- 3. Calcular Métricas Acumuladas (por estudiante) ---
    g = df.groupby('COD_PERSONA')
    
    df['CUM_CREDITOS_LLEVADOS'] = g['CREDITOS'].cumsum()
    df['CUM_CURSOS_LLEVADOS'] = g['COD_CURSO'].cumcount() + 1
    df['CUM_CREDITOS_APROBADOS'] = g['CREDITOS_APROBADOS'].cumsum()
    df['CUM_CURSOS_APROBADOS'] = g['APROBO_CURSO_NUM'].cumsum()
    df['CUM_PONDERADO_TOTAL'] = g['PONDERADO_CURSO'].cumsum()

    # --- 3.5 (NUEVO) Crear Dummies para Carga Actual ---
    # Necesitamos saber cuántos cursos de cada tipo hay en el ciclo actual.
    print("Generando dummies para carga de Familia y Cluster...")
    # Asegurarse de que las columnas existan y sean categóricas
    if 'FAMILIA' not in df.columns or 'CLUSTER_DIFICULTAD' not in df.columns:
        raise ValueError("El DataFrame debe contener 'FAMILIA' y 'CLUSTER_DIFICULTAD'.")
        
    df['FAMILIA'] = df['FAMILIA'].astype(str)
    df['CLUSTER_DIFICULTAD'] = df['CLUSTER_DIFICULTAD'].astype(str)

    df_familia_dummies = pd.get_dummies(df['FAMILIA'], prefix='N_FAMILIA')
    df_cluster_dummies = pd.get_dummies(df['CLUSTER_DIFICULTAD'], prefix='N_CLUSTER')
    
    # Guardar los nombres de las nuevas columnas para la agregación
    familia_cols = df_familia_dummies.columns.tolist()
    cluster_cols = df_cluster_dummies.columns.tolist()
    
    # Unir los dummies al DataFrame principal
    df = pd.concat([df, df_familia_dummies, df_cluster_dummies], axis=1)

    # --- 4. Agregar a Nivel Semestre (MODIFICADO) ---
    print("Agregando a nivel semestre...")
    
    # Definimos el diccionario de agregación
    agg_dict = {
        # Métricas del Ciclo Actual (PC y Carga)
        'SUM_PONDERADO_CICLO': ('PONDERADO_CURSO', 'sum'),
        'SUM_CREDITOS_CICLO': ('CREDITOS', 'sum'),       # -> N_CREDITOS
        'N_CURSOS_CICLO': ('COD_CURSO', 'count'),          # -> N_CURSOS
        
        # Métricas Acumuladas (NG)
        'CUM_CREDITOS_LLEVADOS': ('CUM_CREDITOS_LLEVADOS', 'last'),
        'CUM_CURSOS_LLEVADOS': ('CUM_CURSOS_LLEVADOS', 'last'),
        'CUM_CREDITOS_APROBADOS': ('CUM_CREDITOS_APROBADOS', 'last'),
        'CUM_CURSOS_APROBADOS': ('CUM_CURSOS_APROBADOS', 'last'),
        'CUM_PONDERADO_TOTAL': ('CUM_PONDERADO_TOTAL', 'last'),
        
        'FECHA_NACIMIENTO': ('FECHA_NACIMIENTO', 'last') 
    }
    
    # Añadir dinámicamente las sumas de los dummies de Familia y Cluster
    for col in familia_cols + cluster_cols:
        agg_dict[col] = (col, 'sum')
        
    df_semestre = df.groupby(['COD_PERSONA', 'PER_MATRICULA']).agg(**agg_dict)

    # --- 5. Calcular Features Finales (Históricos y Actuales) ---
    
    # a. Promedios (NG y PC) (temporales, al final del ciclo)
    df_semestre['PROM_ACUMULADO'] = df_semestre['CUM_PONDERADO_TOTAL'] / df_semestre['CUM_CREDITOS_LLEVADOS']
    df_semestre['PROMEDIO_DEL_CICLO'] = df_semestre['SUM_PONDERADO_CICLO'] / df_semestre['SUM_CREDITOS_CICLO']

    # b. Porcentajes de avance (temporales)
    df_semestre['PCT_CREDITOS_APROBADOS'] = df_semestre['CUM_CREDITOS_APROBADOS'] / creditos_requeridos
    df_semestre['PCT_CURSOS_APROBADOS'] = df_semestre['CUM_CURSOS_APROBADOS'] / total_cursos
    df_semestre['PCT_CREDITOS_LLEVADOS'] = df_semestre['CUM_CREDITOS_LLEVADOS'] / creditos_requeridos
    df_semestre['PCT_CURSOS_LLEVADOS'] = df_semestre['CUM_CURSOS_LLEVADOS'] / total_cursos

    # c. Generar SEM_CURSADOS
    df_semestre['PER_MATRICULA_INT'] = df_semestre.index.get_level_values('PER_MATRICULA').str.replace('-', '').astype(int)
    df_semestre['SEM_CURSADOS'] = (
        df_semestre.groupby('COD_PERSONA')['PER_MATRICULA_INT']
        .rank(method='dense')
        .astype(int) - 1
    )

    # --- APLICAR SHIFT (Solo a features históricos) ---
    
    print("Aplicando shift para obtener datos del ciclo anterior (NG/PC)...")
    
    # NOTA: Las nuevas features (N_CURSOS_CICLO, SUM_CREDITOS_CICLO, N_FAMILIA_*)
    # NO se incluyen aquí, ya que representan el ciclo actual (T).
    cols_para_shift = [
        'PROM_ACUMULADO',
        'PROMEDIO_DEL_CICLO',
        'PCT_CREDITOS_APROBADOS',
        'PCT_CURSOS_APROBADOS',
        'PCT_CREDITOS_LLEVADOS',
        'PCT_CURSOS_LLEVADOS'
    ]
    
    g_semestre = df_semestre.groupby('COD_PERSONA')
    
    for col in cols_para_shift:
        df_semestre[col] = g_semestre[col].shift(1).fillna(0)

    print("Calculando feature de diferencia de carga (Shock de Créditos)...")

    # A. Obtener el Promedio Histórico de Créditos por Ciclo (AVG_CREDITOS_HIST)
    #    Fórmula: CUM_CREDITOS_LLEVADOS (Histórico NG) / SEM_CURSADOS (Histórico NG)
    
    # 1. Usar las sumas acumuladas *shifteadas* (que ya están en df_semestre)
    #    NOTA: Reemplazamos 0.0 en SEM_CURSADOS por 1 temporalmente para evitar división por cero.
    semestres_hist = df_semestre['SEM_CURSADOS'].replace(0, 1)
    
    # El CUM_CREDITOS_LLEVADOS usado aquí debe ser el valor POST-SHIFT, que representa el historial.
    df_semestre['AVG_CREDITOS_HIST'] = df_semestre['CUM_CREDITOS_LLEVADOS'] / semestres_hist

    # B. Calcular la Diferencia de Carga (Shock)
    #    Fórmula: (Créditos del Ciclo Actual) - (AVG Histórico)
    
    # 2. Las columnas de carga actual NO fueron shifteadas, son las sumas del ciclo T.
    # df_semestre['DIF_CREDITOS_ACTUAL_vs_HIST'] = (
    #     df_semestre['SUM_CREDITOS_CICLO'] - df_semestre['AVG_CREDITOS_HIST']
    # )
    
    # d. Calcular "Shock" (PC vs NG)
    print("Calculando feature de 'shock' (PC vs NG)...")
    df_semestre['DIF_PROM_SHOCK'] = df_semestre['PROMEDIO_DEL_CICLO'] - df_semestre['PROM_ACUMULADO']
    
    # e. Edad al momento del período (Se mantiene)
    periodo_str = df_semestre.index.get_level_values('PER_MATRICULA').astype(str)
    # ... (código de cálculo de EDAD) ...
    fechas_periodo = pd.to_datetime(
        periodo_str.str.replace('-01', '-03-01')
                       .str.replace('-02', '-08-01')
                       .str.replace('-00', '-01-01'),
        errors='coerce'
    )
    df_semestre['EDAD'] = (fechas_periodo - pd.to_datetime(df_semestre['FECHA_NACIMIENTO'], errors='coerce')).dt.days / 365.25

    # f. Ranking (Se mantiene)
    df_semestre['PROM_ACUMULADO'] = df_semestre['PROM_ACUMULADO'].replace(0, np.nan)
    df_semestre['RANK_PCT'] = df_semestre.groupby('PER_MATRICULA')['PROM_ACUMULADO'].rank(pct=True, ascending=False)
    # ... (código de bins y labels) ...
    bins = [0, 0.1, 0.2, 1/3, 1.0]
    labels = ['10mo Superior', '5to Superior', '3cio Superior', 'General']
    df_semestre['RANKING'] = pd.cut(df_semestre['RANK_PCT'], bins=bins, labels=labels, right=True, include_lowest=True)
    df_semestre['RANKING'] = df_semestre['RANKING'].fillna('General')


    # --- 6. Limpieza Final (MODIFICADO) ---
    df_semestre = df_semestre.reset_index()
    
    # Lista de features históricos y de contexto (los que ya tenías)
    columnas_historicas = [
        'COD_PERSONA', 
        'PER_MATRICULA',
        'PROM_ACUMULADO',
        'PROMEDIO_DEL_CICLO', 
        'DIF_PROM_SHOCK',
        'PCT_CREDITOS_APROBADOS',
        'PCT_CURSOS_APROBADOS',
        'PCT_CREDITOS_LLEVADOS',
        'PCT_CURSOS_LLEVADOS',
        'SEM_CURSADOS', 
        'EDAD',
        'RANKING'
    ]
    
    # Lista de features de Carga Actual (los nuevos)
    # (Usamos los nombres originales de la agregación)
    columnas_carga_actual = ['N_CURSOS_CICLO', 'SUM_CREDITOS_CICLO'] + familia_cols + cluster_cols
    
    # Combinar todas las columnas deseadas
    columnas_finales = columnas_historicas + columnas_carga_actual
    
    df_semestre_final = df_semestre[columnas_finales].copy()
    
    # Renombramos las columnas para claridad final
    df_semestre_final = df_semestre_final.rename(columns={
        'PROM_ACUMULADO': 'PROM_ACUMULADO_HIST', # NG
        'PROMEDIO_DEL_CICLO': 'PROMEDIO_CICLO_PASADO', # PC
        'N_CURSOS_CICLO': 'N_CURSOS_ACTUAL',
        'SUM_CREDITOS_CICLO': 'N_CREDITOS_ACTUAL'
    })
    
    print(f"✅ Procesamiento completado. DataFrame final con {df_semestre_final.shape[0]} filas y {df_semestre_final.shape[1]} columnas.")

    return df_semestre_final




def crear_features_acumuladas_3_5(df_original):
    """
    Crea un DataFrame de "snapshots" por estudiante y período.
    
    CONTIENE:
    1. Métricas HASTA EL CICLO ANTERIOR (NG, PC, Shock, Avance, Edad, Ranking).
    2. (NUEVO) Métricas de CARGA DEL CICLO ACTUAL (N_Cursos, N_Creditos, 
       N_Familia_*, N_Cluster_*).
    
    Argumentos:
    df_original -- El DataFrame 'df_clean' completo.
    
    Devuelve:
    df_semestre -- El DataFrame agregado por (COD_PERSONA, PER_MATRICULA).
    """
    creditos_requeridos = CREDITOS_REQUERIDOS
    total_cursos = TOTAL_CURSOS
    
    print("Iniciando procesamiento de features (Acumuladas + Carga Actual)...")
    df = df_original.copy()

    # --- 0. Generar columna APROBO si no existe ---
    df['APROBO'] = df['NOTA'].apply(lambda x: 'S' if x >= 11.5 else 'N')    
    
    # --- 1. Ordenar los datos ---
    df = df.sort_values(by=['COD_PERSONA', 'PER_MATRICULA'])

    # --- 2. Crear Columnas de Ayuda (a nivel de curso) ---
    valores_aprobados = ['APROBADO', 'S', 'Si'] 
    df['APROBO_CURSO_NUM'] = df['APROBO'].apply(lambda x: 1 if x in valores_aprobados else 0)
    df['CREDITOS_APROBADOS'] = df['CREDITOS'] * df['APROBO_CURSO_NUM']
    df['PONDERADO_CURSO'] = df['NOTA'] * df['CREDITOS']

    # --- 3. Calcular Métricas Acumuladas (por estudiante) ---
    g = df.groupby('COD_PERSONA')
    
    df['CUM_CREDITOS_LLEVADOS'] = g['CREDITOS'].cumsum()
    df['CUM_CURSOS_LLEVADOS'] = g['COD_CURSO'].cumcount() + 1
    df['CUM_CREDITOS_APROBADOS'] = g['CREDITOS_APROBADOS'].cumsum()
    df['CUM_CURSOS_APROBADOS'] = g['APROBO_CURSO_NUM'].cumsum()
    df['CUM_PONDERADO_TOTAL'] = g['PONDERADO_CURSO'].cumsum()

    # --- 3.5 (NUEVO) Crear Dummies para Carga Actual ---
    # Necesitamos saber cuántos cursos de cada tipo hay en el ciclo actual.
    print("Generando dummies para carga de Familia y Cluster...")
    # Asegurarse de que las columnas existan y sean categóricas
    if 'FAMILIA' not in df.columns or 'CLUSTER_DIFICULTAD' not in df.columns:
        raise ValueError("El DataFrame debe contener 'FAMILIA' y 'CLUSTER_DIFICULTAD'.")
        
    df['FAMILIA'] = df['FAMILIA'].astype(str)
    df['CLUSTER_DIFICULTAD'] = df['CLUSTER_DIFICULTAD'].astype(str)

    df_familia_dummies = pd.get_dummies(df['FAMILIA'], prefix='N_FAMILIA')
    df_cluster_dummies = pd.get_dummies(df['CLUSTER_DIFICULTAD'], prefix='N_CLUSTER')
    
    # Guardar los nombres de las nuevas columnas para la agregación
    familia_cols = df_familia_dummies.columns.tolist()
    cluster_cols = df_cluster_dummies.columns.tolist()
    
    # Unir los dummies al DataFrame principal
    df = pd.concat([df, df_familia_dummies, df_cluster_dummies], axis=1)

    # --- 4. Agregar a Nivel Semestre (MODIFICADO) ---
    print("Agregando a nivel semestre...")
    
    # Definimos el diccionario de agregación
    agg_dict = {
        # Métricas del Ciclo Actual (PC y Carga)
        'SUM_PONDERADO_CICLO': ('PONDERADO_CURSO', 'sum'),
        'SUM_CREDITOS_CICLO': ('CREDITOS', 'sum'),       # -> N_CREDITOS
        'N_CURSOS_CICLO': ('COD_CURSO', 'count'),          # -> N_CURSOS
        
        # Métricas Acumuladas (NG)
        'CUM_CREDITOS_LLEVADOS': ('CUM_CREDITOS_LLEVADOS', 'last'),
        'CUM_CURSOS_LLEVADOS': ('CUM_CURSOS_LLEVADOS', 'last'),
        'CUM_CREDITOS_APROBADOS': ('CUM_CREDITOS_APROBADOS', 'last'),
        'CUM_CURSOS_APROBADOS': ('CUM_CURSOS_APROBADOS', 'last'),
        'CUM_PONDERADO_TOTAL': ('CUM_PONDERADO_TOTAL', 'last'),
        
        'FECHA_NACIMIENTO': ('FECHA_NACIMIENTO', 'last') 
    }
    
    # Añadir dinámicamente las sumas de los dummies de Familia y Cluster
    for col in familia_cols + cluster_cols:
        agg_dict[col] = (col, 'sum')
        
    df_semestre = df.groupby(['COD_PERSONA', 'PER_MATRICULA']).agg(**agg_dict)

    # --- 5. Calcular Features Finales (Históricos y Actuales) ---
    
    # a. Promedios (NG y PC) (temporales, al final del ciclo)
    df_semestre['PROM_ACUMULADO'] = df_semestre['CUM_PONDERADO_TOTAL'] / df_semestre['CUM_CREDITOS_LLEVADOS']
    df_semestre['PROMEDIO_DEL_CICLO'] = df_semestre['SUM_PONDERADO_CICLO'] / df_semestre['SUM_CREDITOS_CICLO']

    # b. Porcentajes de avance (temporales)
    df_semestre['PCT_CREDITOS_APROBADOS'] = df_semestre['CUM_CREDITOS_APROBADOS'] / creditos_requeridos
    df_semestre['PCT_CURSOS_APROBADOS'] = df_semestre['CUM_CURSOS_APROBADOS'] / total_cursos
    df_semestre['PCT_CREDITOS_LLEVADOS'] = df_semestre['CUM_CREDITOS_LLEVADOS'] / creditos_requeridos
    df_semestre['PCT_CURSOS_LLEVADOS'] = df_semestre['CUM_CURSOS_LLEVADOS'] / total_cursos

    # c. Generar SEM_CURSADOS
    df_semestre['PER_MATRICULA_INT'] = df_semestre.index.get_level_values('PER_MATRICULA').str.replace('-', '').astype(int)
    df_semestre['SEM_CURSADOS'] = (
        df_semestre.groupby('COD_PERSONA')['PER_MATRICULA_INT']
        .rank(method='dense')
        .astype(int) - 1
    )


    # --- APLICAR SHIFT (Solo a features históricos) ---
    
    print("Aplicando shift para obtener datos del ciclo anterior (NG/PC)...")
    
    # NOTA: Las nuevas features (N_CURSOS_CICLO, SUM_CREDITOS_CICLO, N_FAMILIA_*)
    # NO se incluyen aquí, ya que representan el ciclo actual (T).
    cols_para_shift = [
        'PROM_ACUMULADO',
        'PROMEDIO_DEL_CICLO',
        'PCT_CREDITOS_APROBADOS',
        'PCT_CURSOS_APROBADOS',
        'PCT_CREDITOS_LLEVADOS',
        'PCT_CURSOS_LLEVADOS'
    ]
    
    g_semestre = df_semestre.groupby('COD_PERSONA')
    
    for col in cols_para_shift:
        df_semestre[col] = g_semestre[col].shift(1).fillna(0)

    print("Calculando feature de diferencia de carga (Shock de Créditos)...")

    # A. Obtener el Promedio Histórico de Créditos por Ciclo (AVG_CREDITOS_HIST)
    #    Fórmula: CUM_CREDITOS_LLEVADOS (Histórico NG) / SEM_CURSADOS (Histórico NG)
    
    # 1. Usar las sumas acumuladas *shifteadas* (que ya están en df_semestre)
    #    NOTA: Reemplazamos 0.0 en SEM_CURSADOS por 1 temporalmente para evitar división por cero.
    semestres_hist = df_semestre['SEM_CURSADOS'].replace(0, 1)
    
    # El CUM_CREDITOS_LLEVADOS usado aquí debe ser el valor POST-SHIFT, que representa el historial.
    df_semestre['AVG_CREDITOS_HIST'] = df_semestre['CUM_CREDITOS_LLEVADOS'] / semestres_hist

    # B. Calcular la Diferencia de Carga (Shock)
    #    Fórmula: (Créditos del Ciclo Actual) - (AVG Histórico)
    
    # 2. Las columnas de carga actual NO fueron shifteadas, son las sumas del ciclo T.
    df_semestre['DIF_CREDITOS_ACTUAL_vs_HIST'] = (
        df_semestre['SUM_CREDITOS_CICLO'] - df_semestre['AVG_CREDITOS_HIST']
    )
    
    # d. Calcular "Shock" (PC vs NG)
    print("Calculando feature de 'shock' (PC vs NG)...")
    df_semestre['DIF_PROM_SHOCK'] = df_semestre['PROMEDIO_DEL_CICLO'] - df_semestre['PROM_ACUMULADO']
    
    # e. Edad al momento del período (Se mantiene)
    periodo_str = df_semestre.index.get_level_values('PER_MATRICULA').astype(str)
    # ... (código de cálculo de EDAD) ...
    fechas_periodo = pd.to_datetime(
        periodo_str.str.replace('-01', '-03-01')
                       .str.replace('-02', '-08-01')
                       .str.replace('-00', '-01-01'),
        errors='coerce'
    )
    df_semestre['EDAD'] = (fechas_periodo - pd.to_datetime(df_semestre['FECHA_NACIMIENTO'], errors='coerce')).dt.days / 365.25

    # f. Ranking (Se mantiene)
    df_semestre['PROM_ACUMULADO'] = df_semestre['PROM_ACUMULADO'].replace(0, np.nan)
    df_semestre['RANK_PCT'] = df_semestre.groupby('PER_MATRICULA')['PROM_ACUMULADO'].rank(pct=True, ascending=False)
    # ... (código de bins y labels) ...
    bins = [0, 0.1, 0.2, 1/3, 1.0]
    labels = ['10mo Superior', '5to Superior', '3cio Superior', 'General']
    df_semestre['RANKING'] = pd.cut(df_semestre['RANK_PCT'], bins=bins, labels=labels, right=True, include_lowest=True)
    df_semestre['RANKING'] = df_semestre['RANKING'].fillna('General')


    # --- 6. Limpieza Final (MODIFICADO) ---
    df_semestre = df_semestre.reset_index()
    
    # Lista de features históricos y de contexto (los que ya tenías)
    columnas_historicas = [
        'COD_PERSONA', 
        'PER_MATRICULA',
        'PROM_ACUMULADO',
        'PROMEDIO_DEL_CICLO', 
        'DIF_PROM_SHOCK',
        'PCT_CREDITOS_APROBADOS',
        'PCT_CURSOS_APROBADOS',
        'PCT_CREDITOS_LLEVADOS',
        'PCT_CURSOS_LLEVADOS',
        'SEM_CURSADOS', 
        'EDAD',
        'RANKING'
    ]
    
    # Lista de features de Carga Actual (los nuevos)
    # (Usamos los nombres originales de la agregación)
    columnas_carga_actual = ['N_CURSOS_CICLO', 'SUM_CREDITOS_CICLO' ,'DIF_CREDITOS_ACTUAL_vs_HIST'] + familia_cols + cluster_cols
    
    # Combinar todas las columnas deseadas
    columnas_finales = columnas_historicas + columnas_carga_actual
    
    df_semestre_final = df_semestre[columnas_finales].copy()
    
    # Renombramos las columnas para claridad final
    df_semestre_final = df_semestre_final.rename(columns={
        'PROM_ACUMULADO': 'PROM_ACUMULADO_HIST', # NG
        'PROMEDIO_DEL_CICLO': 'PROMEDIO_CICLO_PASADO', # PC
        'N_CURSOS_CICLO': 'N_CURSOS_ACTUAL',
        'SUM_CREDITOS_CICLO': 'N_CREDITOS_ACTUAL'
    })
    
    print(f"✅ Procesamiento completado. DataFrame final con {df_semestre_final.shape[0]} filas y {df_semestre_final.shape[1]} columnas.")

    return df_semestre_final





def crear_features_acumuladas_4(df_original):
    """
    Crea un DataFrame de "snapshots" por estudiante y período.
    
    CONTIENE:
    1. Métricas HASTA EL CICLO ANTERIOR (NG, PC, Shock, Avance, Edad, Ranking).
    2. Métricas de Carga del Ciclo Actual.
    3. (NUEVO) Desviación Estándar (STD) Histórica y de Ciclo Pasado.
    
    Argumentos:
    df_original -- El DataFrame 'df_clean' completo.
    
    Devuelve:
    df_semestre -- El DataFrame agregado por (COD_PERSONA, PER_MATRICULA).
    """
    creditos_requeridos = CREDITOS_REQUERIDOS
    total_cursos = TOTAL_CURSOS
    
    print("Iniciando procesamiento de features (Acumuladas + Carga Actual + Volatilidad STD)...")
    df = df_original.copy()

    # --- 0. Generar columna APROBO si no existe ---
    df['APROBO'] = df['NOTA'].apply(lambda x: 'S' if x >= 11.5 else 'N')    
    
    # --- 1. Ordenar los datos ---
    df = df.sort_values(by=['COD_PERSONA', 'PER_MATRICULA'])

    # --- 2. Crear Columnas de Ayuda (a nivel de curso) ---
    valores_aprobados = ['APROBADO', 'S', 'Si'] 
    df['APROBO_CURSO_NUM'] = df['APROBO'].apply(lambda x: 1 if x in valores_aprobados else 0)
    df['CREDITOS_APROBADOS'] = df['CREDITOS'] * df['APROBO_CURSO_NUM']
    df['PONDERADO_CURSO'] = df['NOTA'] * df['CREDITOS']

    # --- 3. Calcular Métricas Acumuladas (por estudiante) ---
    g = df.groupby('COD_PERSONA')
    
    df['CUM_CREDITOS_LLEVADOS'] = g['CREDITOS'].cumsum()
    df['CUM_CURSOS_LLEVADOS'] = g['COD_CURSO'].cumcount() + 1
    df['CUM_CREDITOS_APROBADOS'] = g['CREDITOS_APROBADOS'].cumsum()
    df['CUM_CURSOS_APROBADOS'] = g['APROBO_CURSO_NUM'].cumsum()
    df['CUM_PONDERADO_TOTAL'] = g['PONDERADO_CURSO'].cumsum()
    
    # NUEVO: Desviación Estándar de TODAS las notas HASTA ese curso (NG)
    df['STD_NOTAS_ACUMULADA'] = g['NOTA'].expanding().std().reset_index(level=0, drop=True)


    # --- 3.5 (NUEVO) Crear Dummies para Carga Actual ---
    print("Generando dummies para carga de Familia y Cluster...")
    if 'FAMILIA' not in df.columns or 'CLUSTER_DIFICULTAD' not in df.columns:
        raise ValueError("El DataFrame debe contener 'FAMILIA' y 'CLUSTER_DIFICULTAD'.")
        
    df['FAMILIA'] = df['FAMILIA'].astype(str)
    df['CLUSTER_DIFICULTAD'] = df['CLUSTER_DIFICULTAD'].astype(str)

    df_familia_dummies = pd.get_dummies(df['FAMILIA'], prefix='N_FAMILIA')
    df_cluster_dummies = pd.get_dummies(df['CLUSTER_DIFICULTAD'], prefix='N_CLUSTER')
    
    familia_cols = df_familia_dummies.columns.tolist()
    cluster_cols = df_cluster_dummies.columns.tolist()
    
    df = pd.concat([df, df_familia_dummies, df_cluster_dummies], axis=1)

    # --- 4. Agregar a Nivel Semestre (MODIFICADO para STD del ciclo PC) ---
    print("Agregando a nivel semestre (incluyendo STD)...")
    
    agg_dict = {
        # Métricas del Ciclo Actual (PC y Carga)
        'SUM_PONDERADO_CICLO': ('PONDERADO_CURSO', 'sum'),
        'SUM_CREDITOS_CICLO': ('CREDITOS', 'sum'),
        'N_CURSOS_CICLO': ('COD_CURSO', 'count'),
        'STD_CICLO': ('NOTA', 'std'), # <<< NUEVO: STD de las notas SOLO en este ciclo
        
        # Métricas Acumuladas (NG)
        'CUM_CREDITOS_LLEVADOS': ('CUM_CREDITOS_LLEVADOS', 'last'),
        'CUM_PONDERADO_TOTAL': ('CUM_PONDERADO_TOTAL', 'last'),
        'STD_NOTAS_ACUMULADA': ('STD_NOTAS_ACUMULADA', 'last'), # <<< NUEVO: STD acumulada (NG)
        
        # Avance y Metadata
        'CUM_CURSOS_LLEVADOS': ('CUM_CURSOS_LLEVADOS', 'last'),
        'CUM_CREDITOS_APROBADOS': ('CUM_CREDITOS_APROBADOS', 'last'),
        'CUM_CURSOS_APROBADOS': ('CUM_CURSOS_APROBADOS', 'last'),
        'FECHA_NACIMIENTO': ('FECHA_NACIMIENTO', 'last') 
    }
    
    for col in familia_cols + cluster_cols:
        agg_dict[col] = (col, 'sum')
        
    df_semestre = df.groupby(['COD_PERSONA', 'PER_MATRICULA']).agg(**agg_dict)


    # --- 5. Calcular Features Finales y Aplicar SHIFT ---
    
    # a. Promedios y Avance (al final del ciclo T)
    df_semestre['PROM_ACUMULADO'] = df_semestre['CUM_PONDERADO_TOTAL'] / df_semestre['CUM_CREDITOS_LLEVADOS']
    df_semestre['PROMEDIO_DEL_CICLO'] = df_semestre['SUM_PONDERADO_CICLO'] / df_semestre['SUM_CREDITOS_CICLO']

    df_semestre['PCT_CREDITOS_APROBADOS'] = df_semestre['CUM_CREDITOS_APROBADOS'] / creditos_requeridos
    df_semestre['PCT_CURSOS_APROBADOS'] = df_semestre['CUM_CURSOS_APROBADOS'] / total_cursos
    df_semestre['PCT_CREDITOS_LLEVADOS'] = df_semestre['CUM_CREDITOS_LLEVADOS'] / creditos_requeridos
    df_semestre['PCT_CURSOS_LLEVADOS'] = df_semestre['CUM_CURSOS_LLEVADOS'] / total_cursos

    # b. Generar SEM_CURSADOS
    df_semestre['PER_MATRICULA_INT'] = df_semestre.index.get_level_values('PER_MATRICULA').str.replace('-', '').astype(int)
    df_semestre['SEM_CURSADOS'] = (df_semestre.groupby('COD_PERSONA')['PER_MATRICULA_INT'].rank(method='dense').astype(int) - 1)

    # --- APLICAR SHIFT (Prevención de Fuga de Datos) ---
    print("Aplicando shift para obtener datos del ciclo anterior (NG/PC/STD)...")
    
    cols_para_shift = [
        'PROM_ACUMULADO',
        'PROMEDIO_DEL_CICLO', 
        'STD_NOTAS_ACUMULADA',      # <<< NUEVO: Volatilidad NG
        'STD_CICLO',                # <<< NUEVO: Volatilidad PC
        'PCT_CREDITOS_APROBADOS',
        'PCT_CURSOS_APROBADOS',
        'PCT_CREDITOS_LLEVADOS',
        'PCT_CURSOS_LLEVADOS'
    ]
    
    g_semestre = df_semestre.groupby('COD_PERSONA')
    
    for col in cols_para_shift:
        df_semestre[col] = g_semestre[col].shift(1).fillna(0) # fillna(0) si no hay historial
    
    
    # c. Calcular "Shock" (PC vs NG y Shock de Carga)
    print("Calculando features de 'shock' y 'disparidad'...")
    df_semestre['DIF_PROM_SHOCK'] = df_semestre['PROMEDIO_DEL_CICLO'] - df_semestre['PROM_ACUMULADO']

    semestres_hist = df_semestre['SEM_CURSADOS'].replace(0, 1)
    df_semestre['AVG_CREDITOS_HIST'] = df_semestre['CUM_CREDITOS_LLEVADOS'] / semestres_hist
    df_semestre['DIF_CREDITOS_ACTUAL_vs_HIST'] = df_semestre['SUM_CREDITOS_CICLO'] - df_semestre['AVG_CREDITOS_HIST']

    # d. Edad y Ranking (Se mantienen)
    periodo_str = df_semestre.index.get_level_values('PER_MATRICULA').astype(str)
    fechas_periodo = pd.to_datetime(
        periodo_str.str.replace('-01', '-03-01').str.replace('-02', '-08-01').str.replace('-00', '-01-01'),
        errors='coerce'
    )
    df_semestre['EDAD'] = (fechas_periodo - pd.to_datetime(df_semestre['FECHA_NACIMIENTO'], errors='coerce')).dt.days / 365.25

    df_semestre['PROM_ACUMULADO'] = df_semestre['PROM_ACUMULADO'].replace(0, np.nan)
    df_semestre['RANK_PCT'] = df_semestre.groupby('PER_MATRICULA')['PROM_ACUMULADO'].rank(pct=True, ascending=False)
    bins = [0, 0.1, 0.2, 1/3, 1.0]
    labels = ['10mo Superior', '5to Superior', '3cio Superior', 'General']
    df_semestre['RANKING'] = pd.cut(df_semestre['RANK_PCT'], bins=bins, labels=labels, right=True, include_lowest=True)
    df_semestre['RANKING'] = df_semestre['RANKING'].fillna('General')


    # --- 6. Limpieza Final (Añadir STD a las columnas finales) ---
    df_semestre = df_semestre.reset_index()
    
    columnas_historicas = [
        'COD_PERSONA', 'PER_MATRICULA', 'PROM_ACUMULADO', 'PROMEDIO_DEL_CICLO', 
        'STD_NOTAS_ACUMULADA',      # <<< NUEVA STD NG
        'STD_CICLO',                # <<< NUEVA STD PC
        'DIF_PROM_SHOCK', 'DIF_CREDITOS_ACTUAL_vs_HIST', 'AVG_CREDITOS_HIST',
        'PCT_CREDITOS_APROBADOS', 'PCT_CURSOS_APROBADOS', 'PCT_CREDITOS_LLEVADOS', 
        'PCT_CURSOS_LLEVADOS', 'SEM_CURSADOS', 'EDAD', 'RANKING'
    ]
    
    columnas_carga_actual = ['N_CURSOS_CICLO', 'SUM_CREDITOS_CICLO'] + familia_cols + cluster_cols
    columnas_finales = columnas_historicas + columnas_carga_actual
    
    df_semestre_final = df_semestre[columnas_finales].copy()
    
    # Renombrar para claridad final
    df_semestre_final = df_semestre_final.rename(columns={
        'PROM_ACUMULADO': 'PROM_ACUMULADO_HIST', 
        'PROMEDIO_DEL_CICLO': 'PROMEDIO_CICLO_PASADO',
        'DIF_PROM_SHOCK': 'DIF_PROM_SHOCK_PC_vs_NG',
        'N_CURSOS_CICLO': 'N_CURSOS_ACTUAL',
        'SUM_CREDITOS_CICLO': 'N_CREDITOS_ACTUAL',
        'STD_NOTAS_ACUMULADA': 'STD_NOTAS_ACUMULADA_HIST',
        'STD_CICLO': 'STD_CICLO_PASADO'
    })
    
    print(f"✅ Procesamiento completado. DataFrame final con {df_semestre_final.shape[0]} filas y {df_semestre_final.shape[1]} columnas.")

    return df_semestre_final