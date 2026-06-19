import os
import fitz  # PyMuPDF
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extraer_texto_pdf(ruta_pdf):
    texto = ""
    try:
        doc = fitz.open(ruta_pdf)
        for pagina in doc:
            texto += pagina.get_text()
        doc.close()
    except Exception as e:
        print(f"Error al leer PDF {ruta_pdf}: {e}")
    return texto

def calcular_pesos_zona_verde(path_corpus_sociedad):
    ruta = Path(path_corpus_sociedad)
    print(f"\n[INFO] Accediendo a la subcarpeta: {ruta.name}")
    
    if not ruta.exists():
        print(f"!!! ERROR: No se encuentra la carpeta en: {ruta.absolute()}")
        return None

    # 1. DICCIONARIO BIDIMENSIONAL (PODA QUIRÚRGICA PARA DOMINANCIA C1S)
    criterios_sociedad = {
        'C1S': {
            # Se inyecta vocabulario financiero y transaccional duro del día a día
            'positivo': "ahorro barato económico exención rentabilidad beneficio descuento reducción costos gratis subsidio financiamiento asequible inversión precio valor dinero mercado presupuesto ingreso ganancia capital liquidez incentivo peso pesos dólar clp uf comprar compra venta pago",
            'negativo': "caro costoso impuesto elevado inalcanzable deuda barrera gasto inflación mantención patente cobro tarifa excesivo mantenimiento pérdida riesgo cuota crédito interés peaje pagar cobrar"
        },
        'C2S': {
            'positivo': "autonomía rápida cargador disponible enchufe eficiente interoperable seguro electrolinera conector voltaje amperaje rendimiento",
            'negativo': "pana falla lento escaso degradación muerta avería incompatible desgaste recalentamiento"
        },
        'C3S': {
            # Se eliminaron las palabras trampa (futuro, impacto, salud, público, aire)
            # Solo quedan términos técnicos estrictos de ecología
            'positivo': "ecológico sustentable emisiones verde descarbonización silencioso renovable",
            'negativo': "contaminación humo litio minero desecho huella carbono toxicidad chatarra calentamiento"
        }
    }

    # 2. Recopilar texto de los PDFs
    texto_corpus = ""
    archivos = list(ruta.glob("*.pdf"))
    
    if not archivos:
        print(f"!!! ERROR: No se encontraron archivos PDF dentro de {ruta.name}")
        return None

    for archivo in archivos:
        print(f"Analizando opinión/dato social: {archivo.name}...")
        texto_corpus += extraer_texto_pdf(str(archivo))

    if len(texto_corpus.strip()) < 100:
        print("!!! ERROR: El contenido de la Sociedad es insuficiente.")
        return None

    # 3. Cálculo de Similitud Coseno Bidimensional (TF-IDF)
    resultados_brutos = {}
    desglose_polaridad = {}
    tfidf_vec = TfidfVectorizer()

    for clave, dimensiones in criterios_sociedad.items():
        tfidf_matrix_pos = tfidf_vec.fit_transform([texto_corpus, dimensiones['positivo']])
        sim_pos = cosine_similarity(tfidf_matrix_pos[0:1], tfidf_matrix_pos[1:2])[0][0]
        
        tfidf_matrix_neg = tfidf_vec.fit_transform([texto_corpus, dimensiones['negativo']])
        sim_neg = cosine_similarity(tfidf_matrix_neg[0:1], tfidf_matrix_neg[1:2])[0][0]
        
        importancia_total = sim_pos + sim_neg
        resultados_brutos[clave[:3]] = importancia_total 
        desglose_polaridad[clave] = {'Positivo': sim_pos, 'Negativo': sim_neg, 'Total': importancia_total}

    # 4. Normalización Estocástica (Suma = 1.0) para la columna OBJ2
    total_general = sum(resultados_brutos.values())
    if total_general == 0:
        print("!!! ERROR: No hubo coincidencias léxicas con el corpus.")
        return None
        
    pesos_verdes = {k: round(v / total_general, 4) for k, v in resultados_brutos.items()}

    # 5. Despliegue de Resultados Estilo IEEE
    print("\n" + "="*60)
    print(" ANÁLISIS LÉXICO BIDIMENSIONAL - SOCIEDAD (ZONA VERDE)")
    print("="*60)
    for crit, valores in desglose_polaridad.items():
        print(f"[{crit}]")
        print(f"  ├─ Frecuencia Positiva : {valores['Positivo']:.4f}")
        print(f"  ├─ Frecuencia Negativa : {valores['Negativo']:.4f}")
        print(f"  └─ Impacto Bruto       : {valores['Total']:.4f}\n")
    
    print("="*60)
    print(" VECTOR PRIORIDAD NORMALIZADO (COLUMNA OBJ2)")
    print("="*60)
    for k, v in pesos_verdes.items():
        print(f"   {k}: {v}")
    print("="*60)
        
    return pesos_verdes

if __name__ == "__main__":
    PATH_SOCIEDAD = "C:/Users/56966/OneDrive - mail.pucv.cl/Documentos/Tesis_Electromovilidad/Corpus_PDFs/Sociedad"
    vector_obj2 = calcular_pesos_zona_verde(PATH_SOCIEDAD)