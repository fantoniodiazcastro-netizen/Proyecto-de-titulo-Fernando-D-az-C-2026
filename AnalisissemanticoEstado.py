import os
import fitz  # PyMuPDF
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

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

def calcular_pesos_zona_roja(path_corpus_estado, ruta_tulio_local):
    ruta = Path(path_corpus_estado)
    print(f"\n[INFO] Accediendo a la subcarpeta: {ruta.name}")
    
    if not ruta.exists():
        print(f"!!! ERROR: No se encuentra la carpeta de PDFs en: {ruta.absolute()}")
        return None

    # 1. CARGA DEL NUEVO TULIO LOCAL (CON SEGURO PARA EL TOKENIZADOR)
    print("Cargando el nuevo motor local de Tulio (Safetensors)...")
    try:
        # El parámetro use_fast=False previene el error de dependencias de sentencepiece
        tokenizer = AutoTokenizer.from_pretrained(ruta_tulio_local, use_fast=False)
        model_tulio = AutoModel.from_pretrained(ruta_tulio_local, use_safetensors=True)
        print("✓ Tulio cargado exitosamente.")
    except Exception as e:
        print(f"!!! ERROR al cargar el modelo desde {ruta_tulio_local}: {e}")
        return None

    # 2. Definición de Criterios
    criterios = {
        'C1E': "Económico: inversión, presupuesto, subsidios, financiamiento y costos estatales",
        'C2E': "Técnico: infraestructura de carga, normativa RIC, estándares y red eléctrica",
        'C3E': "Ambiental: emisiones de CO2, descarbonización, sustentabilidad y metas climáticas"
    }

    # 3. Recopilar texto de los PDFs
    texto_corpus = ""
    archivos = list(ruta.glob("*.pdf"))
    
    if not archivos:
        print(f"!!! ERROR: No se encontraron archivos PDF dentro de {ruta.name}")
        return None

    for archivo in archivos:
        print(f"Analizando documento: {archivo.name}...")
        texto_corpus += extraer_texto_pdf(str(archivo))

    if len(texto_corpus.strip()) < 100:
        print("!!! ERROR: El contenido extraído es insuficiente para un análisis fiable.")
        return None

    # 4. Cálculo de Similitud Coseno Matemática
    resultados_similitud = {}
    
    # Vectorizar el corpus completo
    inputs_corpus = tokenizer(texto_corpus, max_length=512, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out_corpus = model_tulio(**inputs_corpus)
        
    mask_corp = inputs_corpus['attention_mask'].unsqueeze(-1).expand(out_corpus.last_hidden_state.size()).float()
    emb_corpus = torch.sum(out_corpus.last_hidden_state * mask_corp, 1) / torch.clamp(mask_corp.sum(1), min=1e-9)
    vec_corpus = emb_corpus[0].numpy()
    norm_corpus = np.linalg.norm(vec_corpus)
    
    for clave, descripcion in criterios.items():
        inputs_crit = tokenizer(descripcion, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out_crit = model_tulio(**inputs_crit)
            
        mask_crit = inputs_crit['attention_mask'].unsqueeze(-1).expand(out_crit.last_hidden_state.size()).float()
        emb_crit = torch.sum(out_crit.last_hidden_state * mask_crit, 1) / torch.clamp(mask_crit.sum(1), min=1e-9)
        vec_crit = emb_crit[0].numpy()
        
        # Operación de Similitud de Coseno exacta
        dot_product = np.dot(vec_corpus, vec_crit)
        norm_crit = np.linalg.norm(vec_crit)
        sim = dot_product / (norm_corpus * norm_crit + 1e-12)
        
        resultados_similitud[clave] = max(0.0, float(sim))

    # 5. Normalización para la columna de la matriz ANP
    total = sum(resultados_similitud.values())
    pesos_rojos = {k: round(v / total, 4) for k, v in resultados_similitud.items()}

    print("\n" + "="*45)
    print(" RESULTADOS FINALES ZONA ROJA (COLUMNA OBJ1)")
    print("="*45)
    for k, v in pesos_rojos.items():
        print(f"   {k}: {v}")
    print("="*45)
        
    return pesos_rojos

if __name__ == "__main__":
    # RUTA DE LOS PDFs (Intacta, tal como la tenías)
    PATH_ESTADO = "C:/Users/56966/OneDrive - mail.pucv.cl/Documentos/Tesis_Electromovilidad/Corpus_PDFs/Estado"
    
    # RUTA DEL MODELO: Por favor, cambia "NOMBRE_DE_LA_CARPETA" por el nombre real de tu carpeta en el escritorio
    PATH_TULIO_LOCAL = r"C:\Users\56966\OneDrive - mail.pucv.cl\Escritorio\modelo_tulio_local"
    
    if not os.path.exists(PATH_TULIO_LOCAL):
        print(f"\n[ATENCIÓN] No se encuentra la carpeta del modelo en:\n{PATH_TULIO_LOCAL}")
        print("Por favor, edita la línea 85 del código y pon el nombre exacto de la carpeta de Tulio.")
    else:
        vector_obj1 = calcular_pesos_zona_roja(PATH_ESTADO, PATH_TULIO_LOCAL)