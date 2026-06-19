import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ---> PRUEBA DE VIDA PARA VS CODE
print("\n[SISTEMA] Iniciando ejecución del script de Cadenas de Markov...")

# =====================================================================
# 1. NODOS Y ETIQUETAS
# =====================================================================
nodos = [
    "OBJ1", "C1EE", "C2ET", "C3EA", "A1E", "A2E", "A3E",
    "OBJ2", "C1SE", "C2ST", "C3SA", "A1S", "A2S"
]

etiquetas = {
    "A1E": "Infraestructura (E)", "A2E": "Incentivos (E)", "A3E": "Promoción (E)",
    "A1S": "VEE (Eléctrico)", "A2S": "VCI (Combustión)"
}

# =====================================================================
# 2. SUPERMATRIZ NO PONDERADA
# =====================================================================
raw = np.array([
#  OBJ1    C1EE    C2ET    C3EA    A1E   A2E   A3E   OBJ2    C1SE   C2ST   C3SA   A1S   A2S
[1.0000, 0.2534, 0.3335, 0.4131, 0.00, 0.00, 0.00, 0.0000, 0.000, 0.000, 0.00, 0.00, 0.00], # OBJ1
[0.2534, 1.0000, 0.0000, 0.0000, 0.00, 0.00, 0.00, 0.0000, 0.000, 0.000, 0.00, 0.00, 0.00], # C1EE
[0.3335, 0.0000, 1.0000, 0.0000, 0.00, 0.00, 0.00, 0.0000, 0.000, 0.000, 0.00, 0.00, 0.00], # C2ET
[0.4131, 0.0000, 0.0000, 1.0000, 0.00, 0.00, 0.00, 0.0000, 0.000, 0.000, 0.00, 0.00, 0.00], # C3EA

[0.0000, 0.6558, 1.0000, 0.1243, 1.00, 0.00, 0.00, 0.0000, 0.000, 1.000, 0.33, 0.33, 0.00], # A1E
[0.0000, 0.2301, 0.0000, 0.3193, 0.00, 1.00, 0.00, 0.0000, 1.000, 0.000, 0.33, 0.33, 0.00], # A2E
[0.0000, 0.1140, 0.0000, 0.5562, 0.00, 0.00, 1.00, 0.0000, 0.000, 0.000, 0.33, 0.33, 0.00], # A3E

[0.0000, 0.0000, 0.0000, 0.0000, 0.00, 0.00, 0.00, 1.0000, 0.4421, 0.184, 0.37, 0.00, 0.00], # OBJ2
[0.0000, 0.0000, 0.0000, 0.0000, 0.00, 1.00, 0.33, 0.4420, 1.000, 0.000, 0.00, 0.00, 0.00], # C1SE
[0.0000, 0.0000, 0.0000, 0.0000, 1.00, 0.00, 0.33, 0.1840, 0.000, 1.000, 0.00, 0.00, 0.00], # C2ST
[0.0000, 0.0000, 0.0000, 0.0000, 0.00, 0.00, 0.33, 0.3740, 0.000, 0.000, 1.00, 0.00, 0.00], # C3SA

# ---> Evaluación de Alternativas de Mercado
[0.0000, 0.0000, 0.0000, 0.0000, 0.05, 0.10, 0.05, 0.0000, 0.300, 0.600, 1.00, 1.00, 0.00], # A1S (VEE)
[0.0000, 0.0000, 0.0000, 0.0000, 0.95, 0.90, 0.95, 0.0000, 0.700, 0.400, 0.00, 0.00, 1.00]  # A2S (VCI)
])

# =====================================================================
# 3. NORMALIZACIÓN Y FACTOR DE AMORTIGUAMIENTO
# =====================================================================
W = raw / (raw.sum(axis=0) + 1e-12)
damping = 0.85 
Wp = damping * W + (1 - damping) * (1 / len(nodos))

# Algoritmo de Cadenas de Markov
def get_limit_matrix(matrix, max_iter=500, tol=1e-8):
    M = matrix.copy()
    for i in range(1, max_iter + 1):
        next_M = np.dot(M, matrix)
        if np.allclose(M, next_M, atol=tol): 
            print(f"\n[SISTEMA IEEE] ✓ Convergencia de la Supermatriz alcanzada en la iteración N° {i}")
            return next_M
        M = next_M
    print(f"\n[ADVERTENCIA] Límite de iteraciones ({max_iter}) alcanzado sin convergencia perfecta.")
    return M

W_limit = get_limit_matrix(Wp)
prio_global = W_limit[:, 0]

# =====================================================================
# 4. EXTRACCIÓN DE RESULTADOS Y RANKINGS
# =====================================================================
def get_actor_ranking(names):
    idxs = [nodos.index(n) for n in names]
    vals = prio_global[idxs]
    return pd.DataFrame({'Nodo': [etiquetas[n] for n in names], 'Valor': vals / vals.sum()})

df_estado = get_actor_ranking(["A1E", "A2E", "A3E"])
df_sociedad = get_actor_ranking(["A1S", "A2S"])

# =====================================================================
# 5. IMPRESIÓN EN CONSOLA Y GUARDADO DEL DASHBOARD
# =====================================================================
print("\n" + "="*90)
print(" S U P E R M A T R I Z   L Í M I T E   ( W ∞ )   -   E S C E N A R I O   B A S E ")
print("="*90)
df_limite = pd.DataFrame(W_limit, index=nodos, columns=nodos)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df_limite.round(4).to_string())
print("="*90)

plt.style.use('dark_background')
fig = plt.figure(figsize=(24, 18), facecolor="#0B0F19")
gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

ax0 = fig.add_subplot(gs[0, 0])
im0 = ax0.imshow(W, cmap="viridis", aspect='auto')
ax0.set_title("1. Supermatriz Ponderada ($W$)", fontsize=18, pad=20, fontweight='bold')
ax0.set_xticks(range(len(nodos))); ax0.set_yticks(range(len(nodos)))
ax0.set_xticklabels(nodos, rotation=90); ax0.set_yticklabels(nodos)
plt.colorbar(im0, ax=ax0)

ax1 = fig.add_subplot(gs[0, 1])
im1 = ax1.imshow(W_limit, cmap="magma", aspect='auto')
ax1.set_title(r"2. Supermatriz Límite ($W^{\infty}$)", fontsize=18, pad=20, fontweight='bold')
ax1.set_xticks(range(len(nodos))); ax1.set_yticks(range(len(nodos)))
ax1.set_xticklabels(nodos, rotation=90); ax1.set_yticklabels(nodos)
plt.colorbar(im1, ax=ax1)

gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :])
ax2 = fig.add_subplot(gs_bottom[0, 0])
ax3 = fig.add_subplot(gs_bottom[0, 1])

bars_e = ax2.barh(df_estado['Nodo'], df_estado['Valor'], color="#3B82F6", edgecolor='white')
ax2.set_title("Prioridades del Estado (Suma 100%)", fontsize=16)
ax2.set_xlim(0, 1.2); ax2.invert_yaxis()
for b in bars_e:
    ax2.text(b.get_width()+0.02, b.get_y()+b.get_height()/2, f'{b.get_width()*100:.2f}%', va='center', fontweight='bold', size=14)

bars_s = ax3.barh(df_sociedad['Nodo'], df_sociedad['Valor'], color="#10B981", edgecolor='white')
ax3.set_title("Preferencia de la Sociedad (Suma 100%)", fontsize=16)
ax3.set_xlim(0, 1.2); ax3.invert_yaxis()
for b in bars_s:
    ax3.text(b.get_width()+0.02, b.get_y()+b.get_height()/2, f'{b.get_width()*100:.2f}%', va='center', fontweight='bold', size=14)

nombre_archivo = "Dashboard_Resultados_ANP.png"
plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight', facecolor="#0B0F19")
plt.close()

print(f"\n[SISTEMA] ✓ Imagen guardada exitosamente en la carpeta actual como '{nombre_archivo}'\n")