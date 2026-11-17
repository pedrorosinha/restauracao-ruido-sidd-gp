"""Estratégias de redução de ruído.

Todas recebem uma imagem BGR uint8 (altura, largura, 3)
e devolvem uma imagem com o mesmo shape e o mesmo dtype.
"""

import cv2
import numpy as np

BORDA = cv2.BORDER_REFLECT_101


def media_3x3(imagem):
    return cv2.blur(imagem, (3, 3), borderType=BORDA)


def media_5x5(imagem):
    return cv2.blur(imagem, (5, 5), borderType=BORDA)


def gaussiano_5x5(imagem):
    return cv2.GaussianBlur(imagem, (5, 5), sigmaX=0, borderType=BORDA)


def gaussiano_filter2d_3x3(imagem):
    """Kernel Gaussiano 3×3 (1/16 [1 2 1; 2 4 2; 1 2 1]) aplicado com filter2D."""
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ], dtype=np.float32) / 16
    return cv2.filter2D(imagem, ddepth=-1, kernel=kernel, borderType=BORDA)


def mediana_3(imagem):
    return cv2.medianBlur(imagem, 3)


def mediana_5(imagem):
    return cv2.medianBlur(imagem, 5)


def luminancia_crominancia(imagem):
    """Suavização separada de luminância e cor.

    O ruído real de smartphone tem forte componente de cor (manchas coloridas).
    A imagem é convertida para YCrCb:
        - Y (luminância, onde estão bordas e texturas): mediana 3 (leve);
        - Cr e Cb (cor): Gaussiano 9×9 (forte).
    Depois volta para BGR.
    """
    ycrcb = cv2.cvtColor(imagem, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.medianBlur(y, 3)
    cr = cv2.GaussianBlur(cr, (9, 9), sigmaX=0, borderType=BORDA)
    cb = cv2.GaussianBlur(cb, (9, 9), sigmaX=0, borderType=BORDA)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)


def adaptativo_bordas(imagem):
    """Combinação ponderada por bordas.

    1. Estima as bordas na luminância suavizada (Sobel X, Y e magnitude);
    2. normaliza a magnitude para um peso w entre 0 e 1;
    3. saída = w · suavização_leve + (1 − w) · suavização_forte.

    Regiões planas (w ≈ 0) recebem suavização forte; bordas (w ≈ 1)
    recebem suavização leve, reduzindo a perda de detalhes.
    """
    leve = cv2.GaussianBlur(imagem, (3, 3), sigmaX=0, borderType=BORDA)
    forte = cv2.GaussianBlur(imagem, (9, 9), sigmaX=0, borderType=BORDA)

    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (5, 5), sigmaX=0, borderType=BORDA)
    gx = cv2.Sobel(cinza, cv2.CV_32F, 1, 0, ksize=3, borderType=BORDA)
    gy = cv2.Sobel(cinza, cv2.CV_32F, 0, 1, ksize=3, borderType=BORDA)
    magnitude = cv2.magnitude(gx, gy)

    # O percentil 95 da magnitude corresponde a peso 1.
    referencia = float(np.percentile(magnitude, 95))
    if referencia <= 0:
        referencia = 1.0
    peso = np.clip(magnitude / referencia, 0.0, 1.0)[:, :, np.newaxis]

    saida = peso * leve.astype(np.float32) + (1.0 - peso) * forte.astype(np.float32)
    return np.clip(saida + 0.5, 0, 255).astype(np.uint8)


def adaptativo_estatistico(imagem, tamanho=7):
    """Filtro adaptativo de redução local de ruído (Lee / Wiener local).

    Implementado pelo grupo. Para cada pixel g, na vizinhança tamanho×tamanho:
        m_L  = média local
        σ²_L = variância local  = média(g²) − m_L²
        σ²_η = variância do ruído (estimada como a média de σ²_L na imagem)

        f(x, y) = g − (σ²_η / σ²_L) · (g − m_L),   com a razão limitada a 1

    Decisão por pixel:
        - região plana  (σ²_L ≈ σ²_η) → razão ≈ 1 → saída ≈ média local (suaviza);
        - borda/textura (σ²_L ≫ σ²_η) → razão ≈ 0 → saída ≈ pixel original (preserva).
    Aplicado em cada canal B, G, R separadamente.
    """
    g = imagem.astype(np.float32)
    media = cv2.blur(g, (tamanho, tamanho), borderType=BORDA)
    media_quad = cv2.blur(g * g, (tamanho, tamanho), borderType=BORDA)
    var_local = np.maximum(media_quad - media * media, 0.0)

    var_ruido = var_local.mean(axis=(0, 1), keepdims=True)  # um valor por canal
    razao = np.minimum(var_ruido / np.maximum(var_local, 1e-6), 1.0)

    saida = g - razao * (g - media)
    return np.clip(saida + 0.5, 0, 255).astype(np.uint8)


def adaptativo_estatistico_5x5(imagem):
    return adaptativo_estatistico(imagem, 5)


def adaptativo_estatistico_7x7(imagem):
    return adaptativo_estatistico(imagem, 7)


def bilateral(imagem):
    """Alternativa pesquisada: filtro bilateral.

    Média ponderada por distância espacial E por diferença de intensidade:
    vizinhos com cor muito diferente (outro lado de uma borda) recebem peso baixo.
    """
    return cv2.bilateralFilter(imagem, d=9, sigmaColor=40, sigmaSpace=9,
                               borderType=BORDA)


def non_local_means(imagem):
    """Alternativa pesquisada: Non-Local Means (Buades et al., 2005).

    Cada pixel vira a média de pixels cujo bloco 7×7 ao redor é parecido com o seu,
    procurados numa janela 21×21. Aproveita padrões repetidos da imagem.
    """
    return cv2.fastNlMeansDenoisingColored(imagem, None, h=5, hColor=5,
                                           templateWindowSize=7, searchWindowSize=21)


ESTRATEGIAS = {
    "media_3x3": media_3x3,
    "media_5x5": media_5x5,
    "gaussiano_5x5": gaussiano_5x5,
    "gaussiano_filter2d_3x3": gaussiano_filter2d_3x3,
    "mediana_3": mediana_3,
    "mediana_5": mediana_5,
    "luminancia_crominancia": luminancia_crominancia,
    "adaptativo_bordas": adaptativo_bordas,
    "adaptativo_estatistico_5x5": adaptativo_estatistico_5x5,
    "adaptativo_estatistico_7x7": adaptativo_estatistico_7x7,
    "bilateral": bilateral,
    "non_local_means": non_local_means,
}

CATEGORIAS = {
    "sem_filtro": "linha de base",
    "media_3x3": "filtro clássico",
    "media_5x5": "filtro clássico",
    "gaussiano_5x5": "filtro clássico",
    "gaussiano_filter2d_3x3": "filtro clássico",
    "mediana_3": "filtro clássico",
    "mediana_5": "filtro clássico",
    "luminancia_crominancia": "combinação do grupo",
    "adaptativo_bordas": "combinação do grupo",
    "adaptativo_estatistico_5x5": "adaptativo implementado",
    "adaptativo_estatistico_7x7": "adaptativo implementado",
    "bilateral": "alternativa pesquisada",
    "non_local_means": "alternativa pesquisada",
}