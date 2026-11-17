"""Métricas de comparação entre a imagem restaurada e a referência sem ruído."""

import cv2
import numpy as np

BORDA = cv2.BORDER_REFLECT_101


def calcular_mse(a, b):
    diferenca = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diferenca ** 2))


def calcular_psnr(mse):
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((255.0 ** 2) / mse))


def calcular_ssim(a, b):
    """SSIM médio dos três canais, com janela Gaussiana 11×11 (σ = 1,5)."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    def suavizar(m):
        return cv2.GaussianBlur(m, (11, 11), 1.5, borderType=BORDA)

    valores = []
    for canal in range(a.shape[2]):
        x = a[:, :, canal].astype(np.float32)
        y = b[:, :, canal].astype(np.float32)

        mu_x = suavizar(x)
        mu_y = suavizar(y)
        var_x = suavizar(x * x) - mu_x * mu_x
        var_y = suavizar(y * y) - mu_y * mu_y
        cov_xy = suavizar(x * y) - mu_x * mu_y

        numerador = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        denominador = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
        valores.append(float(np.mean(numerador / denominador)))
    return float(np.mean(valores))


def magnitude_gradiente(imagem):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(cinza, cv2.CV_32F, 1, 0, ksize=3, borderType=BORDA)
    gy = cv2.Sobel(cinza, cv2.CV_32F, 0, 1, ksize=3, borderType=BORDA)
    return cv2.magnitude(gx, gy)


def mascara_bordas(magnitude_ref, percentil=90):
    """Pixels de borda da referência: os 10 % com maior magnitude do gradiente."""
    return magnitude_ref > float(np.percentile(magnitude_ref, percentil))


def preservacao_bordas(restaurada, magnitude_ref, mascara):
    """Indicadores de preservação de bordas, ambos baseados em Sobel.

    - correlacao_bordas: correlação de Pearson entre as magnitudes do gradiente
      da restaurada e da referência (1 = mesmas bordas);
    - razao_gradiente_bordas: magnitude média da restaurada ÷ da referência,
      só nos pixels de borda da referência
      (< 1 = bordas atenuadas; > 1 = bordas ou ruído realçados).
    """
    magnitude = magnitude_gradiente(restaurada)
    a = magnitude.ravel().astype(np.float64)
    b = magnitude_ref.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominador = np.sqrt(np.sum(a * a) * np.sum(b * b))
    correlacao = float(np.sum(a * b) / denominador) if denominador > 0 else 0.0

    if not mascara.any():
        return correlacao, 0.0
    media_ref = float(magnitude_ref[mascara].mean())
    media_rest = float(magnitude[mascara].mean())
    razao = media_rest / media_ref if media_ref > 0 else 0.0
    return correlacao, razao