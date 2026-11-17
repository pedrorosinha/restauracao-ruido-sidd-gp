# Restauração em lote — SIDD, Combinação 5 (Google Pixel, código GP)

## Ambiente

```
python -m pip install opencv-python numpy
```

## Base de dados

1. Baixar **SIDD-Small Dataset → sRGB images only** em https://abdokamel.github.io/sidd/
2. Extrair o arquivo. A estrutura esperada é:

```
SIDD_Small_sRGB_Only/
└── Data/
    ├── 0001_001_S6_00100_00060_3200_L/
    │   ├── GT_SRGB_010.PNG
    │   └── NOISY_SRGB_010.PNG
    ├── ..._GP_..._.../        ← apenas estas pastas são processadas
    └── ...
```

O código da câmera é o 3º campo do nome da pasta
(`<instância>_<cena>_<câmera>_<ISO>_<obturador>_<temperatura>_<brilho>`).

## Execução

```
python restauracao_sidd_gp.py --dados SIDD_Small_sRGB_Only/Data --saida saida
```

Opções: `--sem-imagens` (não grava as restauradas, só tabelas e painéis) e
`--limite N` (processa só os N primeiros pares, para testes).

## Estratégias de redução de ruído

| Nome | Técnica (aula) |
|---|---|
| `media_3x3`, `media_5x5` | `cv2.blur` — filtro da média (Aula 7) |
| `gaussiano_5x5` | `cv2.GaussianBlur` (Aula 7) |
| `gaussiano_filter2d_3x3` | kernel 1/16·[1 2 1; 2 4 2; 1 2 1] com `cv2.filter2D` (Aula 7) |
| `mediana_3`, `mediana_5` | `cv2.medianBlur` — filtro não linear (Aula 7) |
| `luminancia_crominancia` | `cv2.cvtColor` para YCrCb; mediana 3 na luminância e Gaussiano 9×9 nos canais de cor (Aulas 6 e 7) |
| `adaptativo_bordas` | magnitude Sobel define um peso por pixel; bordas → Gaussiano 3×3, regiões planas → Gaussiano 9×9 (Aula 7 + operação ponto a ponto da Aula 6) |

Todas usam `BORDER_REFLECT_101`. Depois de cada filtro, o programa verifica
(`assert`) que `shape` (altura, largura, canais) e `dtype` (`uint8`) não mudaram.

## Métricas (restaurada × referência sem ruído)

- **MSE** e **PSNR** (dB) — erro de pixel; PSNR maior = melhor.
- **SSIM** — similaridade estrutural, calculada com `cv2.GaussianBlur` (janela 11×11, σ=1,5), média dos 3 canais.
- **correlacao_bordas** — correlação entre as magnitudes Sobel da restaurada e da referência (1 = bordas idênticas).
- **razao_gradiente_bordas** — magnitude Sobel média da restaurada ÷ da referência, nos 10 % de pixels de borda mais fortes da referência. < 1 indica bordas suavizadas demais; > 1 indica ruído ou bordas realçados.
- **tempo_ms** — tempo do filtro com `time.perf_counter` (Aula 8).

`sem_filtro` é a própria imagem ruidosa, usada como linha de base.

## Saídas (pasta `saida/`)

| Arquivo | Conteúdo |
|---|---|
| `resultados_por_imagem.csv` | uma linha por imagem × método (ISO, brilho, dimensões, métricas, tempo) |
| `resultados_agregados.csv` | por método: média, desvio-padrão, mediana, ganho médio de PSNR sobre a ruidosa, nº de imagens melhoradas, nº de vezes que foi o melhor PSNR |
| `exclusoes.csv` | pares descartados e o motivo (referência ausente, falha de leitura, dimensões diferentes); fica vazio quando todos os pares GP são usados |
| `restauradas/<método>/<cena>.png` | imagens restauradas em PNG (sem perdas), com as dimensões originais |
| `paineis/<cena>.png` | recorte central 256×256 de ruidosa, referência e cada método, para inspeção visual |

Os CSV usam `;` como separador (abre direto no Excel em português).
