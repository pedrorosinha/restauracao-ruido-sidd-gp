# Restauração em lote — SIDD, Combinação 5 (Google Pixel, código GP)

Sistema de redução de ruído real de aquisição em fotografias de smartphone.
O objetivo é remover o ruído sem eliminar excessivamente bordas e detalhes.

O sistema faz, em lote:
1. lê os pares de imagem ruidosa e imagem de referência;
2. aplica diferentes estratégias de redução de ruído;
3. confere que as dimensões e os canais das imagens foram preservados;
4. grava as imagens restauradas;
5. compara os resultados com as referências sem ruído;
6. gera uma tabela com os resultados de cada imagem e os resultados agregados.

## Ambiente

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows (PowerShell)
python -m pip install opencv-python numpy
```

## Base de dados

**SIDD - Smartphone Image Denoising Dataset**, versão **SIDD-Small, sRGB**
(160 pares de imagem ruidosa e referência, 5 modelos de smartphone).

- Download oficial: https://abdokamel.github.io/sidd/dataset.html
- Página do projeto: https://abdokamel.github.io/sidd/

Depois de extrair, cada cena fica numa pasta assim:

```
SIDD_Small_sRGB_Only/
└── Data/
    ├── 0017_001_GP_00100_00060_5500_N/
    │   ├── GT_SRGB_010.PNG      ← referência sem ruído
    │   └── NOISY_SRGB_010.PNG   ← imagem ruidosa
    └── ...
```

O nome da pasta segue o formato
`<instância>_<cena>_<câmera>_<ISO>_<obturador>_<temperatura>_<brilho>`.
O programa processa só as pastas cujo código de câmera é **GP**.
Ele procura essas pastas em todas as subpastas, então `--dados` pode apontar
para a pasta raiz da base.

## Execução

```
python main.py --dados <pasta do SIDD> --saida saida
```

| Opção | Efeito |
|---|---|
| `--limite N` | processa só os N primeiros pares, em ordem alfabética das pastas (0 = todos) |
| `--sem-nlm` | pula o Non-Local Means, o método mais lento |
| `--sem-imagens` | não grava as imagens restauradas (só tabelas e painéis) |

Os resultados deste repositório foram gerados com `--limite 10`, ou seja,
os 10 primeiros pares GP, por causa do tempo de processamento.
Para processar todos os pares, use `--limite 0`.

## Estrutura do código

| Arquivo | Função |
|---|---|
| `main.py` | ponto de entrada: lê os argumentos, processa cada par e gera as tabelas |
| `leitura.py` | localiza os pares GP e lê as imagens |
| `filtros.py` | estratégias de redução de ruído |
| `metricas.py` | MSE, PSNR, SSIM e preservação de bordas |
| `relatorio.py` | grava imagens, painéis, CSVs e o resumo no terminal |

## Estratégias

| Método | Categoria | Descrição |
|---|---|---|
| `media_3x3`, `media_5x5` | filtro clássico | `cv2.blur` |
| `gaussiano_5x5` | filtro clássico | `cv2.GaussianBlur` |
| `gaussiano_filter2d_3x3` | filtro clássico | kernel 1/16·[1 2 1; 2 4 2; 1 2 1] com `cv2.filter2D` |
| `mediana_3`, `mediana_5` | filtro clássico | `cv2.medianBlur` |
| `luminancia_crominancia` | combinação do grupo | YCrCb: mediana 3 na luminância e Gaussiano 9×9 nos canais de cor |
| `adaptativo_bordas` | combinação do grupo | o peso de cada pixel vem da magnitude Sobel: bordas recebem Gaussiano 3×3 e regiões planas recebem Gaussiano 9×9 |
| `adaptativo_estatistico_5x5`, `_7x7` | **adaptativo implementado** | filtro de Lee / Wiener local, baseado em média e variância locais (ver abaixo) |
| `bilateral` | alternativa pesquisada | `cv2.bilateralFilter`: pondera os vizinhos pela distância e pela diferença de cor |
| `non_local_means` | alternativa pesquisada | `cv2.fastNlMeansDenoisingColored` (Buades et al., 2005) |

Todos os filtros usam `BORDER_REFLECT_101` quando a função permite.
Depois de cada filtro, o programa verifica (`assert`) que o `shape`
(altura, largura, canais) e o `dtype` (`uint8`) não mudaram.

### Filtro adaptativo por estatísticas locais

Para cada pixel `g`, numa vizinhança N×N:

- `m_L` = média local
- `σ²_L` = variância local = média(g²) − m_L²
- `σ²_η` = variância do ruído, estimada como a média de `σ²_L` na imagem

```
f(x, y) = g − (σ²_η / σ²_L) · (g − m_L),   com a razão limitada a 1
```

- **Região plana** (`σ²_L ≈ σ²_η`): a razão fica perto de 1 e a saída fica perto da média local, ou seja, o filtro suaviza.
- **Borda ou textura** (`σ²_L ≫ σ²_η`): a razão fica perto de 0 e a saída fica perto do pixel original, ou seja, o filtro preserva.

O filtro é aplicado em cada canal B, G, R separadamente.

## Métricas

Todas comparam a imagem restaurada com a referência sem ruído.

- **MSE** e **PSNR** (dB): erro por pixel. PSNR maior é melhor.
- **SSIM**: similaridade estrutural, com janela Gaussiana 11×11 (σ = 1,5), média dos 3 canais.
- **correlacao_bordas**: correlação entre as magnitudes Sobel da restaurada e da referência. 1 significa bordas idênticas.
- **razao_gradiente_bordas**: magnitude Sobel média da restaurada dividida pela da referência, calculada nos 10 % de pixels de borda mais fortes da referência. Abaixo de 1, as bordas foram suavizadas; acima de 1, há ruído ou bordas realçados.
- **tempo_ms**: tempo de execução do filtro.

`sem_filtro` é a própria imagem ruidosa, usada como linha de base.

## Saídas (`saida/`)

| Arquivo | Conteúdo |
|---|---|
| `resultados_por_imagem.csv` | uma linha por imagem e método |
| `resultados_agregados.csv` | por método: média, desvio-padrão, mediana, ganho de PSNR sobre a ruidosa, imagens melhoradas e vezes em que foi o melhor |
| `exclusoes.csv` | pares descartados e o motivo (fica vazio quando nenhum par é excluído) |
| `paineis/<cena>.png` | recorte central de 256×256 da ruidosa, da referência e de cada método |
| `restauradas/<método>/<cena>.png` | imagens restauradas com as dimensões originais (não versionadas: muito grandes) |

Os CSVs usam `;` como separador, então abrem direto no Excel em português.

Os três painéis incluídos no repositório:

| Painel | ISO | Situação |
|---|---|---|
| `0017_001_GP_00100_00060_5500_N.png` | 100 | pouco ruído |
| `0038_002_GP_00800_00640_3200_L.png` | 800 | ruído intermediário, pouca luz |
| `0036_002_GP_06400_03200_3200_N.png` | 6400 | ruído extremo |