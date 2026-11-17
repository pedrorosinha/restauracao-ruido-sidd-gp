"""
Restauração em lote de fotografias de smartphone com ruído real de aquisição.

Base: SIDD - Smartphone Image Denoising Dataset (versão Small, sRGB).
Subtema: Combinação 5 - Google Pixel (código de câmera GP).

Uso:
    python main.py --dados <pasta do SIDD> --saida saida
"""

import argparse
import time
from pathlib import Path

from filtros import CATEGORIAS, ESTRATEGIAS
from leitura import ler_imagem, localizar_pares
from metricas import (calcular_mse, calcular_psnr, calcular_ssim,
                      magnitude_gradiente, mascara_bordas, preservacao_bordas)
from relatorio import (agregar, escrever_csv, gravar_imagem,
                       imprimir_resumo, painel_recorte)


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Restauração em lote do SIDD (câmera GP).")
    parser.add_argument("--dados", required=True, help="pasta do SIDD Small sRGB")
    parser.add_argument("--saida", default="saida", help="pasta de resultados")
    parser.add_argument("--camera", default="GP", help="código da câmera (padrão: GP)")
    parser.add_argument("--sem-imagens", action="store_true",
                        help="não grava as imagens restauradas (só tabelas e painéis)")
    parser.add_argument("--sem-nlm", action="store_true",
                        help="pula o Non-Local Means (o método mais lento)")
    parser.add_argument("--limite", type=int, default=0,
                        help="processa só os N primeiros pares (0 = todos)")
    return parser.parse_args()


def processar_par(par, metodos, pasta_saida, gravar_restauradas):
    """Aplica todas as estratégias a um par e devolve uma linha de resultado por método."""
    ruidosa = ler_imagem(par["ruidosa"])
    referencia = ler_imagem(par["referencia"])
    if ruidosa.shape != referencia.shape:
        raise ValueError(f"shapes diferentes {ruidosa.shape} x {referencia.shape}")

    magnitude_ref = magnitude_gradiente(referencia)
    mascara = mascara_bordas(magnitude_ref)

    linhas = []
    restauradas = {}
    for metodo in metodos:
        inicio = time.perf_counter()
        restaurada = ruidosa if metodo == "sem_filtro" else ESTRATEGIAS[metodo](ruidosa)
        tempo_ms = (time.perf_counter() - inicio) * 1000

        # Requisito: dimensões, canais e tipo preservados.
        assert restaurada.shape == ruidosa.shape, f"{metodo} alterou o shape"
        assert restaurada.dtype == ruidosa.dtype, f"{metodo} alterou o dtype"

        mse = calcular_mse(restaurada, referencia)
        correlacao, razao = preservacao_bordas(restaurada, magnitude_ref, mascara)
        linhas.append({
            "id": par["id"],
            "cena": par["cena"],
            "iso": par["iso"],
            "brilho": par["brilho"],
            "altura": ruidosa.shape[0],
            "largura": ruidosa.shape[1],
            "canais": ruidosa.shape[2],
            "metodo": metodo,
            "categoria": CATEGORIAS[metodo],
            "mse": round(mse, 4),
            "psnr_db": round(calcular_psnr(mse), 4),
            "ssim": round(calcular_ssim(restaurada, referencia), 4),
            "correlacao_bordas": round(correlacao, 4),
            "razao_gradiente_bordas": round(razao, 4),
            "tempo_ms": round(tempo_ms, 2),
        })

        if metodo != "sem_filtro":
            restauradas[metodo] = restaurada
            if gravar_restauradas:
                gravar_imagem(pasta_saida / "restauradas" / metodo / f"{par['cena']}.png",
                              restaurada)

    gravar_imagem(pasta_saida / "paineis" / f"{par['cena']}.png",
                  painel_recorte(ruidosa, referencia, restauradas))
    return linhas


def main():
    args = ler_argumentos()
    pasta_dados = Path(args.dados)
    pasta_saida = Path(args.saida)
    if not pasta_dados.is_dir():
        raise FileNotFoundError(f"Pasta de dados não encontrada: {pasta_dados}")

    pares, exclusoes = localizar_pares(pasta_dados, args.camera.upper())
    if args.limite > 0:
        pares = pares[:args.limite]
    print(f"Câmera {args.camera.upper()}: {len(pares)} pares encontrados")
    if not pares:
        raise RuntimeError("Nenhum par encontrado. Verifique o caminho --dados.")

    metodos = ["sem_filtro"] + list(ESTRATEGIAS)
    if args.sem_nlm:
        metodos.remove("non_local_means")
    resultados = []

    for indice, par in enumerate(pares, start=1):
        print(f"[{indice}/{len(pares)}] {par['id']}")
        try:
            linhas = processar_par(par, metodos, pasta_saida, not args.sem_imagens)
        except (FileNotFoundError, ValueError) as erro:
            exclusoes.append((par["cena"], par["ruidosa"].name, str(erro)))
            print(f"    excluído: {erro}")
            continue
        resultados.extend(linhas)

        base = linhas[0]
        melhor = max(linhas, key=lambda r: r["psnr_db"])
        print(f"    ruidosa: {base['psnr_db']:.2f} dB | melhor: {melhor['metodo']} "
              f"{melhor['psnr_db']:.2f} dB")

    if not resultados:
        raise RuntimeError("Nenhum par pôde ser processado. Veja as exclusões.")

    escrever_csv(pasta_saida / "resultados_por_imagem.csv", resultados,
                 list(resultados[0].keys()))

    agregados = agregar(resultados, metodos)
    campos = list(dict.fromkeys(k for item in agregados for k in item))
    escrever_csv(pasta_saida / "resultados_agregados.csv", agregados, campos)

    escrever_csv(pasta_saida / "exclusoes.csv",
                 [{"cena": c, "arquivo": a, "motivo": m} for c, a, m in exclusoes],
                 ["cena", "arquivo", "motivo"])

    imprimir_resumo(agregados)
    print(f"\nPares processados: {len({r['id'] for r in resultados})} | exclusões: {len(exclusoes)}")
    print(f"Resultados gravados em: {pasta_saida.resolve()}")


if __name__ == "__main__":
    main()