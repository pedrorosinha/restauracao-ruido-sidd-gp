"""Gravação de imagens, painéis de comparação e tabelas CSV."""

import csv
import statistics

import cv2
import numpy as np


def gravar_imagem(caminho, imagem):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(caminho), imagem):
        raise OSError(f"Falha ao salvar: {caminho}")


def painel_recorte(ruidosa, referencia, restauradas, tamanho=256):
    """Recortes da região central: ruidosa, referência e cada estratégia."""
    altura, largura = referencia.shape[:2]
    lado = min(tamanho, altura, largura)
    y0 = (altura - lado) // 2
    x0 = (largura - lado) // 2

    def recorte(imagem, titulo):
        r = imagem[y0:y0 + lado, x0:x0 + lado].copy()
        cv2.rectangle(r, (0, 0), (lado - 1, 18), (0, 0, 0), -1)
        cv2.putText(r, titulo, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return r

    blocos = [recorte(ruidosa, "ruidosa"), recorte(referencia, "referencia")]
    blocos += [recorte(img, nome) for nome, img in restauradas.items()]
    while len(blocos) % 5 != 0:
        blocos.append(np.zeros_like(blocos[0]))
    linhas = [np.hstack(blocos[i:i + 5]) for i in range(0, len(blocos), 5)]
    return np.vstack(linhas)


def escrever_csv(caminho, linhas, campos):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos, delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas)


def agregar(resultados, metodos):
    """Média, desvio, mediana, ganho sobre a ruidosa e vitórias de cada método."""
    metricas = ["psnr_db", "ssim", "mse", "correlacao_bordas",
                "razao_gradiente_bordas", "tempo_ms"]
    base = {r["id"]: r["psnr_db"] for r in resultados if r["metodo"] == "sem_filtro"}

    tabela = []
    for metodo in metodos:
        linhas = [r for r in resultados if r["metodo"] == metodo]
        if not linhas:
            continue
        item = {"metodo": metodo, "categoria": linhas[0]["categoria"],
                "n_imagens": len(linhas)}
        for m in metricas:
            valores = [float(r[m]) for r in linhas if np.isfinite(float(r[m]))]
            item[f"{m}_media"] = round(statistics.mean(valores), 4) if valores else ""
            item[f"{m}_desvio"] = round(statistics.stdev(valores), 4) if len(valores) > 1 else 0.0
            if m in ("psnr_db", "ssim"):
                item[f"{m}_mediana"] = round(statistics.median(valores), 4) if valores else ""

        ganhos = [r["psnr_db"] - base[r["id"]] for r in linhas if r["id"] in base]
        item["ganho_psnr_db_media"] = round(statistics.mean(ganhos), 4) if ganhos else 0.0
        item["imagens_melhoradas"] = sum(g > 0 for g in ganhos)
        tabela.append(item)

    vitorias = {m: 0 for m in metodos}
    for ident in sorted({r["id"] for r in resultados}):
        linhas = [r for r in resultados if r["id"] == ident]
        vitorias[max(linhas, key=lambda r: r["psnr_db"])["metodo"]] += 1
    for item in tabela:
        item["vezes_melhor_psnr"] = vitorias[item["metodo"]]

    tabela.sort(key=lambda i: i["psnr_db_media"], reverse=True)
    return tabela


def imprimir_resumo(agregados):
    print("\nResultados agregados (ordenados por PSNR médio):")
    print(f"{'método':<28}{'categoria':<26}{'PSNR (dB)':>10}{'ganho':>8}{'SSIM':>8}"
          f"{'corr.bordas':>13}{'razão grad.':>13}{'tempo ms':>10}")
    for item in agregados:
        print(f"{item['metodo']:<28}{item['categoria']:<26}{item['psnr_db_media']:>10.2f}"
              f"{item['ganho_psnr_db_media']:>8.2f}{item['ssim_media']:>8.4f}"
              f"{item['correlacao_bordas_media']:>13.4f}"
              f"{item['razao_gradiente_bordas_media']:>13.4f}{item['tempo_ms_media']:>10.1f}")