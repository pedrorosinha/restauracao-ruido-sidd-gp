"""Localização e leitura dos pares (ruidosa, referência) do SIDD."""

from pathlib import Path

import cv2


def localizar_pares(pasta_dados, camera):
    """Procura as pastas de cena da câmera e devolve os pares encontrados.

    Nome das pastas no SIDD:
        <instancia>_<cena>_<camera>_<ISO>_<obturador>_<temperatura>_<brilho>
        ex.: 0017_001_GP_00100_00060_5500_N
    Dentro de cada pasta: NOISY_SRGB_010.PNG e GT_SRGB_010.PNG.
    """
    pares = []
    exclusoes = []

    for pasta in sorted(p for p in Path(pasta_dados).rglob("*") if p.is_dir()):
        campos = pasta.name.split("_")
        if len(campos) < 3 or campos[2].upper() != camera:
            continue

        arquivos = {a.name.upper(): a for a in pasta.iterdir() if a.is_file()}
        ruidosas = sorted(n for n in arquivos if n.startswith("NOISY_SRGB"))

        if not ruidosas:
            exclusoes.append((pasta.name, "-", "nenhuma imagem NOISY_SRGB encontrada"))
            continue

        for nome_ruidosa in ruidosas:
            nome_ref = nome_ruidosa.replace("NOISY_", "GT_", 1)
            if nome_ref not in arquivos:
                exclusoes.append((pasta.name, nome_ruidosa, f"referência {nome_ref} ausente"))
                continue
            pares.append({
                "id": f"{pasta.name}/{Path(nome_ruidosa).stem}",
                "cena": pasta.name,
                "iso": campos[3] if len(campos) > 3 else "",
                "brilho": campos[6] if len(campos) > 6 else "",
                "ruidosa": arquivos[nome_ruidosa],
                "referencia": arquivos[nome_ref],
            })

    return pares, exclusoes


def ler_imagem(caminho):
    imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho}")
    return imagem