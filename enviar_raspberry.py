
from pathlib import Path
import subprocess

# ========== CONFIGURACOES ==========

IP_RASPBERRY = "192.168.1.50"
USUARIO = "pi"
PORTA = 22

PASTA_REMOTA = f"/home/{USUARIO}/joker"

# Pasta onde este script está
BASE = Path(__file__).resolve().parent

ARQUIVO_CSV_PADRAO = BASE / "hsv.csv"
ARQUIVO_IMAGEM_PADRAO = BASE / "img.jpg"


# ========== TRANSFERENCIA ==========

def enviar_arquivos(arquivo_csv=None, arquivo_imagem=None):
    csv_path = Path(arquivo_csv) if arquivo_csv is not None else ARQUIVO_CSV_PADRAO
    imagem_path = Path(arquivo_imagem) if arquivo_imagem is not None else ARQUIVO_IMAGEM_PADRAO

    arquivos = [csv_path, imagem_path]

    for arquivo in arquivos:
        if not arquivo.is_file():
            print(f"Arquivo não encontrado: {arquivo}")
            return False

    destino = f"{USUARIO}@{IP_RASPBERRY}"

    try:
        print("Conectando à Raspberry...")

        subprocess.run(
            ["ssh", "-p", str(PORTA), destino, f"mkdir -p {PASTA_REMOTA}"],
            check=True,
            capture_output=True,
            text=True,
        )

        print("Enviando arquivos...")

        subprocess.run(
            [
                "scp",
                "-P",
                str(PORTA),
                str(csv_path),
                str(imagem_path),
                f"{destino}:{PASTA_REMOTA}/",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        print("Transferência concluída!")
        return True

    except FileNotFoundError:
        print("Comandos SSH/SCP não encontrados no Windows.")
        return False

    except subprocess.CalledProcessError as erro:
        print(f"Erro na transferência: {erro}")
        if erro.stderr:
            print(erro.stderr)
        return False


if __name__ == "__main__":
    enviar_arquivos()
