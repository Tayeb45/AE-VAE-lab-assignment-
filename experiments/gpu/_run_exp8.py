import nbformat
from nbclient import execute
from pathlib import Path

root = Path(__file__).resolve().parents[2]
nb_path = Path(__file__).resolve().parent / "exp_8_vae_camus.ipynb"
nb = nbformat.read(nb_path, as_version=4)
try:
    execute(nb, cwd=str(root), timeout=14400, kernel_name="python3", startup_timeout=180)
finally:
    nbformat.write(nb, nb_path)
print("execution finished")
