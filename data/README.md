# Datos originales
CSV extraídos del ZIP aportado por el equipo, sin modificaciones. Sign Language MNIST de datamunge, Kaggle. El registro de descarga del notebook E1 declara licencia CC0-1.0.

- `sign_mnist_train.csv`: 27.455 filas, 785 columnas.
- `sign_mnist_test.csv`: 7.172 filas, 785 columnas.

Las huellas SHA-256 se guardan en `results/data_quality.json`. No se usan las copias duplicadas de subcarpetas ni metadatos de macOS del ZIP. El filtrado a seis clases ocurre en el notebook. El índice de fila guardado es el índice pandas base cero, excluyendo la cabecera CSV.

Fuente indicada en la evaluación 1: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
