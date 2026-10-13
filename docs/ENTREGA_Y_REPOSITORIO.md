# Publicación y entrega

El paquete contiene el contenido del repositorio. **Falta publicar en el repositorio del equipo y entregar su enlace**, porque no se proporcionó una cuenta o URL de destino. No se inventa un enlace.

1. Descomprimir el ZIP completo y abrir la carpeta `Evaluacion2_001D`.
2. Revisar `README.md`, abrir el notebook ejecutado y ensayar la presentación.
3. Confirmar o corregir la matriz de responsabilidades para que refleje la distribución real del grupo.
4. Crear un repositorio del equipo y subir esta carpeta conservando la estructura. Si el navegador rechaza archivos grandes, utilizar Git desde terminal o GitHub Desktop. Los CSV individuales están bajo el límite habitual de 100 MB por archivo, pero el servicio elegido puede imponer otros límites.
5. Comprobar que README muestra imágenes, que notebook y presentación están accesibles y que la docente puede abrir el repositorio.
6. Entregar el enlace según las instrucciones del curso y antes del plazo anunciado por la docente. Los PDF proporcionados no indican una fecha concreta.

## Ejecución local
Python 3.12 fue el entorno de esta corrida. Crear un entorno virtual y, desde la raíz del proyecto:

```bash
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyterlab
```

Abrir `notebooks/Evaluacion2_CNN_MLP.ipynb` y ejecutar todas las celdas. Como alternativa:

```bash
python src/entrenar.py
```

El script regenera modelos, figuras y resultados pero no modifica las salidas guardadas del notebook. Las conclusiones estáticas de README y PPTX corresponden a la corrida entregada; si se cambia el experimento hay que actualizarlas según las nuevas métricas.

## Google Colab
Subir el ZIP completo al entorno Colab, descomprimirlo en `/content/` y abrir/subir el notebook. Si faltan dependencias, ejecutar `!pip install -r /content/Evaluacion2_001D/requirements.txt` y reiniciar la sesión antes de ejecutar todo. El notebook busca `/content/Evaluacion2_001D/data/`. Se fuerza CPU para conservar el protocolo de comparación, aunque los tiempos variarán respecto de la máquina de esta corrida.

## Reproducibilidad y límites
Semilla, particiones y huellas permiten reconstruir el experimento. Distintas versiones de librerías, hardware y orden de operaciones pueden producir pequeñas diferencias. No se promete identidad numérica universal. El notebook contiene verificaciones de dimensiones, particiones y recarga de modelos.
