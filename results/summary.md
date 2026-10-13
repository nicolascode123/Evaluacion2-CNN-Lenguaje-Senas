### Síntesis experimental
- MLP reentrenado: **87.81%** de accuracy, **0.8780** F1 macro y **219** errores.
- CNN: **97.27%** de accuracy, **0.9706** F1 macro y **49** errores.
- Diferencia de accuracy: **+9.46 puntos porcentuales**.
- Parámetros: **109.126 vs 56.454**. El costo convolucional depende también de operaciones y hardware, no solo de pesos.
- Tiempos de fit: **23.85 s vs 27.00 s**, misma CPU y protocolo.
- Brecha train-test: **12.19 pp vs 2.73 pp**.
- Evidencia histórica E1: 87,03% y F1 macro 0,8694. Debe distinguirse del MLP reentrenado.
