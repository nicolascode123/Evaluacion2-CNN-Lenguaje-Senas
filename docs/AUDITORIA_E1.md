# Revisión de la evaluación 1 y del ejemplo docente

## Caso y evidencia recuperada
- Docente: Jazna Hidalgo, identificada en la portada y cierre de la presentación E1.
- Integrantes para esta entrega: Nicolás Fernández Vera, Bastian Gutierrez y Victor Gutierrez. Sección 001D, indicada por el usuario.
- Variante: A/C/F/K/O/Y, originales 0/2/5/10/14/24 y nuevas 0/1/2/3/4/5.
- Train filtrado 6.902, dividido en 5.521 train y 1.381 validación. Test oficial filtrado: 1.797.
- MLP 784–128–64–6, 109.126 parámetros, ReLU/ReLU/Softmax. Adam, sparse categorical crossentropy, 20 épocas y batch 32.
- Salidas históricas: accuracy 0,8703, precision macro 0,8778, recall macro 0,8698, F1 macro 0,8694 y 233 errores. Validación final 1,0000 y test loss 0,5159.
- Tiempo: salida ejecutada 32,7547 segundos. La presentación dice 35,18 segundos. Se documenta esta diferencia sin reconciliarla artificialmente.

## Correcciones conceptuales y de implementación
1. El MLP conserva los píxeles al aplanar, pero no incorpora el sesgo espacial local de una convolución. No corresponde decir que borra toda la información espacial.
2. El prototipo reconoce seis letras ASL estáticas. No demuestra traducción de conversaciones, teleconferencias o lengua de señas chilena.
3. Se reemplaza la descarga automática y borrado de archivos de E1 por carga local de los CSV provistos. No exige credenciales Kaggle ni elimina ZIP del usuario.
4. En el ejemplo CNN docente se usa CIFAR-10. Sus tres canales, diez clases y dimensiones 32×32 no corresponden a nuestro caso.
5. El ejemplo contiene normalización repetida en una celda del MLP y usa el test como validación. Nuestro notebook normaliza una vez y mantiene validación independiente.
6. La función `get_prediction` del ejemplo llama a `modelo_a` aunque recibe otro modelo. La entrega E2 evalúa explícitamente cada modelo correcto.
7. Se recalculan parámetros de acuerdo con pesos y sesgos, sin copiar errores aritméticos de explicaciones del ejemplo.
8. Se distingue entre mapas de filtros y mapas específicos de clase. Se añade Grad-CAM sin modificar la arquitectura entrenada.
9. Aunque el notebook E1 importa Dropout, su modelo no lo utiliza. La CNN E2 tampoco incluye capas de regularización por penalización expresa de la rúbrica.
