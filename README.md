!python3 quaternionic_sat_fixed.py --dimacs sha3_256_collision_1088.cnf  

Para evitar los errores de redondeo cuando no tienes precisión infinita, no calculas el punto exacto, sino que evalúas predicados geométricos para obtener un resultado booleano (verdadero o falso). En lugar de resolver algebraicamente $x = y$, determinas la relación posicional (por ejemplo, si un punto está a la izquierda, derecha o sobre una línea).
De esta manera logramos hacer P=NP de manera hiper practica.
