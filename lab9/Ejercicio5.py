import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, chisquare

class MersenneTwister:
    def __init__(self, 
                 w=32, 
                 n=624, 
                 m=397, 
                 r=31, 
                 a=0x9908B0DF, 
                 u=11, 
                 d=0xFFFFFFFF, 
                 s=7, 
                 b=0x9D2C5680, 
                 t=15, 
                 c=0xEFC60000, 
                 l=18, 
                 f=1812433253,
                 seed=None):
        """
        Inicializa el generador Mersenne Twister con los parámetros especificados.

        :param w: Ancho de palabra en bits.
        :param n: Tamaño del estado interno.
        :param m: Desplazamiento utilizado en el twist.
        :param r: Bit de separación.
        :param a: Coeficiente de la matriz.
        :param u: Shift para el primer tempering XOR.
        :param d: Máscara para el primer tempering XOR.
        :param s: Shift para el segundo tempering XOR.
        :param b: Máscara para el segundo tempering XOR.
        :param t: Shift para el tercer tempering XOR.
        :param c: Máscara para el tercer tempering XOR.
        :param l: Shift para el cuarto tempering XOR.
        :param f: Constante de multiplicación para la inicialización.
        :param seed: Semilla para inicializar el generador.
        """
        # Parámetros del MT
        self.w = w
        self.n = n
        self.m = m
        self.r = r
        self.a = a
        self.u = u
        self.d = d
        self.s = s
        self.b = b
        self.t = t
        self.c = c
        self.l = l
        self.f = f

        # Estado interno del generador
        self.MT = [0] * self.n
        self.index = self.n + 1  # Indica que el generador necesita ser inicializado

        if seed is not None:
            self.seed_mt(seed)
        else:
            # Si no se proporciona semilla, usar el tiempo actual
            self.seed_mt(int(time.time()))
    
    def seed_mt(self, seed):
        """Inicializa el generador con una semilla."""
        self.index = self.n
        self.MT[0] = seed
        for i in range(1, self.n):
            temp = self.MT[i-1] ^ (self.MT[i-1] >> (self.w - 2))
            self.MT[i] = (self.f * temp + i) & ((1 << self.w) - 1)
    
    def extract_number(self):
        """Extrae un número pseudoaleatorio."""
        if self.index >= self.n:
            if self.index > self.n:
                raise Exception("El generador no ha sido inicializado correctamente")
            self.twist()
        
        y = self.MT[self.index]
        y ^= (y >> self.u) & self.d
        y ^= (y << self.s) & self.b
        y ^= (y << self.t) & self.c
        y ^= (y >> self.l)

        self.index += 1
        return y & ((1 << self.w) - 1)
    
    def twist(self):
        """Actualiza el estado interno del generador."""
        for i in range(self.n):
            x = (self.MT[i] & ((1 << self.w) - (1 << self.r))) + (self.MT[(i + 1) % self.n] & ((1 << self.r) - 1))
            xA = x >> 1
            if x % 2 != 0:
                xA ^= self.a
            self.MT[i] = self.MT[(i + self.m) % self.n] ^ xA
        self.index = 0
    
    def random(self):
        """Genera un número aleatorio en el intervalo [0, 1)."""
        return self.extract_number() / float(1 << self.w)  # Normalización a [0, 1)

def generar_numeros_uniformes(cantidad, semilla=None, **params):
    """
    Genera una lista de números aleatorios con distribución uniforme entre 0 y 1
    utilizando el generador Mersenne Twister implementado con parámetros personalizados.

    :param cantidad: Número de números aleatorios a generar.
    :param semilla: (Opcional) Semilla para el generador de números aleatorios.
    :param params: Parámetros personalizados para el generador.
    :return: Lista de números aleatorios.
    """
    mt = MersenneTwister(seed=semilla, **params)
    numeros = [mt.random() for _ in range(cantidad)]
    return np.array(numeros)

# Ejemplo de uso
if __name__ == "__main__":
    cantidad_numeros = 1000
    semilla = 42  # Opcional: puedes cambiar o eliminar para obtener diferentes secuencias

    parametros_practicos = [{
        'w': 48,
        'n': 500,
        'm': 250,
        'r': 27,
        'a': 0xA1B2C3D4,
        'u': 14,
        'd': 0x1234ABCD,
        's': 10,
        'b': 0x5E6F7A8B,
        't': 20,
        'c': 0xCDEF1234,
        'l': 25,
        'f': 987654321
    },{  
        'w': 32, 
        'n': 624, 
        'm': 397, 
        'r': 31, 
        'a': 0x9908B0DF, 
        'u': 11, 
        'd': 0xFFFFFFFF, 
        's': 7, 
        'b': 0x9D2C5680, 
        't': 15, 
        'c': 0xEFC60000, 
        'l': 18, 
        'f': 1812433253
    },{    
        'w': 64,
        'n': 312,
        'm': 199,
        'r': 29,
        'a': 0x87654321,
        'u': 15,
        'd': 0x12345678,
        's': 11,
        'b': 0x4D3C2B1A,
        't': 19,
        'c': 0xABCDEF12,
        'l': 22,
        'f': 123456789
    }]

# Generación y comparación de muestras
for i, parametros_prac in enumerate(parametros_practicos):
    # Generar muestra teórica (GLC aleatorio)
    muestra_teorica = generar_numeros_uniformes(cantidad_numeros, semilla)
    # Generar muestra práctica con parámetros específicos
    muestra_practica = generar_numeros_uniformes(cantidad_numeros, semilla, **parametros_prac)
    
    # Pruebas de Hipótesis
    # Prueba de Kolmogorov-Smirnov para comparar con una distribución uniforme
    ks_stat_teorico, p_val_teorico = kstest(muestra_teorica, 'uniform')
    ks_stat_practico, p_val_practico = kstest(muestra_practica, 'uniform')
    
    # Prueba de Chi-Cuadrado
    bins = np.linspace(0, 1, 11)  # Dividimos el rango en 10 intervalos para Chi-Cuadrado
    hist_teorico, _ = np.histogram(muestra_teorica, bins=bins)
    hist_practico, _ = np.histogram(muestra_practica, bins=bins)
    chi2_teorico, chi2_pval_teorico = chisquare(hist_teorico)
    chi2_practico, chi2_pval_practico = chisquare(hist_practico)
    
    print(f"\nConjunto de Parámetros {i+1}:", parametros_prac)

    # Resultados
    print(f"\nPrueba de Kolmogorov-Smirnov (Teórico): estadístico = {ks_stat_teorico:.4f}, valor p = {p_val_teorico:.4f}")
    print(f"Prueba de Chi-Cuadrado (Teórico): estadístico = {chi2_teorico:.4f}, valor p = {chi2_pval_teorico:.4f}")
    
    print(f"\nPrueba de Kolmogorov-Smirnov (Práctico): estadístico = {ks_stat_practico:.4f}, valor p = {p_val_practico:.4f}")
    print(f"Prueba de Chi-Cuadrado (Práctico): estadístico = {chi2_practico:.4f}, valor p = {chi2_pval_practico:.4f}")
    
    if p_val_teorico > 0.05:
        print("Muestra teórica: No se rechaza la hipótesis nula. La muestra generada puede considerarse como uniforme.\n")
    else:
        print("Muestra teórica: Se rechaza la hipótesis nula. La muestra generada no puede considerarse como uniforme.\n")
    
    if p_val_practico > 0.05:
        print("Muestra práctica: No se rechaza la hipótesis nula. La muestra generada puede considerarse como uniforme.\n")
    else:
        print("Muestra práctica: Se rechaza la hipótesis nula. La muestra generada no puede considerarse como uniforme.\n")
    
    # Graficar histogramas de la muestra teórica y práctica
    plt.figure(figsize=(12, 5))
    
    # Histograma de la muestra teórica
    plt.subplot(1, 2, 1)
    plt.hist(muestra_teorica, bins=15, density=True, alpha=0.6, color='green', edgecolor='black')
    plt.title(f'Histograma de Muestra Teórica')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    
    # Histograma de la muestra práctica
    plt.subplot(1, 2, 2)
    plt.hist(muestra_practica, bins=15, density=True, alpha=0.6, color='blue', edgecolor='black')
    plt.title(f'Histograma de Muestra Práctica (Conjunto {i+1})')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    
    plt.show()
