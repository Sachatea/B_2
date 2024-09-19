import numpy as py
from sympy import (
    symbols,  # Для оголошення символів
    Number,
    sin,      # Синус
    cos,      # Косинус
    tan,      # Тангенс
    asin,     # Арксинус
    acos,     # Арккосинус
    atan,     # Арктангенс
    sinh,     # Гіперболічний синус
    cosh,     # Гіперболічний косинус
    tanh,     # Гіперболічний тангенс
    asinh,    # Арггиперболічний синус
    acosh,    # Арггиперболічний косинус
    atanh,    # Арггиперболічний тангенс
    exp,      # Експоненціальна функція
    log,      # Натуральний логарифм
    sqrt,     # Квадратний корінь
    Abs,      # Абсолютна величина
    factorial, # Факторіал
    gamma,    # Гамма-функція
    integrate, # Для обчислення інтегралів
    diff       # Для обчислення похідних
)
import sympy as sp
t = symbols('t')

def evaluate_expression(expr, value):
    if isinstance(expr, int) or isinstance(expr, float): 
        return expr  
    else:
        return expr.subs(t, value).evalf() 

def pidstava(A, t_v):
    matrix_evaluated = py.array([[evaluate_expression(expr, t_v) for expr in row]for row in A])
    return matrix_evaluated

def Det(A, T, N):
    krok = T/(N+1)
    k = min(A.shape)
    matrix_zeros = py.zeros((k*N, k*N))

    for i in range(1, N+1):
        for j in range(1, N+1):
            Aij = pidstava(A.T,i*krok) @ pidstava(A,j*krok)
            for a in range(0, k):
                for b in range(0, k):
                    matrix_zeros[a+k*(i-1),b+k*(j-1)] = Aij[a,b]
     
    return py.linalg.det(matrix_zeros)

def pseudo(A, b, T, v):
    P_0 = A @ A.T
    P = py.array([[float(sp.integrate(expr, (t, 0, T)).evalf()) for expr in row]for row in P_0])  
    xh = A.T @ py.linalg.pinv(P) @ b
    eps = b.T @ b -b.T @ P @ py.linalg.pinv(P) @ b
    Av_0 = A @ v
    Av = py.array([float(sp.integrate(expr, (t, 0, T)).evalf()) for expr in Av_0])
    return P, xh, eps, Av

def start():
    print("\n\n")

    N = 10
    T = 1
    A = py.array([[1,t],[t**2, 0]])
    b = py.array([1.5, 0.25])
    v = py.array([t, t**2])

    print('визначники')
    for i in range(1, N):
        det = Det(A, T, i)
        print(det)
        
    v = py.array([t, t])
    P, xh, eps, Av = pseudo(A, b, T, v)
    x = A.T @ py.linalg.pinv(P) @ b + v - A.T @ py.linalg.pinv(P) @ Av
    print('відповідь',x)
    print('псевдо розвязок',xh)
    print('точність',round(eps, 9))

    print("\n\n",'перевірка', 'b')
    perevirka_0 = A @ x
    perevirka = py.array([float(sp.integrate(expr, (t, 0, T)).evalf()) for expr in perevirka_0])
    print(perevirka)
    
if __name__ == "__main__":
   start()