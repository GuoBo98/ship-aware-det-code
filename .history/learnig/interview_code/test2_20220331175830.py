'''
求解 x**2 - C = 0 ----> y = x**2 - C
初始值 C = x0 , (x0 , x0^2 - C ) --> y' = 2*x0
过初始点求导 y = 2*x0(x-x0) + x0**2 - C 
        --->  = 2*x0*x - x0**2 - C = 0
                                 x = (x0 ** 2 +C )/(2*x0)

'''
from cv2 import sqrt
import time


def my_sqrt(x , s):
    C = x
    res = x 
    while abs(res**2 - x) > s**2:
        res = (res ** 2 + C )  / (2 * res)
    print(res)

a = time.time()
print(sqrt(1000))
b = time.time()
lib_time = b-a

a = time.time()
my_sqrt(1000 , 1e-5)
b = time.time()
my_time = b - a

print(my_time - lib_time)


        