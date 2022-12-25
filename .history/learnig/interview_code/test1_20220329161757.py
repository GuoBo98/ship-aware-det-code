# def sqrt(x):
#     y = x / 3
#     res = 0
#     while abs(res - x) > 1e-3:
#         res = y ** 2
#         if res > x :
#             y = y - 1e-4
#         else:
#             y = y + 1e-4
#     return y

# print(sqrt(2))
import math



def sqrt_my(x):
    low = 0 
    high = x 
    res = -1
    mid = 0
    while low < high and abs(res - x) > 1e-3:
        mid = (low + high) / 2
        res = mid ** 2
        if mid ** 2 <= x :    
            low = mid
        else:
            high = mid            
    print(mid)

print(math.sqrt(0.01))
sqrt_my(0.01)

def sqrt_my(x):
    low = 0 
    high = 1
    res = -1
    mid = 0
    while low < high and abs(res - x) > 1e-3:
        mid = (low + high) / 2
        res = mid ** 2
        if mid ** 2 <= x :    
            low = mid
        else:
            high = mid            
    print(mid)

sqrt_my(0.01)