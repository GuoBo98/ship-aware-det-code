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
    while low < high and abs(res - x) > 1e-3:
        mid = (low + high) / 2
        if mid ** 2 <= x :
            res = mid ** 2
            low = mid 
        else:
            high = mid 
            
    print(mid)

sqrt_my(3)
print(math.sqrt(3))