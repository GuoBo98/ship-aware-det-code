def my_sqrt(x , s):
    s = 1e-3
    res = x
    C = x 
    while abs(res ** 2 - x) > s**2:
        res = ((res ** 2) + C) / (2 * res)
    print(res)

my_sqrt(0,1e-3)
        