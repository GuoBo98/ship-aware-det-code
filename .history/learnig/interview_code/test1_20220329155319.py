def sqrt(x):
    y = x / 3
    res = 0
    while abs(res - x) > 1e-3:
        res = y ** 2
        if res > x :
            y = y - 1e-4
        else:
            y = y + 1e-4
    return y

print(sqrt(2))