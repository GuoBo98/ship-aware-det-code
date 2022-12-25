def sqrt(x):
    y = x / 5
    res = y ** 2
    while abs(res - x) <= 1e-3:
        if res > x :
            y = y - 1e-4
        else:
            y = y + 1e-4
    return y

print(sqrt(2))