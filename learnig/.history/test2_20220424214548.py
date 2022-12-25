


''' 第二题，非numpy版本 '''
def mat_mul(a, b):
    r = len(a)
    c = len(b[0])
    assert len(a[0]) == len(b)

    res = [[0] * c for _ in range(r)]
    for i in range(r):
        for j in range(c):
            for k in range(len(b)):
                res[i][j] += a[i][k] * b[k][j]
    return res

c = int(input())
f = list(map(float, input().split()))
w = [[0] * c for _ in range(c)]
for i in range(c):
    w[i] = list(map(float, input().split()))

# 不需要做 GAP 了
f = [[i] for i in f]
res = mat_mul(w, f)
idx = 0
for i in range(len(res)):
    if res[i] > res[idx]:
        idx = i
print(i)



'''
2
0.3 0.4
0.4 0.6
0.5 0.7

1
'''