
import numpy as np

[n, m] = list(map(int, input().split()))
red_idxs = list(map(int, input().split()))
blue_idxs = list(map(int, input().split()))
max_indx = max(max(red_idxs), max(blue_idxs))
nums = [i for i in range(max_indx+1)]
nums = np.array(nums)

q = int(input())
res = [0, 0, 0]
for _ in range(q):
    [l, r] = list(map(int, input().split()))
    b_n, r_n = 0, 0
    for i in range(l, r+1):
        if i in blue_idxs:
            b_n = b_n + 1
        elif i in red_idxs:
            r_n = r_n + 1
    if r_n > b_n:
        res[0] += 1
    elif r_n == b_n:
        res[1] += 1
    elif r_n < b_n:
        res[2] += 1
print(' '.join(str(item) for item in res))