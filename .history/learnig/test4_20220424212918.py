
import numpy as np

[n, m] = list(map(int, input().split()))
red_idxs = list(map(int, input().split()))
blue_idxs = list(map(int, input().split()))
max_indx = max(max(red_idxs), max(blue_idxs))
nums = [-1 for i in range(max_indx+1)]
nums = np.array(nums)
nums[red_idxs] = 1
nums[blue_idxs] = 0

q = int(input())
res = [0, 0, 0]
for _ in range(q):
    [l, r] = list(map(int, input().split()))
    nums_test = nums[l:r+1]
    r_n = nums_test.count(1)
    b_n = nums_test.count(0)
    if r_n > b_n:
        res[0] += 1
    elif r_n == b_n:
        res[1] += 1
    elif r_n < b_n:
        res[2] += 1
print(' '.join(str(item) for item in res))