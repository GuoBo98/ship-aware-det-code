n = map(int,input())
nums = list(map(int,input().split()))

for i in range(len(nums)):
    a = nums[i]
    if n-a in nums[i:]:
        print(a,n-a)

