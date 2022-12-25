N =  int(input())
nums = list(map(int,input().split()))
new_nums = []
for i in range(N-1):
    new_nums.append(abs(nums[i] - nums[i+1]))
    
flag = 0
min_num = min(new_nums)
max_num = max(new_nums)

count = 0
for i in range(min_num,max_num+1):
    if i not in new_nums:
        count = count + 1
        flag = 1 
        
set_lst=set(new_nums)

if len(set_lst) != len(new_nums) or flag == 1:
    #最大重复次数
    max_times = new_nums.count(max(new_nums,key=new_nums.count))
    print('NO')
    print(max_times,count)
else:
    print('YES')
    print(min_num,max_num)
    

