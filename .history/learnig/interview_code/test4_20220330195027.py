[m , n ] = list(map(int,input().split()))
[start_x , start_y] = list(map(int,input().split()))
[stop_x , stop_y] = list(map(int,input().split()))
nums = int(input())
lakes = []
for i in range(nums):
    lakes.append(list(map(int,input().split())))
res , on_path = [],[]



def dfs(x,y):
    
    if [x,y] in lakes:
        return
    if x == stop_x and y == stop_y:
        res.append(on_path)
        return
    on_path.append([x,y])
    dfs(x+1,y)
    dfs(x,y+1)
    dfs(x-1,y)
    dfs(x,y-1)
    on_path.pop()


dfs(start_x,start_y)
print(1)