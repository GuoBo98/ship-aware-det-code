a , b ,x , y = list(map(int,input().split()))

if (not ((a >= x and b >=y) or (a >= y and b>=x))):
    print(0)
else:
    dp = [[0] * (b+1) for i in range(a+1)]
    for i in range(a+1):
        for j in range(b+1):
            if (not ((i >= x and j >=y) or (i >= y and j>=x))):
                dp[i][j] = 0
            elif i >= x and j >=y and i >= y and j >= x :
                dp[i][j] = max(dp[i-x][j-y] , dp[i-y][j-x]) + 1
            elif i >=x and j >= y :
                dp[i][j] = dp[i-x][j-y] + 1 
            else:
                dp[i][j] = dp[i-y][j-x] + 1 
print(dp)
                