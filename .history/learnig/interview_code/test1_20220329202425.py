s = str(input())

n = len(s)

if not ('0' in s and '1' in s) :
    print(0)
elif n < 2 :
    print(len(s))
else:
    max_len = 1 
    start = 0 
    dp = [[False] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = True

    for idx in range(2, n+ 1):
        for i in range(n):
            j = idx + i - 1
            if j >= n :
                break
            if s[i] != s[j]:
                dp[i][j] = False
            
            else:
                if j - i < 3 :
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i+1][j-1]
                    
            if dp[i][j] and j - i + 1  > max_len:
                max_len = j - i + 1
                start = i 
print(max_len)
            
    