for _ in range(T):
    n, k = list(map(int, input().split()))
    arr = list(map(int, input().split()))
    if k == 1:
        print(n)
        break
    mono = []
    for val in arr:
        if not mono:
            mono.append(val)
            # break
        i = 0
        
        while i < len(mono):
            if mono[i] >= val:
                mono[i] = val
                break
            i += 1
        if i == len(mono):
            # if len(mono) >= k-1:
            #     res += 1
            # else:
            mono.append(val)
    
    res = 0
    if len(mono) >= k:
        remain = len(mono) - k + 1
        for _ in range(remain):
            if mono[0] >= mono[-1]:
                res += mono[-1]
                mono.pop(-1)
            else:
                res += mono[0]
                mono.pop(0)

    print(res)