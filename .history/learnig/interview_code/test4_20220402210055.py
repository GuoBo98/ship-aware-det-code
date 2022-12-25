T = int(input())

for _ in range(T):
    n, k = list(map(int, input().split()))
    arr = list(map(int, input().split()))
    if k == 1:
        print(n)
    else:
        memory = []
        for val in arr:
            if not memory:
                memory.append(val)
                # break
            i = 0
            
            while i < len(memory):
                if memory[i] >= val:
                    memory[i] = val
                    break
                i += 1
            if i == len(memory):
                # if len(memory) >= k-1:
                #     res += 1
                # else:
                memory.append(val)
        
        res = 0
        if len(memory) >= k:
            remain = len(memory) - k + 1
            for _ in range(remain):
                if memory[0] >= memory[-1]:
                    res += memory[-1]
                    memory.pop(-1)
                else:
                    res += memory[0]
                    memory.pop(0)

        print(res)