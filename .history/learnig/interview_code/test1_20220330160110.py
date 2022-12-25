def maxArea(height) -> int:
    left = 0 
    right = len(height) - 1 
    res = []
    while left <= right:
        area = min(height[left] , height[right]) * (right - left)
        if height[left] < height[right]:
            left = left + 1
        if height[left] >= height[right]:
            right = right - 1
        res.append(area)
    return max(res)

maxArea([1,8,6,2,5,4,8,3,7])
