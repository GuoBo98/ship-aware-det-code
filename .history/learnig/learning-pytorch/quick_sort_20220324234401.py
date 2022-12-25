def partition(nums , left ,right):
    if left < right :
        i , j , x = left , right , nums[left]
        while i < j:
            #从右往左找小于x的数 
            while i < j and nums[j] >= x :
                j = j - 1
            if i < j :
                nums[i] = nums[j] 
                i = i + 1
            
            #从左往右找大于x的数
            while i < j and nums[i] <= x :
                i = i + 1
            if i < j :
                nums[j] = nums[i] 
                j = j - 1
        nums[i] = x 
        return i 

def quick_sort(nums , left , right):

    if left < right:
        i = partition(nums , left ,right)
        quick_sort(nums , left, i-1)
        quick_sort(nums , i+1, right)

nums = [10, 7, 8, 9, 1, 5]     
left = 0 
right = len(nums) - 1
quick_sort(nums , left ,right)
print(nums)        
            
            
    