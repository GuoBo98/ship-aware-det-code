from matplotlib import collections


nums = input().strip('[').strip(']').split(',')
'''构建二叉树'''
class TreeNode():
    def __init__(self , val ,left = None,right = None) -> None:
        self.val = val 
        self.left = left
        self.right = right

def build_tree(nums):
    nodes = []
    for val in nums:
        if val in ('null' , '#'):
            nodes.append(None)
        else:
            nodes.append(TreeNode(val))
    for i in range(len(nodes)):
        if nodes[i]:
            if i*2+1 < len(nodes):
                nodes[i].left = nodes[i*2+1]
            if i*2+2 < len(nodes):
                nodes[i].right = nodes[i*2+2]
    return nodes[0]


root = build_tree(nums)
count = collections.Counter()


    
    