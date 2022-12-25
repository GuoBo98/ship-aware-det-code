import collections
from typing import Counter


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
counter = collections.Counter()
res = []

def traverse(root:TreeNode):
    if not root:
        return '#'
    left = traverse(root.left)
    right = traverse(root.right)
    chain = left + ',' + right + ',' + str(root.val)
    counter[chain] +=1
    if counter[chain] == 2 :
        res.append(root)
    return chain

def serilize(root:TreeNode):
    if not root:
        return 'Null'
    return str(root.val) + ',' + serilize(root.left) + ',' + serilize(root.right)
traverse(root)
res1 = serilize(res[-1])
res2 = []
for item in res1:
    res2.append(item)
    
    


    
    