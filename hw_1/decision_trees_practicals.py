#!/usr/bin/env python
# coding: utf-8

# Task 1.

# In[270]:


print('Task 1')


# In[1]:


class Tree:
    def leaf(data):
    #Create a leaf tree
        return Tree(data=data)

  # pretty-print trees
    def __repr__(self):
        if self.is_leaf():
            return "Leaf(%r)" % self.data
        else:
            return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right) 

  # all arguments after `*` are *keyword-only*!
    def __init__(self, *, data = None, left = None, right = None):
        self.data = data
        self.left = left
        self.right = right

    def is_leaf(self):
    #Check if this tree is a leaf tree
        return self.left == None and self.right == None

    def children(self):
    #List of child subtrees
        return [x for x in [self.left, self.right] if x]

    def depth(self):
    #Compute the depth of a tree
    #A leaf is depth-1, and a child is one deeper than the parent.
        return max([x.depth() for x in self.children()], default=0) + 1


# In[2]:


l1 = Tree.leaf("like")
l2 = Tree.leaf("nah")
tree_1 = Tree(data='morning?', left=l1, right=l2)
tree_2 = Tree(data='likedOtherSys?', left=l2, right=l1)
tree_3 = Tree(data='takenOtherSys?', left=tree_1, right=tree_2)
tree_4 = Tree(data='isSystems?', left=l1, right=tree_3)


# In[3]:


print(tree_4)


# Task 2.

# In[271]:


print('Task 2')


# In[5]:


import pandas as pd


# In[6]:


data = pd.read_csv('t2_data.csv')


# In[8]:


rat_bool = []
for row in data['rating']:
    if row >= 0:
        rat_bool.append(True)
    else:
        rat_bool.append(False)


# In[9]:


data['ok'] = rat_bool


# In[10]:


print(data)


# Task 3.

# In[272]:


print('Task 3')


# In[11]:


def single_feature_score(data, goal, feature):
    preds = list(data[feature])
    true = list(data[goal])
    correct = 0
    for i in range(len(preds)):
        if preds[i] == true[i]:
            correct += 1
    return correct/len(preds)

def best_feature(data, goal, features):
  # optional: avoid the lambda using `functools.partial`
  return max(features, key=lambda f: single_feature_score(data, goal, f))


# In[290]:


features = data.columns.tolist()[1:-1]


# In[273]:


print('Best feature: ', best_feature(data, 'ok', features))


# In[15]:


def worst_feature(data, goal, features):
  # optional: avoid the lambda using `functools.partial`
  return min(features, key=lambda f: single_feature_score(data, goal, f))


# In[274]:


print('Worst feature: ', worst_feature(data, 'ok', features))


# Results are the following: "ai" is the best feature and "systems" is the worst one.

# Task 4.

# In[275]:


print('Task 4')


# In[278]:


def DecisionTreeTrain(data, remaining_features):
    c = data['ok'].value_counts()
    try:
        guess = c[c == max(data['ok'].value_counts())].index.to_list()[0]
    except:
        return None
        
    if remaining_features is None or remaining_features == []: #emptiness check
        return Tree.leaf(guess)
    
    stats = data[remaining_features].value_counts().to_dict()
    if len(stats.keys()) == 1: #unambiguous check
        return Tree.leaf(guess)
    else:
        results = []
        for f in remaining_features:
            true = data[data[f] == True]
            false = data[data[f] == False]
            pos_num = neg_num = 0
            for i in true['ok']:
                if i == True:
                    pos_num += 1
            for i in false['ok']:
                if i == False:
                    neg_num += 1
            score = pos_num + neg_num
            results.append([f, score])
        best_f = sorted(results, key=lambda i: i[1], reverse = True)[0][0]
        true = data[data[best_f] == True]
        false = data[data[best_f] == False]
        best_f_id = remaining_features.index(best_f)
        remaining_features = [x for i,x in enumerate(remaining_features) if i!=best_f_id]
        
        right = DecisionTreeTrain(true, remaining_features)
        left = DecisionTreeTrain(false, remaining_features)
        
        return Tree(data=best_f, left=left, right=right)


# In[279]:


trained_tree = DecisionTreeTrain(data, features)


# In[280]:


print(trained_tree)


# In[264]:


def DecisionTreeTest(tree, test_point):
    if tree.is_leaf() == True:
        return tree.data
    else:
        f = tree.data
        left = tree.left
        right = tree.right
        if test_point[test_point[f] == False] is not None:
            return DecisionTreeTest(left, test_point)
        else:
            return DecisionTreeTest(right, test_point)
        


# In[281]:


print(DecisionTreeTest(trained_tree, data))


# The features that were chosen as the best and the worst in the task 3 are in the highest and lowest nodes in the tree that we've got in the DecisionTreetrain respectively, and if i got it right it makes sense.

# Task 5.

# In[292]:


def DecisionTreeTrain_2(data, remaining_features, max_depth):
    if max_depth == 0:
        return None
    c = data['ok'].value_counts()
    try:
        guess = c[c == max(data['ok'].value_counts())].index.to_list()[0]
    except:
        guess = None
        
    if remaining_features is None or remaining_features == []: #emptiness check
        return Tree.leaf(guess)
    
    stats = data[remaining_features].value_counts().to_dict()
    if len(stats.keys()) == 1: #unambiguous check
        return Tree.leaf(guess)
    else:
        results = []
        for f in remaining_features:
            true = data[data[f] == True]
            false = data[data[f] == False]
            pos_num = neg_num = 0
            for i in true['ok']:
                if i == True:
                    pos_num += 1
            for i in false['ok']:
                if i == False:
                    neg_num += 1
            score = pos_num + neg_num
            results.append([f, score])
        best_f = sorted(results, key=lambda i: i[1], reverse = True)[0][0]
        true = data[data[best_f] == True]
        false = data[data[best_f] == False]
        best_f_id = remaining_features.index(best_f)
        remaining_features = [x for i,x in enumerate(remaining_features) if i!=best_f_id]
        
        sub_depth = max_depth-1
        right = DecisionTreeTrain_2(true, remaining_features, sub_depth)
        left = DecisionTreeTrain_2(false, remaining_features, sub_depth)
            
        return Tree(data=best_f, left=left, right=right)
        


# In[293]:


trained_tree_2 = DecisionTreeTrain_2(data, features, 3)


# In[295]:


print(trained_tree_2)


# In[297]:


print("Tree's depth = ", trained_tree_2.depth())

