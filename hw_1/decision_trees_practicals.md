Task 1.


```python
print('Task 1')
```

    Task 1
    


```python
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
```


```python
l1 = Tree.leaf("like")
l2 = Tree.leaf("nah")
tree_1 = Tree(data='morning?', left=l1, right=l2)
tree_2 = Tree(data='likedOtherSys?', left=l2, right=l1)
tree_3 = Tree(data='takenOtherSys?', left=tree_1, right=tree_2)
tree_4 = Tree(data='isSystems?', left=l1, right=tree_3)
```


```python
print(tree_4)
```

    Tree('isSystems?') { left = Leaf('like'), right = Tree('takenOtherSys?') { left = Tree('morning?') { left = Leaf('like'), right = Leaf('nah') }, right = Tree('likedOtherSys?') { left = Leaf('nah'), right = Leaf('like') } } }
    

Task 2.


```python
print('Task 2')
```

    Task 2
    


```python
import pandas as pd
```


```python
data = pd.read_csv('t2_data.csv')
```


```python
rat_bool = []
for row in data['rating']:
    if row >= 0:
        rat_bool.append(True)
    else:
        rat_bool.append(False)
```


```python
data['ok'] = rat_bool
```


```python
print(data)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>easy</th>
      <th>ai</th>
      <th>systems</th>
      <th>theory</th>
      <th>morning</th>
      <th>ok</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-1</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-1</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-1</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-1</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-2</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-2</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-2</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-2</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Task 3.


```python
print('Task 3')
```

    Task 3
    


```python
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
```


```python
features = data.columns.tolist()[1:-1]
```


```python
print('Best feature: ', best_feature(data, 'ok', features))
```

    Best feature:  ai
    


```python
def worst_feature(data, goal, features):
  # optional: avoid the lambda using `functools.partial`
  return min(features, key=lambda f: single_feature_score(data, goal, f))
```


```python
print('Worst feature: ', worst_feature(data, 'ok', features))
```

    Worst feature:  systems
    

Results are the following: "ai" is the best feature and "systems" is the worst one.

Task 4.


```python
print('Task 4')
```

    Task 4
    


```python
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
```


```python
trained_tree = DecisionTreeTrain(data, features)
```


```python
print(trained_tree)
```

    Tree('ai') { left = Tree('theory') { left = Tree('morning') { left = Leaf(False), right = Tree('easy') { left = Tree('systems') { left = Leaf(True), right = Leaf(False) }, right = Leaf(False) } }, right = Tree('easy') { left = Tree('morning') { left = Tree('systems') { left = Leaf(True), right = Leaf(False) }, right = None }, right = Leaf(True) } }, right = Tree('theory') { left = Tree('easy') { left = Tree('systems') { left = Leaf(True), right = Leaf(True) }, right = Tree('systems') { left = Leaf(True), right = Leaf(False) } }, right = Tree('easy') { left = Leaf(True), right = Tree('systems') { left = Leaf(True), right = Leaf(True) } } } }
    


```python
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
        
```


```python
print(DecisionTreeTest(trained_tree, data))
```

    False
    

The features that were chosen as the best and the worst in the task 3 are in the highest and lowest nodes in the tree that we've got in the DecisionTreetrain respectively, and if i got it right it makes sense.

Task 5.


```python
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
        
```


```python
trained_tree_2 = DecisionTreeTrain_2(data, features, 3)
```


```python
print(trained_tree_2)
```

    Tree('ai') { left = Tree('theory') { left = Leaf('morning'), right = Leaf('easy') }, right = Tree('theory') { left = Leaf('easy'), right = Leaf('easy') } }
    


```python
print("Tree's depth = ", trained_tree_2.depth())
```

    Tree's depth =  3
    
