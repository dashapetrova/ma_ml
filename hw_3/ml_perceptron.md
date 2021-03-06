Task 1.


```
from typing import Union, List
from math import sqrt

class Scalar:
  pass
class Vector:
  pass

class Scalar:

  def __init__(self: Scalar, val: float):
    self.val = float(val)

  def __mul__(self: Scalar, other: Union[Scalar, Vector]) -> Union[Scalar, Vector]:
    # hint: use isinstance to decide what `other` is
    # raise an error if `other` isn't Scalar or Vector!
    if isinstance(other, Scalar):
      return Scalar(self.val*other.val)
    elif isinstance(other, Vector):
      return Vector(*[self.val*i for i in other])
    else:
      return NotImplemented

  def __add__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val + other.val)

  def __sub__(self: Scalar, other: Scalar) -> Scalar:
    return Scalar(self.val - other.val)

  def __truediv__(self: Scalar, other: Scalar) -> Scalar:
    # implement division of scalars
    return Scalar(self.val / other.val)

  def __rtruediv__(self: Scalar, other: Vector) -> Vector:
    # implement division of vector by scalar
    return Vector(*[self.val / i for i in other])

  def __repr__(self: Scalar) -> str:
    return "Scalar(%r)" % self.val

  def sign(self: Scalar) -> int:
    # returns -1, 0, or 1
    if self.val == 0:
      return 0
    elif self.val < 0:
      return -1
    else:
      return 1

  def __float__(self: Scalar) -> float:
    return self.val

class Vector:

  def __init__(self: Vector, *entries: List[float]):
    self.entries = entries

  def zero(size: int) -> Vector:
    return Vector(*[0 for i in range(size)])

  def __add__(self: Vector, other: Vector) -> Vector:
    return Vector(*[i + j for i, j in zip(self.entries, other.entries)])

  def __sub__(self: Vector, other: Vector) -> Vector:
    return Vector(*[i - j for i, j in zip(self.entries, other.entries)])

  def __mul__(self: Vector, other: Vector) -> Scalar:
    res = 0
    for i in range(len(self.entries)):
      res += (self.entries[i] * other.entries[i])
    return Scalar(res)

  def magnitude(self: Vector) -> Scalar:
    return sum([i ** 2 for i in self.entries])**0.5

  def unit(self: Vector) -> Vector:
    return self / self.magnitude()

  def __len__(self: Vector) -> int:
    return len(self.entries)

  def __repr__(self: Vector) -> str:
    return "Vector%s" % repr(self.entries)
    
  def __iter__(self: Vector):
    return iter(self.entries)
```

Task 2.


```
def PerceptronTrain(D, max_iter = 100):
  
  w = Vector.zero(len(D[0][0]))
  b = Scalar(0)

  for i in range(max_iter):
    for x, y in D:
      activ = x * w + b # activation
      s = (y * activ).sign() # sign check
      if s <= 0: 
        w += y * x
        b += y

  return w, b
```


```
def PerceptronTest(D, w, b):
  
  results = []
  for x in D:
    activ = x * w + b
    results.append(activ.sign())
  
  return results
```

Task 3.


```
from random import randint
v = Vector(randint(-100, 100), randint(-100, 100))
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs]
```


```
#90-10 test-train split

border = int(0.9 * len(xs))

def split_data(mas, border):
  return mas[:border], mas[border:]

X_train, X_test = split_data(xs, border)
y_train, y_test = split_data(ys, border)
```


```
data = list(zip(X_train, y_train))

#train perceptron
weights, bias = PerceptronTrain(data)
```


```
#get predictions
y_preds = PerceptronTest(X_test, weights, bias)
```


```
corr = 0

for i in range(len(y_test)):
  if y_test[i].sign() == y_preds[i]:
    corr += 1

acc_score = corr / len(y_test)
print(f'Accuracy score = {acc_score}')
```

    Accuracy score = 0.88
    

Accuracy score is 0.88

Task 4.


```
from random import randint
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [Scalar(1) if x.entries[0]*x.entries[1] < 0 else Scalar(-1) for x in xs]
```


```
#90-10 test-train split

border = int(0.9 * len(xs))

def split_data(mas, border):
  return mas[:border], mas[border:]

X_train, X_test = split_data(xs, border)
y_train, y_test = split_data(ys, border)
```


```
data = list(zip(X_train, y_train))

#train perceptron
weights, bias = PerceptronTrain(data)
```


```
#get predictions
y_preds = PerceptronTest(X_test, weights, bias)

corr = 0

for i in range(len(y_test)):
  if y_test[i].sign() == y_preds[i]:
    corr += 1

acc_score = corr / len(y_test)
print(f'Accuracy score = {acc_score}')
```

    Accuracy score = 0.48
    

Accuracy score is 0.48, so the model really performs worse than in task 3.

Task 5.


```
from random import shuffle
```


```
#let's add shuffle function to the function
#test function remains the same
def PerceptronTrain(D, max_iter = 100, shuff = False):
  
  w = Vector.zero(len(D[0][0]))
  b = Scalar(0)

  for i in range(max_iter):
    if shuff == True:
      shuffle(D)
    for x, y in D:
      activ = x * w + b # activation
      s = (y * activ).sign() # sign check
      if s <= 0: 
        w += y * x
        b += y

  return w, b
```


```
v = Vector(randint(-100, 100), randint(-100, 100))
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs]
```


```
#90-10 test-train split

border = int(0.9 * len(xs))

def split_data(mas, border):
  return mas[:border], mas[border:]

X_train, X_test = split_data(xs, border)
y_train, y_test = split_data(ys, border)
```


```
data = list(zip(X_train, y_train))
```


```
data_sorted = sorted(data, key= lambda x: x[1].val)
```


```
def acc_score(y_preds):

  corr = 0

  for i in range(len(y_test)):
    if y_test[i].sign() == y_preds[i]:
      corr += 1

  acc_score = corr / len(y_test)

  return acc_score
```


```
no_permute = []
begin_permute = []
each_ep_permute = []

for i in range(0,100,5):
  #no permutation
  weights, bias = PerceptronTrain(data_sorted, i)
  y_preds = PerceptronTest(X_test, weights, bias)
  cur_score = acc_score(y_preds)
  no_permute.append(cur_score)

  #permutation at the beginning
  shuffle(data_sorted)
  weights, bias = PerceptronTrain(data_sorted, i)
  y_preds = PerceptronTest(X_test, weights, bias)
  cur_score = acc_score(y_preds)
  begin_permute.append(cur_score)

  #permutation at each epoch
  weights, bias = PerceptronTrain(data_sorted, i, True)
  y_preds = PerceptronTest(X_test, weights, bias)
  cur_score = acc_score(y_preds)
  each_ep_permute.append(cur_score)
```


```
import matplotlib.pyplot as plt
```


```
x_axis = [x for x in range(1, 100, 5)]

plt.plot(x_axis, no_permute, color='green', label='no permutation', marker='o')
plt.plot(x_axis, begin_permute, color='blue', label='random permutation at the beginning', marker='o')
plt.plot(x_axis, each_ep_permute, color='orange', label='random permutation at each epoch', marker='o')

plt.xlabel('Number of epoch', fontsize=12)
plt.ylabel('Accuracy score', fontsize=12)
plt.legend()
plt.show()
```


    
![png](output_28_0.png)
    


Task 6.


```
def AveragedPerceptronTrain(D, max_iter = 100):

  w = Vector.zero(len(D[0][0]))
  b = Scalar(0)
  
  u = Vector.zero(len(D[0][0]))
  be = Scalar(0)

  c = Scalar(1)

  for i in range(max_iter):
    shuffle(D)
    for x, y in D:
      activ = x * w + b
      s = (y * activ).sign()
      if s <= 0: 
        w += y * x
        b += y
        u += y * c * x
        be += y * c
      c += Scalar(1)

  return w - (Scalar(1)/c) * u, b - (Scalar(1)/c) * be
```


```
weights, bias = PerceptronTrain(data, shuff = True)
y_preds = PerceptronTest(X_test, weights, bias)
score_norm = acc_score(y_preds)

weights_av, bias_av = AveragedPerceptronTrain(data)
y_preds_av = PerceptronTest(X_test, weights_av, bias_av)
score_av = acc_score(y_preds_av)
```


```
print(f'Normal train score = {score_norm}')
print(f'Averaged train score = {score_av}')
```

    Normal train score = 0.86
    Averaged train score = 0.88
    

Normal train score = 0.86

Averaged train score = 0.88

Performance of the models is nearly the same.
