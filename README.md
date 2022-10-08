 ## All the my learnings and information regarding ML and small practise projects are pushed in this repo !
 
 ###  1. main : 
 Diabetes Dataset on ML ( Practise )
 
 ### 2. main2 : 
  Diabetes Dataset on ML ( Practise )
 
 ### 3. classifier1 : 

<table>
<tr>
<td>
 Originally used as an example data set on which Fisher's linear discriminant analysis was applied, it became a typical test case for many statistical classification techniques in machine learning such as support vector machines 
 </td>
</tr>
</table>
Here I have used the <b>iris flower dataset</b> or <b>Fisher's Iris data set</b> to predict the flower species by its different labels like (1) sepal length, (2) sepal width, (3) petal length, (4) petal width and (5) species 

```
from sklearn.datasets import load_iris

iris = load_iris()
iris
```
This code gives:

```
{'data': array([[5.1, 3.5, 1.4, 0.2],
                [4.9, 3. , 1.4, 0.2],
                [4.7, 3.2, 1.3, 0.2],
                [4.6, 3.1, 1.5, 0.2],...
'target': array([0, 0, 0, ... 1, 1, 1, ... 2, 2, 2, ...
'target_names': array(['setosa', 'versicolor', 'virginica'], dtype='<U10'), 
...}
```
