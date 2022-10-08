 ## main1 : \
 Diabetes Dataset on ML Practise \
 
 ### main2 : \
 
 
 ## classifier1 : 
 Classifier Problem on ML Practise : Originally used as an example data set on which Fisher's linear discriminant analysis was applied, it became a typical test case for many statistical classification techniques in machine learning such as support vector machines

## Overview of the App

<table>
<tr>
<td>
This application can be used by Flipkart to get a detailed overall sentiment analysis of customer reviews for Flipkart products scrapped from multiple social media platforms like - Twitter, LinkedIn, Reddit. It also contains a section which displays the top 90 Electronics Products extracted from various social media platforms like Facebook and Instagram along with the details of the products taken from Flipkart. Additionally the application also gives meaningful insights from the analysis shown of the products sold and customer feedbacks during Flipkart Big Billion Days.
</td>
</tr>
</table>

There are 3 main sections in the app as follows -

1. <b>Social Media</b> - This section contains a total of 6 plots to display the analysis of sentiment counts and percentages via Histograms and Pie Charts for tweets, posts, contents extacted from social media platforms like Twitter, Reddit, LinkedIn using web scrapping tools like ParseHub, Apify. The data scrapped from different social media platforms are passed through a Recurrent Neural Network Sentiment Anlaysis Model with 99.2% accuracy to predict the results.

2. <b>Top Products</b> - This section contains top 90 trendy electronics products on social media platforms like Facebook, Instagram using web scrapping tools like ParseHub, Apify and the details of the respective products taken from official site of Flipkart. The data scrapped from multiple social media platforms is used to match with products on flipkart and then details of those products are extracted.

3. <b>Big Billion Days Sale</b> - This section contains a total of 6 plots to showcase the insights of sale of products and products categories sold along with customer feedbacks and reviews for the same. The dataset is collected for Big Billion Days 2021 and analysis is drawn.

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
