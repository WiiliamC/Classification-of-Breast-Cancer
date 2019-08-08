# Classification-of-Breast-Cancer
---
 
to be updated
 
---
## Contents
* ### [1.Introduction](1.Introduction)
* ### [2.Prepare](2.Prepare)
* ### [3.How_to_train_your_own_model](3.How_to_train_your_own_model)
* ### [4.How_to_use_your_own_model_to_predict](4.How_to_use_your_own_model_to_classify_a_single_image)
* ### [5.How_to_run_prediction_on_Intel_NCS2](4.How_to_use_your_own_model_to_classify_a_single_image)
* ### [6.Expectation](4.How_to_use_your_own_model_to_classify_a_single_image)
## 1.Introduction
In this program, we proposed a method for breast cancer classification using Transfer Learning (TL). In our CNN, InceptionV3 is imported as a feature extractor. It is used to preprocess our data which comes from BreaKHis, a public data set available at http://web.inf.ufpr.br/vri/breast-cancer-database. 
Download and unzip "project.zip", and you'll find three files inside: "data set", "InceptionV3" and "run.py".
In "data set", there are 7909 breast cancer histopathological images from BreaKHis, a public data set available at http://web.inf.ufpr.br/vri/breast-cancer-database. 
"InceptionV3" contains a pre-trained .pb CNN model provided by Google. In our program, we use InceptionV3 as a feature extractor. It will tranfer an image to its feature vector. This process is called "preprocess"
![Preprocess](https://img-blog.csdnimg.cn/20190806170317582.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDM0MDE5NA==,size_16,color_FFFFFF,t_70)
After preprocessing, all the feature vectors (.txt) are stored in a new folder called "bottlenecks".
Next, we take 80% of the .txt files randomly as our training data and the left 20% as testing data. We put them respectively in two new folders named "train" and "test". In the "test" folder, there are 4 folders named "40x", "100x", "200x" and "400x". This is to make testing operation easier.
Finally, these feature vectors are loaded. After training and testing, you'll get a .pb model and two images in a new folder named "output".
## 2.Prepare
|python|3.6  |
|--|--|
| tensorflow / tensorflow-gpu |1.13.1  |
| RAM |32GB|
## 3.How_to_train_your_own_model
1.Unzip "project"
2.Run "run.py"
3.After the program is finished, you can find your .pb model in the generated output folder.
## 4.How_to_use_your_own_model_to_predict
***To be updated***
## 5.How_to_run_prediction_on_Intel_NCS2
***To be updated***
## 6.Expectation
***To be updated***

