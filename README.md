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
In this project, we proposed a method for breast cancer classification using Transfer Learning (TL). In our CNN, InceptionV3 is imported as a feature extractor. It is used to preprocess our data which comes from BreaKHis, a public data set available at http://web.inf.ufpr.br/vri/breast-cancer-database. It contains 7909 breast cancer images.  
InceptionV3 is a powerful image classification CNN provided by Google. In our project, we use InceptionV3 as a feature extractor. It will tranfer an image to its feature vector. This process is called "preprocess".  
![Preprocess](https://img-blog.csdnimg.cn/20190806170317582.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDM0MDE5NA==,size_16,color_FFFFFF,t_70)  
After preprocessing, all the feature vectors (.txt) are stored in a new folder called "bottlenecks".  
Next, we take 80% of the .txt files randomly as our training data and the left 20% as testing data. We put them respectively in two new folders named "train" and "test". In the "test" folder, there are 4 folders named "40x", "100x", "200x" and "400x". This is to make testing operation easier.  
Finally, these feature vectors are loaded. After training and testing, you'll get a .pb model in a new folder named "output".
In case some problem arises and stops you from training your own model, I'll upload a model I trained, THE ACCURACY OF WHICH REACHES AROUND 95%.  
After training/downloading the model, you can try to do prediction with it.  
Furthermore, if you've installed OpenVINO, you can try to do inference on your CPU or Intel NCS2. In this way, you can realize breast cancer classification on a raspberry pi with an Intel NCS2.  
## 2.Prepare
#### 1. Requirement
|python|3.6  |
|--|--|
| tensorflow / tensorflow-gpu |1.13.1  |
| RAM |32GB|
#### 2. Prepare your project folder
1.Download run.py and predict.py and put them in a new folder. Let's call this folder "project folder".  
2.Download BreakHis data set from http://web.inf.ufpr.br/vri/breast-cancer-database. Put all the 7909 images in a folder called "data set". Move this folder into your project folder and make sure there is nothing else except breast cancer images (but images can be put in sub-folders).  
3.Download googlenet-v3.frozen.pb and put it in a sub-folder called "InceptionV3". Now your project folder is ready.  

## 3.How_to_train_your_own_model
Just run run.py and be patient. The time cost depends on your device.  
After the program is finished, you'll get a BC-Classifier.pb in a folder called "output". This is your own model!  
## 4.How_to_use_your_own_model_to_predict
Run predict.py. You need to enter the directory of your model and the image to be predicted. By default, the directory of model is "cwd\output\BC-Classifier.pb" and of image is "cwd\image to be predicted\image.png". "cwd" means current work directory, which should be your project folder.  
## 5.How_to_run_prediction_on_Intel_NCS2
***To be updated***
## 6.Expectation
***To be updated***

