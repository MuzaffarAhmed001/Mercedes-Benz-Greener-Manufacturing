# Mercedes-Benz-Greener-Manufacturing
Predicting Car Testing Time using Machine learning.

1. Business problem

The aim is to reduce testing time by analyzing the currently available data which is collected from hundreds of tests on thousands of car configurations. There might be cases that two cars belong to same Class but differ in one or more feature. So there testing time will be different. Hence for such cases, the machine learning model can help to predict accurate time spent on a test bench for cars with the same class but some different features.

The basic problem statement is to create a machine learning model that will predict the accurate time a car spends on the test bench and tackle the curse of dimensionality.

2. Mapping to Machine Learning problem

This is Regression problem in which we need to predict testing time in which car pass the test.Performance Metric used here is R-squared( Coefficient of Determination).

3. Data Overview

The ground truth is labeled 'y' and represents the time (in seconds) that the car took to pass testing for each variable.

Train.csv: It contains 4209 data points and 378 features.

Test.csv: It contains 4209 data points and 377 features. We need to predict the 'y' variable for the 'ID's in this file.

There are 8 Categorical Features, 369 Binary Features and 1 Float Feature(Dependent Variable).

4. Exploratory Data Analysis:

Target variable Analysis

![target_var_disb](https://user-images.githubusercontent.com/67824198/150626081-e01ac92b-477a-48f0-8186-eb1fcd47ebe7.PNG)

From above plot, we can see that distribution of target variable is skewed and upon converting to Normal distribution will give good results

![target_var_scatter](https://user-images.githubusercontent.com/67824198/150626112-9b10937b-a4dc-4196-b8b5-0b3860caaa94.PNG)

From scatter plot and box plot of target variable, we can see that there are outliers in target variable and which are removed from the dataset to improve model performance.

Categorical Variable Analysis

There are 8 categorical features in our dataset from X0 to X8.

![cat_box](https://user-images.githubusercontent.com/67824198/150626156-9ea6aab8-13be-4333-a869-6957a180be46.png)

From above box plot of categorical variables, we can see that X4 is having low variance. Hence we can remove X4 feature from our dataset.

Analysis of Binary Features

There are 368 binary features in our dataset from X10 to X385.

![Binary-feat](https://user-images.githubusercontent.com/67824198/150626178-458a273e-c6bc-4103-ab2c-920b4fbd9e1f.png)

From above variance plot, we can see that there are many numerical features which are having same variance and zero variance.We can drop these zero variance and same variance features as these features don't contribute to Model performance.

Till now, we have removed 69 features in total and left with 309 features.

Here we are using Ordinal Encoder to encode the categorical features.

Here we have created new features using dimensionality reduction technique like PCA and Interaction Features.

We need to create Interaction Features from top 3 important features. Important features can be find out by Random Forest Model.

![Feat_imp](https://user-images.githubusercontent.com/67824198/150626261-d1d3ef72-6607-4bc2-a739-5fee955cade4.png)

Modelling

We have in total 3 Datasets.

Original Dataset with Label Encoded

Original Dataset+PCA

Original Dataset+PCA+ Interaction Features

We are modelling these dataset by Random Forest, Xgboost, LGBM and Stacking Model.

Stacking Model Implementation

![stack_train](https://user-images.githubusercontent.com/67824198/150626297-ff78f2da-018e-4507-b2a2-7aa208709802.PNG)

![stack_test](https://user-images.githubusercontent.com/67824198/150626304-532d24e6-6186-4c5a-83ff-1a16c5e9bf3a.PNG)

Conclusion

![pretty_table](https://user-images.githubusercontent.com/67824198/150626337-436c9ca8-4f94-4724-a5a6-e9624b628142.PNG)

Best Dataset is Original Features(Label Encoded)+PCA+Feature Interaction. Best Model is Xgboost which is giving a R-square value of 0.625 and a kaggle score of 0.55343
