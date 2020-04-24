# Decision-Tree-Hypertension-Prediction
A python decision tree classifier that predicts hypertension of real data set with an accuracy greater then 98%



## Model Overview 






### Data Preprocessing and Visualization 

We can utilize libraries and built in functions to load, clean, and better understand our data. I used a data set of 1159 patients from the area of Surkhet, a mid-western region of nepal, in a hyptertension research study by Khanal MK et all (2017). 

The features we selected were gender, family history of hypertension, their finacial status, body mass index, waist circumfrance, systolic blood pressure, diastolic blood pressure, and a label of whether they received treatment for HTN or not. 

I created a simple graph with the seaborn library to get a better idea of how our data points are spread out. In this example, I just plotted their SBP and BMI and their labels of HTN.



#### Train-Test Split
When using machine learning models for prediction or classification, it is extremely important that we split our data into training and testing groups so that our results are not skewed. I went the common practice of an 80% training set and 20% testing set. 



#### Helper Functions
To make creating our main decision tree algorithm simpler, we will create helper functions for that. 

**Data Purity**
Data purity can be simplified as splitting our data into groups and judging them on whether they are "pure" or not depending on the number of labels found in that sub-group. For example, if grouped all the patients with a SBP under 100, all the data points in this group will be considered "pure" since all of them will have a label of 0 for hypertension. The sub group will be "unpure" if there is more then 1 possible label in the set. 

**Classifying**
This is a simple function that basically classifies our data into a certain label or sub-group.

**Data Spliting**
This function is to help split our data into sub-groups. It takes in the parameters of data, the column we want to split, and the value. For example, we can just split our SBP column into groups of below and above 140.


### Entropy
Decision trees involes parititioning data into subsets that contain similar values (homogenous)
-If sample is completely homogenous, the entropy is 0
-If sample is equally divided, the entropy is 1
Entropy in decision trees are used to draw boundaries in the data
-If a branch has entropy of 0, it is a leaf node(we can classify, no need to split)



### Decision Tree Algorithm
In general, this decision tree will work as a recursive function to continually creating sub-tree's based on purity until
 all the values have been classified or 'labeled'. You can get a better picture by reviewing the notebook and looking through the code. 


**Dependencies** 
```
Python: 2.7
NumPy: 1.15.4
Pandas: 0.23.4
Seaborn: 0.9.0
```


**Credits**

Data Source: Khanal MK, Dhungana RR, Bhandari P, Gurung Y, Paudel KN (2017) Prevalence, associated factors, awareness, treatment, and control of hypertension: findings from a cross sectional study conducted as a part of a community based intervention trial in Surkhet, Mid-western region of Nepal. PLOS ONE 12(10): e0185806. https://doi.org/10.1371/journal.pone.0185806

Constructed this model by following Sebastian Mantey's youtube series regarding the logic and implementations of decision trees.
