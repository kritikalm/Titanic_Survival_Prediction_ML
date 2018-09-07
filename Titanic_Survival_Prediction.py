
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('precision', 2)

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns


# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# STEP-1) Read in and Explore the Data

# import train and test CSV files
train = pd.read_csv('train.csv')   # 12 columns
test = pd.read_csv('test.csv')     # 11 columns

# take a look at the training data

print train.describe()

print train.describe(include="all")


# STEP-3) Data Analysis
print train.columns

print
print train.head(5)
print train.sample(5)

#

print "Data types for each feature : -"
print train.dtypes

train.describe(include = "all")




# check for any other unusable values
print
print pd.isnull(train).sum()


# 4) Data Visualization
# 4.A) Sex Feature
# -----------------
# draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=train)
plt.show()
# print percentages of females vs. males that survive
# print "Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100


print "------------------\n\n"
print train["Survived"][train["Sex"] == 'female']


print "*****************\n\n"
print train["Survived"][train["Sex"] == 'female'].value_counts()
print "====================================\n\n"
print train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)
#print train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]
print "Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100
print "Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100


#--------------------
#4.B) Pclass Feature
#--------------------

sns.barplot(x="Pclass", y="Survived", data=train)
plt.show()
#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)


#4.C) SibSp Feature
#----------------------
#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=train)

#I won't be printing individual percent values for all of these.
print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)

plt.show()



#--------------------
#4.D)Parch Feature
#--------------------

#draw a bar plot for Parch vs. survival
sns.barplot(x="Parch", y="Survived", data=train)
plt.show()


#-----------------
#4.E)Age Feature
#-----------------


#sort the ages into logical categories
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins =   [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)
print train
#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()

#--------------------
#4.F) Cabin Feature
#--------------------

train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

print "###################################\n\n"
print train


#calculate percentages of CabinBool vs. survived
print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
#draw a bar plot of CabinBool vs. survival
sns.barplot(x="CabinBool", y="Survived", data=train)
plt.show()

#5) Cleaning Data
#*********************************


print test.describe(include="all")

#Cabin Feature
#we'll start off by dropping the Cabin feature since not a lot more useful information can be extracted from it.
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)

#Ticket Feature
#we can also drop the Ticket feature since it's unlikely to yield any useful information
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

#Embarked Feature
#now we need to fill in the missing values in the Embarked feature
print "Number of people embarking in Southampton (S):" ,

southampton = train[train["Embarked"] == "S"].shape[0]
print southampton


print "Number of people embarking in Cherbourg (C):" ,
cherbourg = train[train["Embarked"] == "C"].shape[0]
print cherbourg

print "Number of people embarking in Queenstown (Q):" ,
queenstown = train[train["Embarked"] == "Q"].shape[0]
print queenstown


#replacing the missing values in the Embarked feature with S
train = train.fillna({"Embarked": "S"})




#create a combined group of both datasets
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


print "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
print train
print



print pd.crosstab(train['Title'], train['Sex'])


# replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print "\n\nAfter grouping rare title : \n" , train




print train[['Title', 'Survived']].groupby(['Title'], as_index=False).count()




print "\nMap each of the title groups to a numerical value."
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)





print "\n\nAfter replacing title with numeric values.\n"
print train



#Next, we'll try to predict the missing Age values from the most common age for their Title.

# fill missing age with mode age group for each title
mr_age = train[train["Title"] == 1]["AgeGroup"].mode()  # Young Adult
print "mode() of mr_age : ", mr_age

miss_age = train[train["Title"] == 2]["AgeGroup"].mode()  # Student
print "mode() of miss_age : ", miss_age

mrs_age = train[train["Title"] == 3]["AgeGroup"].mode()  # Adult
print "mode() of mrs_age : ", mrs_age

master_age = train[train["Title"] == 4]["AgeGroup"].mode()  # Baby
print "mode() of master_age : ", master_age

royal_age = train[train["Title"] == 5]["AgeGroup"].mode()  # Adult
print "mode() of royal_age : ", royal_age

rare_age = train[train["Title"] == 6]["AgeGroup"].mode()  # Adult
print "mode() of rare_age : ", rare_age

print "\n\n**************************************************\n\n"
print train.describe(include="all")
print train




print "\n\n********   train[AgeGroup][x] :  \n\n"
for x in range(10) :
    print train["AgeGroup"][x]






age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}


for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":   # x=5 ( means for 6th record )
        train["AgeGroup"][x] = age_title_mapping[  train["Title"][x]  ]

for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]



print"\n\nAfter replacing Unknown values from AgeGroup column : \n"
print train

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
print
print train





# dropping the Age feature for now, might change
train = train.drop(['Age'], axis=1)
test = test.drop(['Age'], axis=1)

print "\n\nAge column droped."
print train




#Name Feature

train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


#Sex Feature
#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

print train

#Embarked Feature
#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)
print
print train.head()



#fill in missing Fare value in test set based on mean fare for that Pclass
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

#map Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])




#drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)
#check train data
print "\n\nFare column droped\n"
print train


#check test data
print
print test.head()

#****************************************
#6) Choosing the Best Model
#****************************************

#Splitting the Training Data
#We will use part of our training data (20% in this case) to test the accuracy of our different models.

from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)



#For each model, we set the model, fit it with 80% of our training data,
# predict for 20% of the training data and check the accuracy.

from sklearn.metrics import accuracy_score

#MODEL-1) LogisticRegression
#------------------------------------------
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-1: Accuracy of LogisticRegression : ", acc_logreg



#MODEL-2) Gaussian Naive Bayes
#------------------------------------------
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-2: Accuracy of GaussianNB : ", acc_gaussian

#MODEL-3) Support Vector Machines
#------------------------------------------
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-3: Accuracy of Support Vector Machines : ", acc_svc

#MODEL-4) Linear SVC
#------------------------------------------
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-4: Accuracy of LinearSVC : ",acc_linear_svc

#MODEL-5) Perceptron
#------------------------------------------
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-5: Accuracy of Perceptron : ",acc_perceptron


#MODEL-6) Decision Tree Classifier
#------------------------------------------
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-6: Accuracy of DecisionTreeClassifier : ", acc_decisiontree


#MODEL-7) Random Forest
#------------------------------------------
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-7: Accuracy of RandomForestClassifier : ",acc_randomforest


#MODEL-8) KNN or k-Nearest Neighbors
#------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-8: Accuracy of k-Nearest Neighbors : ",acc_knn


#MODEL-9) Stochastic Gradient Descent
#------------------------------------------
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-9: Accuracy of Stochastic Gradient Descent : ",acc_sgd


#MODEL-10) Gradient Boosting Classifier
#------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print "MODEL-10: Accuracy of GradientBoostingClassifier : ",acc_gbk


models = pd.DataFrame({
    'Model': ['Logistic Regression','Gaussian Naive Bayes','Support Vector Machines',
              'Linear SVC', 'Perceptron',  'Decision Tree',
              'Random Forest', 'KNN','Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_logreg, acc_gaussian, acc_svc,
              acc_linear_svc, acc_perceptron,  acc_decisiontree,
              acc_randomforest,  acc_knn,  acc_sgd, acc_gbk]
                    })


print
print models.sort_values(by='Score', ascending=False)


#7) Creating Submission Result File
#***********************************

#It is time to create a submission.csv file which includes our predictions for test data

#set ids as PassengerId and predict survival
ids = test['PassengerId']
predictions = randomforest.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)

print "All survival predictions done."
print "All predictions exported to submission.csv file."

print output

