PROJECT : TITANIC SURVIVAL PREDICTION USING MACHINE LEARNING

This project analyzes the Titanic data set and makes two predictions. 
One prediction to see which passengers on board the ship would survive and then another prediction to see if we would’ve survived.

Data Set Column Descriptions
pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survived: Survival (0 = No; 1 = Yes)
name: Name
sex: Sex
age: Age
sibsp: Number of siblings/spouses aboard
parch: Number of parents/children aboard
fare: Passenger fare (British pound)
embarked: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
adult_male: A male 18 or older (0 = No, 1=Yes)
deck: Deck of the ship
who: man (18+), woman (18+), child (<18)
alive: Yes, no
embarked_town: Port of embarkation ( Cherbourg, Queenstown, Southampton)
class: Passenger class (1st; 2nd; 3rd)
alone: 1= alone, 0= not alone ( you have at least 1 sibling, spouse, parent or child on board)

age
Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp
The dataset defines family relations in this way:
Sibling= brother, sister, stepbrother, stepsister
Spouse= husband, wife (mistresses and fiancés were ignored)

parch
The dataset defines family relations in this way:
Parent= mother, father
Child= daughter, son, stepdaughter, stepson
Some children traveled only with a nanny, therefore parch=0 for them.

Here a function is created that has within it many different machine learning models that is used to make predictions.
The model that was most accurate on the training data was the Decision Tree Classifier.

The following is the output of the project  : 
0]Logistic Regression Training Accuracy: 0.7978910369068541
[1]K Nearest Neighbor Training Accuracy: 0.8664323374340949
[2]Support Vector Machine (Linear Classifier) Training Accuracy: 0.7768014059753954
[3]Support Vector Machine (RBF Classifier) Training Accuracy: 0.8506151142355008
[4]Gaussian Naive Bayes Training Accuracy: 0.8031634446397188
[5]Decision Tree Classifier Training Accuracy: 0.9929701230228472
[6]Random Forest Classifier Training Accuracy: 0.9753954305799648
[[73  9]
 [18 43]]
Model[0] Testing Accuracy = "0.8111888111888111 !"

[[71 11]
 [20 41]]
Model[1] Testing Accuracy = "0.7832167832167832 !"

[[70 12]
 [18 43]]
Model[2] Testing Accuracy = "0.7902097902097902 !"

[[75  7]
 [22 39]]
Model[3] Testing Accuracy = "0.7972027972027972 !"

[[69 13]
 [23 38]]
Model[4] Testing Accuracy = "0.7482517482517482 !"

[[60 22]
 [10 51]]
Model[5] Testing Accuracy = "0.7762237762237763 !"

[[67 15]
 [13 48]]
Model[6] Testing Accuracy = "0.8041958041958042 !"

[1 1 1 0 0 0 1 0 0 1 1 1 1 0 0 1 0 1 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1
 1 1 0 0 0 1 0 1 0 1 0 0 1 1 0 1 0 1 0 0 1 1 1 0 0 0 1 0 0 1 0 1 1 1 1 1 1
 0 0 1 0 0 0 0 1 0 1 1 0 0 0 1 0 0 0 1 1 1 0 1 1 0 0 0 1 0 0 0 0 1 0 0 0 1
 0 1 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 1 1 1 1 1 0 0 0 0 0 1]

[0 0 1 0 0 0 1 0 0 0 1 1 1 0 0 1 0 1 1 0 0 1 1 1 0 0 0 0 1 0 0 0 0 1 1 0 1
 1 1 1 1 1 0 0 0 0 1 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 0 1 0 0 1 1 0 1 0 1 1 1
 0 0 1 1 0 0 0 1 1 1 1 0 0 0 1 0 0 0 1 1 1 0 1 1 0 1 0 1 0 0 0 0 1 0 0 0 0
 1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 1]
[0]
Oh no! You didn not make it

