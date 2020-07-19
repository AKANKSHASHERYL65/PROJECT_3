import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
titanic = sns.load_dataset('titanic')
titanic.head(10)
titanic.shape
titanic.describe()
titanic['survived'].value_counts()
sns.countplot(titanic['survived'],label="Count")

cols = ['who', 'sex', 'pclass', 'sibsp', 'parch', 'embarked']

n_rows = 2
n_cols = 3

fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.2))

for r in range(0, n_rows):
    for c in range(0, n_cols):
        i = r * n_cols + c
        ax = axs[r][c]
        sns.countplot(titanic[cols[i]], hue=titanic["survived"], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title="survived", loc='upper right')

plt.tight_layout()

titanic.groupby('sex')[['survived']].mean()
titanic.pivot_table('survived', index='sex', columns='class')
titanic.pivot_table('survived', index='sex', columns='class').plot()
sns.barplot(x='class', y='survived', data=titanic)
age = pd.cut(titanic['age'], [0, 18, 80])
titanic.pivot_table('survived', ['sex', age], 'class')
plt.scatter(titanic['fare'], titanic['class'], color='purple', label='Passenger Paid')
plt.ylabel('Class')
plt.xlabel('Price / Fare')
plt.title('Price Of Each Class')
plt.legend()
plt.show()
titanic.isna().sum()
for val in titanic:
   print(titanic[val].value_counts())
   print()


titanic = titanic.drop(['deck', 'embark_town', 'alive', 'class', 'alone', 'adult_male', 'who'], axis=1)
titanic = titanic.dropna(subset =['embarked', 'age'])
titanic.shape
titanic.dtypes
print(titanic['sex'].unique())
print(titanic['embarked'].unique())


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

titanic.iloc[:,2]= labelencoder.fit_transform(titanic.iloc[:,2].values)

titanic.iloc[:,7]= labelencoder.fit_transform(titanic.iloc[:,7].values)

print(titanic['sex'].unique())
print(titanic['embarked'].unique())

X = titanic.iloc[:, 1:8].values
Y = titanic.iloc[:, 0].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



def models(X_train, Y_train):
    # Using Logistic Regression Algorithm to the Training Set
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)


    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(X_train, Y_train)


    from sklearn.svm import SVC
    svc_lin = SVC(kernel='linear', random_state=0)
    svc_lin.fit(X_train, Y_train)


    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='rbf', random_state=0)
    svc_rbf.fit(X_train, Y_train)


    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)


    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)


    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(X_train, Y_train)


    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
    print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
    print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
    print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))

    return log, knn, svc_lin, svc_rbf, gauss, tree, forest

model = models(X_train,Y_train)

from sklearn.metrics import confusion_matrix
for i in range(len(model)):
   cm = confusion_matrix(Y_test, model[i].predict(X_test))

   TN, FP, FN, TP = confusion_matrix(Y_test, model[i].predict(X_test)).ravel()
   print(cm)
   print('Model[{}] Testing Accuracy = "{} !"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))
   print()


forest = model[6]
importances = pd.DataFrame({'feature':titanic.iloc[:, 1:8].columns,'importance':np.round(forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances

importances.plot.bar()


pred = model[6].predict(X_test)
print(pred)


print()


print(Y_test)

my_survival = [[3,1,21,0, 0, 0, 1]]

pred = model[6].predict(my_survival)
print(pred)

if pred == 0:
  print('Oh no! You didn not make it')
else:
  print('Nice! You survived')

