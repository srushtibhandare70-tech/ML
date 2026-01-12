from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import make_classification

x,y=make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=32)
c_space=np.logspace(-6,2,5)
param_grid = {'C':c_space}

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,param_grid,cv=5)
logreg_cv.fit(x,y)
print("Tuned logistic regreestion parameters:{}",format(logreg_cv.best_params_))
print("best scroce is {}",format(logreg_cv.best_score_))