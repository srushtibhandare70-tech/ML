from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target
svm = SVC(kernel="linear", C=1)
svm.fit(X, y)
DecisionBoundaryDisplay.from_estimator(
 svm,
 X,
 response_method="predict",
 alpha=0.9,
 cmap="Pastel1",
 xlabel=cancer.feature_names[0],
 ylabel=cancer.feature_names[1],
 )
plt.title('SVM Decision Boundary With Line Kernel',fontweight="bold")
plt.scatter(X[:, 0], X[:, 1],
 c=y,
 s=20, edgecolors="blue")
plt.show()