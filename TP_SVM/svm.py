import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def genere_ex_1(n1=100, n2=50, mu1=[0,3], mu2=[3,0], sd1=0.15, sd2=0.2):
    X = np.concatenate((np.random.multivariate_normal(mu1, np.diagflat(sd1*np.ones(2)), n1), 
                        np.random.multivariate_normal(mu2, np.diagflat(sd2*np.ones(2)), n2)))
    Y = np.concatenate((np.ones((n1, 1)), -1*np.ones((n2,1))))[:,0]

    return X, Y

def genere_ex_2(n=300, mu=[0,0], std=0.25, delta=0.2):
    X = np.random.multivariate_normal(mu, np.diagflat(std*np.ones(2)), n)
    Y = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        x = X[i, 0]
        y = X[i, 1]
        if y < x * (x-1) * (x+1):
            Y[i] = -1
            X[i, 1] = X[i, 1] - delta
        else:
            Y[i] = 1
            X[i, 1] = X[i, 1] + delta
    
    return X, Y

def genere_black_hole(n=300, mu=[0,0], std=1, delta=0.2, radius=1.0):
    X = np.random.multivariate_normal(mu, np.diagflat(std*np.ones(2)), n)
    Y = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        x = X[i, 0]
        y = X[i, 1]
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        if r > radius:
            Y[i] = -1
            X[i, 0] = X[i, 0] + delta * np.cos(theta)
            X[i, 1] = X[i, 1] + delta * np.sin(theta)
        else:
            Y[i] = 1
            X[i, 0] = X[i, 0] - delta * np.cos(theta)
            X[i, 1] = X[i, 1] - delta * np.sin(theta)
    
    return X, Y

def plot_data_hyperplan(X, Y, classifier, figname, support_indices):
    plt.figure(figsize=(12,12*9/16))

    plot_dict = {1 : "b", -1 : "r"}
    c = [plot_dict[y] for y in Y]
    s = [8 ** 2] * np.size(Y)
    for i in support_indices:
        s[i] *= 4

        
    minx1 = min(X[:,0])
    maxx1 = max(X[:,0])
    minx2 = min(X[:,1])
    maxx2 = max(X[:,1])

    xx = np.linspace(minx1, maxx1, 100)
    yy = np.linspace(minx2, maxx2, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]
    
    P = classifier.predict_proba(Xfull)
    Z = classifier.decision_function(Xfull)

    # Plot
    plt.contourf(xx, yy, P[:,1].reshape((100,100)), 100, cmap="magma")
    plt.colorbar()
    plt.contour(xx, yy, Z.reshape((100, 100)), [-1, 0, 1], colors=["cyan", "g", "cyan"])
    plt.scatter(X[:,0], X[:,1], c=c, s=s, edgecolors="black")
    plt.xlim((minx1, maxx1))
    plt.ylim((minx2, maxx2))
    plt.grid()
    plt.savefig(figname)
    plt.close()
    return

def print_confusion_matrix(tn, tp, fn, fp):
    print("\n\n**********Confusion Matrix*********** \n")
    print("\t\tReality")
    print("\t\tNeg \tPos \tTotal")
    print(f"Pred \tTest- \t{tn}\t{fn}\t{tn + fn}")
    print(f"\tTest+ \t{fp}\t{tp}\t{fp + tp}")
    print(f"\tTotal\t{fp + tn}\t{fn + tp}\t{fp + tp + fn + tn}")

def plot_roc(x_test, y_test, classifier):
    fpr = []
    tpr = []

    thresholds = np.linspace(0, 1, 1000)
    y_predict_proba = classifier.predict_proba(x_test)[:,1]
    for threshold in thresholds:
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for i in range(len(y_predict_proba)):
            if y_test[i] == 1:
                if y_predict_proba[i] > threshold:
                    # True positive
                    true_positive += 1
                else:
                    # False negative
                    false_negative += 1
            else:
                if y_predict_proba[i] > threshold:
                    # False positive
                    false_positive += 1
                else:
                    # true negative
                    true_negative += 1
        sensitivity = true_positive / (true_positive + false_negative) 
        specificity = true_negative / (true_negative + false_positive) 
        fpr.append(1 - specificity)
        tpr.append(sensitivity)
    plt.plot(fpr, tpr)
    plt.show()

def main(X, Y):
    classifier = SVC(kernel='linear', probability=True)
    classifier = classifier.fit(X, Y)
    plot_data_hyperplan(X, Y, classifier, "Graph_SVM_linear", classifier.support_)

    X, Y = genere_ex_2()
    X_test, Y_test = genere_ex_2(n=500, delta=0.01)
    classifier = GridSearchCV(SVC(probability=True), {"kernel":["poly"], "degree":[2, 3, 4, 5], "C":[0.1, 1, 5, 10, 50, 100]})
    classifier = classifier.fit(X, Y)
    print(classifier.best_params_, classifier.best_score_)
    C = classifier.best_params_["C"]
    kernel = classifier.best_params_["kernel"]
    degree = classifier.best_params_["degree"]
    classifier = SVC(kernel = kernel, degree=degree, C=C, coef0=1, probability=True)
    classifier = classifier.fit(X, Y)
    Y_predict = classifier.predict(X_test)
    
    # WARNING: THRESHOLD = 0.5 !!!!!!!
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for i in range(len(Y_predict)):
        if Y_test[i] == 1:
            if Y_predict[i] == 1:
                # True positive
                true_positive += 1
            else:
                # False negative
                false_negative += 1
        else:
            if Y_predict[i] == 1:
                # False positive
                false_positive += 1
            else:
                # true negative
                true_negative += 1

    sensitivity = true_positive / (true_positive + false_negative) 
    specificity = true_negative / (true_negative + false_positive) 
    vpp = true_positive / (true_positive + false_positive) 
    vpn = true_negative / (true_negative + false_negative) 

    print_confusion_matrix(true_negative, true_positive, false_negative, false_positive)
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"VPP: {vpp}")
    print(f"VPN: {vpn}")

    # Plot ROC
    from sklearn.metrics import roc_curve
    fpr, tpr, threshold = roc_curve(Y_test, classifier.predict_proba(X_test)[:,1])
    plt.plot(fpr, tpr)
    plt.show()
    plot_roc(X_test, Y_test, classifier)
    plot_data_hyperplan(X, Y, classifier, "Graph_SVM_poly", classifier.support_)
    
    # Black hole
    X, Y = genere_black_hole(n=1000)
    classifier = SVC(probability=True, kernel="rbf")
    classifier = classifier.fit(X, Y)
    plot_data_hyperplan(X, Y, classifier, "Graph_SVM_black_hole", classifier.support_)

X, Y = genere_ex_1()
main(X, Y)