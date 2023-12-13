import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.neural_network import MLPRegressor
import numpy as np

def tan_hyp(x):
    A = np.array([[1, 1], [-2, 1]])
    b = np.array([0.2, -0.3])

    z = A.dot(x) + b
    h = np.transpose(z).dot(np.ones(z.shape))
    y = (np.tanh(h) + 1) / 2 
    return y

def plot_surf(figname, regr=None):
    step_v = 0.005

    x1v = np.arange(0,1,step_v)
    x2v = np.arange(0,1,step_v)
    Xv, Yv = np.meshgrid(x1v, x2v)

    R = np.zeros(Xv.shape)
    for i, x1 in enumerate(x1v):
        for j, x2 in enumerate(x2v):
            if not regr:
                R[j, i] = tan_hyp(np.array([x1, x2]))
            else:
                R[j, i] = regr.predict(np.array([[x1, x2]]))[0]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Plot the surface
    surf = ax.plot_surface(Xv, Yv, R, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

def plot_surf_and_scatter(figname, X, Y, regr=None):
    step_v = 0.005

    x1v = np.arange(0,1,step_v)
    x2v = np.arange(0,1,step_v)
    Xv, Yv = np.meshgrid(x1v, x2v)

    R = np.zeros(Xv.shape)
    for i, x1 in enumerate(x1v):
        for j, x2 in enumerate(x2v):
            if not regr:
                R[j, i] = tan_hyp(np.array([x1, x2]))
            else:
                R[j, i] = regr.predict(np.array([[x1, x2]]))[0]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # Plot the surface
    surf = ax.plot_surface(Xv, Yv, R, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.scatter(X[:,0], X[:,1], Y)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

def plot_confusion(x_t, y_t, reg, namefig="confusion"):
    plt.plot(y_t, reg.predict(x_t), '.b')
    plt.plot(y_t, y_t, '-r')
    plt.grid()
    # plt.show()
    plt.savefig(namefig+'.jpg', dpi=200)
    plt.close()


def main():

    # Get learning set
    with open("sinc_dim2_input.csv", 'r') as f:
        f.readline() # Skip the header
        X_app = np.loadtxt(f, delimiter=';')

    y_app = np.zeros(X_app[:,0].shape)
    for i, x in enumerate(X_app):
        y_app[i] = tan_hyp(x)

    # Generate test set
    mesh_line = np.linspace(0, 1, 40)
    test_set_size = 40*40
    X_test = np.zeros((test_set_size, 2))
    y_test = np.zeros((test_set_size, 1))
    i = 0
    j = 0
    for idx in range(test_set_size):
        x = np.array([mesh_line[i], mesh_line[j]])
        X_test[idx] = x
        y_test[idx] = tan_hyp(x)

        j += 1
        if j == 40:
            j = 0
            i += 1

    # Training
    regr = MLPRegressor(hidden_layer_sizes=100).fit(X_app, y_app)
    y_predict = regr.predict(X_test)
    
    # Plots
    plot_surf_and_scatter("fjlksdmjfq", X_app, y_app, regr)
    plot_surf_and_scatter("fjlksdmjfq", X_test, y_test, regr)
    plot_confusion(X_test, y_test, regr, "CONFUSION_MLP")
main()