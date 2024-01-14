import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.neural_network import MLPRegressor
import numpy as np

def plot_error_profile(L_error_app, L_error_test, namefig="error_profile"):
    plt.plot(range(2, len(L_error_app)+2), L_error_app, '-r')
    plt.plot(range(2, len(L_error_test)+2), L_error_test, '-b')
    plt.grid()
    plt.yscale("log")
    # plt.show()
    plt.savefig(namefig+'.jpg', dpi=200)
    plt.close()

def get_MSE(x, y, reg):
    return np.sum(pow(reg.predict(x) - y, 2)) / x.shape[0]

def tan_hyp(x):
    A = np.array([[5, 0], [0, 7]])
    b = np.array([-0.5, -0.5])

    z = A.dot(x + b) 
    h = np.transpose(z).dot(np.ones(z.shape))
    y = (np.tanh(h) + 1) / 2 
    return y 

def sinc(x):
    A = np.array([[5, 0], [0, 7]])
    b = np.array([-0.5, -0.5])

    z = A.dot(x + b) 
    h =  np.sqrt(np.transpose(z).dot(z))
    y = np.sin(h) / h 
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
                R[i, j] = tan_hyp(np.array([x1, x2]))
            else:
                R[i, j] = regr.predict(np.array([[x1, x2]]))[0]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Plot the surface
    surf = ax.plot_surface(Xv, Yv, R, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(figname, dpi=300)
    # plt.show()
    plt.close()

def plot_surf_and_scatter(figname, X, Y, regr=None, func=tan_hyp):
    step_v = 0.005

    x1min = min(X[:,0])
    x1max = max(X[:,0])
    x2min = min(X[:,1])
    x2max = max(X[:,1])
    x1v = np.arange(x1min,x1max,step_v)
    x2v = np.arange(x2min,x2max,step_v)
    Xv, Yv = np.meshgrid(x1v, x2v)

    R = np.zeros(Xv.shape)
    for i, x1 in enumerate(x1v):
        for j, x2 in enumerate(x2v):
            if not regr:
                R[j, i] = func(np.array([x1, x2]))
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
    # plt.show()
    plt.close()

def plot_confusion(x_t, y_t, reg, namefig="confusion"):
    plt.plot(y_t, reg.predict(x_t), '.b')
    plt.plot(y_t, y_t, '-r')
    plt.grid()
    # plt.show()
    plt.savefig(namefig+'.jpg', dpi=200)
    plt.close()


def main():

    FUNC = tan_hyp

    # Get learning set
    with open("sinc_dim2_input.csv", 'r') as f:
        f.readline() # Skip the header
        X_app = np.loadtxt(f, delimiter=';')

    y_app = np.zeros(X_app.shape[0])
    for i, x in enumerate(X_app):
        y_app[i] = FUNC(x)

    # Generate test set
    POINTS_PER_DIM = 40
    mesh_line = np.linspace(0, 1, POINTS_PER_DIM)
    test_set_size = POINTS_PER_DIM ** 2
    X_test = np.zeros((test_set_size, 2))
    y_test = np.zeros(test_set_size)
    i = 0
    j = 0
    for idx in range(test_set_size):
        x = np.array([mesh_line[i], mesh_line[j]])
        X_test[idx] = x
        y_test[idx] = FUNC(x)

        j += 1
        if j == 40:
            j = 0
            i += 1

    L_error_app = []
    L_error_test = []
    
    # Training
    for hl_size in range(2, 10 + 1):
        # L_error_app.append(get_MSE(X_app, y_app, regr))
        # L_error_test.append(get_MSE(X_test, y_test, regr))
        L_error_app.append(0)
        L_error_test.append(0)
        print(f"{hl_size} neuronnes")

        NUMBER_OF_TESTS = 25
        for i in range(NUMBER_OF_TESTS):    
            regr = MLPRegressor(hidden_layer_sizes=(hl_size,), max_iter=2000, activation="tanh", solver="lbfgs", tol=0.0001)
            regr = regr.fit(X_app, y_app)
            L_error_app[-1] += get_MSE(X_app, y_app, regr)
            L_error_test[-1] += get_MSE(X_test, y_test, regr)
        L_error_app[-1] /= NUMBER_OF_TESTS
        L_error_test[-1] /= NUMBER_OF_TESTS



    best = np.argmin(L_error_test) + 2
    print("Meilleur modele -> Neurones =", best)
    
    regr = MLPRegressor(hidden_layer_sizes=(best,), max_iter=2000, activation="tanh", learning_rate="adaptive").fit(X_app, y_app)

    # Plots
    plot_surf_and_scatter("MLP learning set", X_app, y_app, regr, FUNC)
    plot_surf_and_scatter("MLP Test set", X_test, y_test, regr, FUNC)
    plot_confusion(X_test, y_test, regr, "CONFUSION_MLP")
    
    plot_error_profile(L_error_app, L_error_test, "Profil_Err_App_Test_MLP")

main()