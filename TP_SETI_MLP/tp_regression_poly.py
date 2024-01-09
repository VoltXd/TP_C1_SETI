from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def genere_exemple_dim1(xmin, xmax, nb_ex, sigma):
    x = np.arange(xmin, xmax, (xmax - xmin) / nb_ex)
    y = np.sin(-np.pi + 2*x*np.pi) + np.random.normal(loc=0.0, scale=sigma, size=x.size)
    return x.reshape(-1, 1), y

def get_MSE(x, y, reg):
    return sum(pow(reg.predict(x) - y, 2)) / x.shape[0]

def plot_model(x_a, y_a, x_t, y_t, reg, namefig="model"):
    y_pred = reg.predict(x_t)
    plt.plot(x_a[:,1], y_a, '*r')
    plt.plot(x_t[:,1], y_t, '-b')
    plt.plot(x_t[:,1], y_pred, '-r')
    plt.grid()
    # plt.show()
    plt.savefig(namefig+'.jpg', dpi=200)
    plt.close()

def plot_error_profile(L_error_app, L_error_test, namefig="error_profile"):
    plt.plot(range(1, len(L_error_app)+1), L_error_app, '-r')
    plt.plot(range(1, len(L_error_test)+1), L_error_test, '-b')
    plt.grid()
    plt.yscale("log")
    # plt.show()
    plt.savefig(namefig+'.jpg', dpi=200)
    plt.close()

def plot_confusion(x_t, y_t, reg, namefig="confusion"):
    plt.plot(y_t, reg.predict(x_t), '.b')
    plt.plot(y_t, y_t, '-r')
    plt.grid()
    # plt.show()
    plt.savefig(namefig+'.jpg', dpi=200)
    plt.close()

def main(degre_max=12, nb_ex=20, sigma=0.2):
    x_min = 0
    x_max = 1.2
    
    x_app, y_app = genere_exemple_dim1(x_min, x_max, nb_ex, sigma)
    x_test, y_test = genere_exemple_dim1(x_min, x_max * 2, 200, 0)

    L_error_app = []
    L_error_test = []
    
    for i in range(1, degre_max+1):
        print("Degre = ", i)

        # Transformation des données d'entrée des bases d'app et de test
        poly = PolynomialFeatures(i)
        Xa = poly.fit_transform(x_app, y_app)
        Xt = poly.fit_transform(x_test, y_test)

        # Création du modèle linéaire
        reg = LinearRegression().fit(Xa, y_app)

        # Estimation des erruers d'apprentissage et de test
        L_error_app.append(get_MSE(Xa, y_app, reg))
        L_error_test.append(get_MSE(Xt, y_test, reg))

        # Plot du modèle de degré i
        plot_model(Xa, y_app, Xt, y_test, reg, "Model_%02d" % i)
        plot_confusion(Xt, y_test, reg, "Confusion_%02d" % i)
    
    # Déterminer le degré optimal
    best = np.argmin(L_error_test) + 1
    print("Meilleur modele -> degre =", best)
    plot_error_profile(L_error_app, L_error_test, f"Profil_Err_App_Test_{nb_ex}")

    # Création du modèle final optimal
    poly = PolynomialFeatures(best)
    Xa = poly.fit_transform(x_app, y_app)
    Xt = poly.fit_transform(x_test, y_test)

    # Création du modèle linéaire
    reg = LinearRegression().fit(Xa, y_app)

    plot_confusion(Xt, y_test, reg, "Confusion_best")

if __name__ == "__main__":
    nb_exs = [10, 50, 200]
    for nb_ex in nb_exs:
        main(nb_ex=nb_ex)