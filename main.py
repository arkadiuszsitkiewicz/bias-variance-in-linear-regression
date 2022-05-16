from src.data.prepare_data import PrepareData
from src.vis.visualize import Visualization
from src.calc.linear_reg import LinearRegression
from src.calc.learning_curve import LearningCurve
from src.calc.poly_features import PolyFeatures
from src.calc.validation_curve import ValidationCurve


# todo Load input data
data = PrepareData("data/raw_data.csv")
X, y, = data.X, data.y
X_val, y_val = data.X_val, data.y_val
X_test, y_test = data.X_test, data.y_test

# todo fitting a line to the data. Visualization of data for degree = 1 and, lamb = 0
degree, lamb = 1, 0
thetas = LinearRegression(X, y, lamb).optimal_thetas
Visualization(X, y, thetas, degree, lamb, vistype="FIT")

# Badamy zaleznosc naszej hipotezy tj zaleznosc dopsaowania prostej dla liczby features oraz lambdy

learn_curve = LearningCurve(X, y, X_val, y_val, lamb)
error_train, error_val = learn_curve.error_train, learn_curve.error_val
Visualization(error_train, error_val, degree=degree, lamb=lamb, vistype="learning curve")

# Próbujemy dopasować krzywą, w tym celu staramy się dopasować wielomian 8 stopnia

degree, lamb = 8, 0
poly_dataset = PolyFeatures(degree, X, X_val, X_test)
X_poly, X_poly_val, X_poly_test = poly_dataset.X, poly_dataset.X_val, poly_dataset.X_test
thetas = LinearRegression(X_poly, y, lamb).optimal_thetas

# Wizualizacja - widzimy, że nasz dobór parametrów ma bardzo dobrą dokładność na trainig set natomiast nie generaluzuje
# dobrze poza nieznanymi wartościamy (overfit)
Visualization(X, y, theta=thetas, degree=degree, lamb=lamb, vistype="FIT")
learn_curve = LearningCurve(X_poly, y, X_poly_val, y_val, lamb)
error_train, error_val = learn_curve.error_train, learn_curve.error_val
Visualization(error_train, error_val, degree=degree, lamb=lamb, vistype="learning curve")

# Wykres zależności doboru lambdy od erroru dla train oraz CV set, widać, że generalizacja jest najlepsza dla
# wartości lambda 0.1. Zatem taką wartość ustalamy w naszym modelu
validation_curve = ValidationCurve(X_poly, y, X_poly_val, y_val)
error_train, error_val = validation_curve.error_train, validation_curve.error_val
Visualization(error_train, error_val, degree=degree, lamb=lamb, vistype="SET LAMBDA")

# Liczymy test_er dla degree = 8, lambda = 0.1
degree = 8
lamb = 0.1
theta_er = LinearRegression(X_poly, y, lamb).optimal_thetas
test_er = LinearRegression.cost(theta_er, X_poly_test, y_test, 0)
print(test_er)

# Na koniec sprawdzamy jeszcze dopasowanie do naszego test setu
Visualization(X_test, y_test, theta=theta_er, degree=degree, lamb=lamb, vistype="FIT")