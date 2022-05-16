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

# todo fitting a line to the data. Visualization of data for degree polynomial = 1 and, lamb = 0
degree, lamb = 1, 0
thetas = LinearRegression(X, y, lamb).optimal_thetas
Visualization(X, y, thetas, degree, lamb, vistype="FIT")

# todo Examining the hypothesis, i.e. the dependence of the straight line fit.
#  Number of features and lambda value are taken into consideration
learn_curve = LearningCurve(X, y, X_val, y_val, lamb)
error_train, error_val = learn_curve.error_train, learn_curve.error_val
Visualization(error_train, error_val, degree=degree, lamb=lamb, vistype="learning curve")

# todo Fitting a curve. Let's try 8 degree polynomial
degree, lamb = 8, 0
poly_dataset = PolyFeatures(degree, X, X_val, X_test)
X_poly, X_poly_val, X_poly_test = poly_dataset.X, poly_dataset.X_val, poly_dataset.X_test
thetas = LinearRegression(X_poly, y, lamb).optimal_thetas

# todo Visualization - 8 degree polynomial works really precisely on a training set but does not generalize
#  on new examples (overfitting)
Visualization(X, y, theta=thetas, degree=degree, lamb=lamb, vistype="FIT")
learn_curve = LearningCurve(X_poly, y, X_poly_val, y_val, lamb)
error_train, error_val = learn_curve.error_train, learn_curve.error_val
Visualization(error_train, error_val, degree=degree, lamb=lamb, vistype="learning curve")

# todo Plot of error/lambda for Training and Cross Validation set. The generalization is the best when
#  lambda value equals 0.1. Let's take this value for the model
validation_curve = ValidationCurve(X_poly, y, X_poly_val, y_val)
error_train, error_val = validation_curve.error_train, validation_curve.error_val
Visualization(error_train, error_val, degree=degree, lamb=lamb, vistype="SET LAMBDA")

# todo Test set error for degree polynomial = 8, lambda = 0.1
degree = 8
lamb = 0.1
theta_er = LinearRegression(X_poly, y, lamb).optimal_thetas
test_er = LinearRegression.cost(theta_er, X_poly_test, y_test, 0)


# todo The fit of a final model performed on test set
Visualization(X_test, y_test, theta=theta_er, degree=degree, lamb=lamb, vistype="FIT")