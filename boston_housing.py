"""Load the Boston dataset and examine its target (label) distribution."""

# Load libraries
import numpy as np
import matplotlib.pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import cross_validation as cva
from sklearn.metrics import fbeta_score, make_scorer, mean_squared_error
from sklearn.metrics import mean_absolute_error,explained_variance_score,median_absolute_error, r2_score
from sklearn import grid_search
#from scipy.stats.mstats import mode
import scipy.stats as stats
#from IPython.display import Image
################################
### ADD EXTRA LIBRARIES HERE ###
################################

from sklearn.neighbors import NearestNeighbors

def find_nearest_neighbor_indexes(x, X):  # x is your vector and X is the data set.
   neigh = NearestNeighbors( n_neighbors = 10 )
   neigh.fit(X)
   distance, indexes = neigh.kneighbors( x )
   return indexes
   
def load_data():
    """Load the Boston dataset."""

    boston = datasets.load_boston()
        
    
    return boston

def threquency(housing_prices):

    pl.hist(housing_prices, 50, facecolor='green', alpha=0.75)
    pl.xlabel('House price')
    pl.ylabel('Frequency')
    pl.title('Frequency of housing prices')
#    pl.hist
    pl.show()

def cut_outliers(city_data,outlier=0.25):
    housing_prices = city_data.target
    
    housing_features = city_data.data
    rows, cols = housing_features.shape
    housing_prices=housing_prices.reshape(rows, 1)
    #order array by first columnt (price)
    housing_prices_ordered = np.concatenate ([housing_prices, housing_features],1)
    housing_prices_ordered=housing_prices_ordered[housing_prices_ordered[:,0].argsort()]   
    
    #calucate cut off ranges
    cut_rows = int(rows*outlier)
    total_range = rows - cut_rows*2
    housing_prices_outlied = np.zeros((total_range), dtype=float)
    housing_features_outlied = np.zeros((total_range,cols), dtype=float)
    
    #fill outlied arrays
    for j,i in enumerate(range (cut_rows, rows - cut_rows)):
        housing_prices_outlied[j] = housing_prices_ordered[i,0]
        housing_features_outlied[j,:] = housing_prices_ordered[i,1:]
    
    
    out_city_data = load_data()
    
    
    out_city_data.data=housing_features_outlied
    out_city_data.target=housing_prices_outlied
    
    return out_city_data
    

def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""
    print "Exploring city data:"
    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    ###################################
    ### Step 1. YOUR CODE GOES HERE ###
    ###################################

    # Please calculate the following values using the Numpy library
    
    data_size, features_dim=housing_features.shape
    # Size of data?    
    print "Data size:",data_size
   
    # Number of features?
    print "Number of features",features_dim
    
    # Minimum value?
    print "Price minimum value: %.3f" % np.min(housing_prices)
    
    # Maximum Value?
    print "Price maximum value: %.3f" % np.max(housing_prices)
    
    # Calculate mean?
    print "Price mean value: %.3f" % np.mean(housing_prices)
    
    # Calculate median?
    print "Price median:", np.median(housing_prices)
    
    # Calculate standard deviation?
    print "Price STD: %.3f" % np.std(housing_prices)
    
    
    n, (smin, smax), sm, sv, ss, sk = stats.describe(housing_prices)
    sstr = 'Skewness = %6.4f, kurtosis = %6.4f'
    print sstr %(ss, sk)
    threquency(housing_prices)
    
    

def performance_metric(label, prediction):
    """Calculate and return the appropriate performance metric."""

    ###################################
    ### Step 2. YOUR CODE GOES HERE ###
    #The mean squared error metric is used due the regression model. 
    #The MSE is classic metic for regression models
    #The DecisionTreeRegressor regressor supports only MSE to measure the quality
    #The MSE is sensitive for an errors in a data set, but looking on Boston Data
    #I conlude that 16 houses has 50K price and that houses are in the  outlier zone
    #I don`t want to ignore this fact.
    
    ###################################
#
    return mean_squared_error(label, prediction)
#    return mean_absolute_error(label, prediction)
        
#    return explained_variance_score(label, prediction, multioutput='raw_values')
#    return median_absolute_error(label, prediction)
#    return r2_score(label, prediction)
    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
#    pass


def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into training and testing set."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target
    
#    cut_outliers(city_data)
    
    
    ###################################
    ### Step 3. YOUR CODE GOES HERE ###
    rows, cols = X.shape
    rs = cva.ShuffleSplit(rows,
                          1,
                          test_size = .25,
                          random_state = 1111
                          )
    print "Testing size:", rs.n_test
    print "Training size:", rs.n_train
    
    #create arrays with shuffled elements
    for tr_indxs, ts_indxs in rs:
        X_train = np.zeros((len(tr_indxs),cols),dtype=float)
        y_train = np.zeros((len(tr_indxs),1),dtype=float)
        X_test = np.zeros((len(ts_indxs),cols),dtype=float)
        y_test = np.zeros((len(ts_indxs),1),dtype=float)
        #fill training arrays        
        for i, train_i in enumerate(tr_indxs):
            X_train[i,:] = X[train_i,:]
            y_train[i] = y[train_i]
        #fill testing arrays
        for i, test_i in enumerate(ts_indxs):
            X_test[i,:] = X[test_i,:]
            y_test[i] = y[test_i]
            

                                                          

    ###################################

    return X_train, y_train, X_test, y_test


def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.linspace(1, len(X_train), 50)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))
    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):

        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth, random_state=1111)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))
        

    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err)


def learning_curve_graph(sizes, train_err, test_err):
    """Plot training and test error as a function of the training size."""
    
    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, label = 'test error')
    pl.plot(sizes, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()



def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))
        

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision tree learn."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.show()
    
def vizualize_model(max_depth,regressor,city_data):
    
#    X_train, y_train, X_test, y_test = split_data(city_data)
#    regressor = DecisionTreeRegressor(max_depth = max_depth)
#    regressor.fit(X_train, y_train)
#    out_city_data = cut_outliers(city_data,outlier=0.2)
    
    X, y = city_data.data, city_data.target
    y_pred = regressor.predict(X)
    
    fig, ax = pl.subplots()
    ax.scatter(y,y_pred)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    pl.show()

def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor

    regressor = DecisionTreeRegressor()
    
    parameters = {'max_depth':(4,5,6),'max_features': [3,5,6,7,8,9,10]}

    ###################################
    ### Step 4. YOUR CODE GOES HERE ###
    ###################################

    # 1. Find the best performance metric
    # should be the same as your performance_metric procedure
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    my_scorer = make_scorer(performance_metric, greater_is_better=False)
       
    reg = grid_search.GridSearchCV(regressor, parameters,cv=11, scoring=my_scorer, n_jobs=10)
#    reg = grid_search.RandomizedSearchCV(regressor, parameters,cv=11, scoring=my_scorer, n_jobs=10)
    
    # 2. Use gridearch to fine tune the Decision Tree Regressor and find the best model
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV

    # Fit the learner to the training data

    print "Final Model: "
    print reg.fit(X, y)
    print "Best estimator: "
    print reg.best_estimator_
    
    print "Best parameters: "
    print reg.best_params_
    
    print "Details of SearchGridCV:"
    for _metrics in reg.grid_scores_:
        print _metrics
#    print reg.grid_scores_
    print "Best score"
    print reg.best_score_
    
    print "Prediction:"
    # Use the model to predict the output of a particular sample
    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    y = reg.predict(x)
    print "House: " + str(x)
    print "Prediction: " + str(y)
    
    indexes = find_nearest_neighbor_indexes(x, X)
    sum_prices = []
    for i in indexes:
        sum_prices.append(city_data.target[i])
    neighbor_avg = np.mean(sum_prices)
    print "Nearest Neighbors average: " +str(neighbor_avg)
    
    return (reg.best_params_['max_depth'],reg.best_params_['max_features'],reg.best_score_,y)


def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()
    
    

    # Explore the data
    explore_city_data(city_data)
    
    out_city_data = cut_outliers(city_data,outlier=0.1)
    
    explore_city_data(out_city_data)
    

    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)
#
    # Learning Curve Graphs
    max_depths = range(1,11 )
    for max_depth in max_depths:
        learning_curve(max_depth, X_train, y_train, X_test, y_test)

    # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    city_data2=cut_outliers(city_data,0.05)
    # Tune and predict Model
    n=20
    best_depth=[]
   
    for i in range(0,n):
        print "iter"+str(i)
        best_depth.append( fit_predict_model(out_city_data))
    
    a=abs(np.array(best_depth))
    
#    Results
    print "List of evaluations"
    print"Max_depth","\tNum_of_features","\tMSE","\tPredicted"
    print a
    print "MSE:", np.min(a[:,2]), "Predicted cost:", np.min(a[:,3])
        
    

if __name__ == "__main__":
    main()
