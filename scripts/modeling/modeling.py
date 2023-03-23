# Imports
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle


def train_knn_model(X_train,y_train, k):
    """Train KNeighborsClassifier given train dataset and number of neighbors

    Parameters
    ----------
    X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.  
    y_train : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values. Will be cast to X's dtype if necessary.
    k : int, default=5
        Number of neighbors to use by default for kneighbors queries.

    Returns
    -------
    sklearn model
        KNeighborsClassifier model
    """
    knn = KNeighborsClassifier(n_neighbors = k)#, metric='minkowski')
    knn.fit(X_train,y_train)
    return knn


def calc_accuracy_score(y_test, yhat):
    """Calculates accuracy score of knn model.

    Parameters
    ----------
    y_test : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values as given in test dataset to compare with.
    yhat : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values as predicted with model.

    Returns
    -------
    float
        Accuracy score of knn model
    """
    return accuracy_score(y_test,yhat)


def calc_confusion_matrix(y_test, yhat):
    """Calculated the confusion matrix.

    Parameters
    ----------
    y_test : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values as given in test dataset to compare with.
    yhat : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values as predicted with model.

    Returns
    -------
    pd.DataFrame
        Confusion matrix as a DataFrame.
    """
    return pd.DataFrame(confusion_matrix(y_test, yhat))
    

def knn_experiment_with_range_for_k(X_train, y_train, X_test, y_test, lower_boundary_range, upper_boundary_range):
    """Experiments a number of KNN models given a range of neighbors.
    This function requires a train_test_split to be done before!

    Parameters
    ----------
    X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.  
    y_train : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values. Will be cast to X's dtype if necessary.
    X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
        Test data.  
    y_test : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values as given in test dataset to compare with.
    lower_boundary_range : int
        Integer to define lower boundary of range for number of neighbors.
    upper_boundary_range : int
        Integer to define upper boundary of range for number of neighbors.

    Returns
    -------
    pd.DataFrame
        DataFrame with the accuracy score for multiple knn models.
    """
    df_accuracy_score = pd.DataFrame(columns=['Aantal_buren', 'Nauwkeurigheidsscore'])

    for i in range(lower_boundary_range, upper_boundary_range+1):
        knn = train_knn_model(X_train, y_train, k=i)
        yhat = knn.predict(X_test)
        acc = calc_accuracy_score(y_test,yhat)
        df_accuracy_score = pd.concat([df_accuracy_score, pd.DataFrame(data=[{'Aantal_buren':i, 'Nauwkeurigheidsscore':acc}], index=[i])])

    return df_accuracy_score


def knn_experiment(df, y_column, lower_boundary_range, upper_boundary_range):
    """Experiments a number of KNN models given a range of neighbors.
    This function includes a train_test_split!

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split into train and test dataset
    y_column : str
        Name of the target variable.
    lower_boundary_range : int
        Integer to define lower boundary of range for number of neighbors.
    upper_boundary_range : int
        Integer to define upper boundary of range for number of neighbors.

    Returns
    -------
    pd.DataFrame
        DataFrame with the accuracy score for multiple knn models.
    """
    list_variables = list(set(df.columns) - set([y_column]))
    X=df[list_variables]
    y=df[y_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size = 0.2, random_state=1)
    df_accuracy_score = knn_experiment_with_range_for_k(X_train, y_train, X_test, y_test, lower_boundary_range, upper_boundary_range)
    return df_accuracy_score


def knn_with_specific_neighbor(df, k, y_column):
    """Train KNN model with specific neighbor and return model AND confusion matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split into train and test dataset
    k : int, default=5
        Number of neighbors to use by default for kneighbors queries.
    y_column : str
        Name of the target variable.

    Returns
    -------
    sklearn model
        KNeighborsClassifier model
    pd.DataFrame
        Confusion matrix as a DataFrame.
    """
    # Train_test_split
    list_variables = [e for e in df.columns if e not in [y_column]]
    X=df[list_variables]
    y=df[y_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size = 0.2, random_state=1)
    knn = train_knn_model(X_train, y_train, k=k)
    yhat = knn.predict(X_test)
    df_confusion = calc_confusion_matrix(y_test,yhat)
    
    return knn, df_confusion


def find_best_number_neighbors(X_train, y_train, X_test, y_test):
    """Finds best number of neighbors in a range of 0,20.

    Parameters
    ----------
    X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.  
    y_train : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values. Will be cast to X's dtype if necessary.
    X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
        Test data.  
    y_test : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values as given in test dataset to compare with.

    Returns
    -------
    int
        Number of neighbors that is fitted best according to the accuracy score.
    """
    acc = []

    for i in range(1,20):
        train_knn_model(X_train, y_train, k=i)
        yhat = knn.predict(X_test)
        acc.append(accuracy_score(y_test,yhat))
        print(f"For k = {i} : {accuracy_score(y_test,yhat)}")
        best_k = acc.index(max(acc))
        print(f"Best result for accuracy score is {max(acc)} with {best_k} neighbors")

    return best_k


def create_knn_model(df, y_column):
    """Trains a KNN model with the best possible neighbors in a range of 0 to 20.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split into train and test dataset
    y_column : str
        Name of the target variable.

    Returns
    -------
    sklearn model
        KNeighborsClassifier model
    """
    
    # Train_test_split
    # list_variables = list(set(df.columns) - set([y_column]))
    list_variables = [e for e in df.columns if e not in [y_column]]
    X=df[list_variables]
    y=df[y_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size = 0.2, random_state=1)
    # Find best number of neighbours
    best_k = find_best_number_neighbours(X_train, y_train, X_test, y_test)

    # Fit model
    knn = KNeighborsClassifier(n_neighbors = best_k)
    model = knn.fit(X_train,y_train)

    return model


def save_model2pickle(model, default_filename):
    """Saves model in pickle format to folder 'models' in this project.

    Parameters
    ----------
    model : sklearn model
        Trained model of sklearn.
    default_filename : str
        Filename for model.
    """
    # save the model to disk
    filename = f'models/{default_filename}.sav'
    pickle.dump(model, open(filename, 'wb'))


def load_model_from_pickle(default_filename):
    """Loads pickle file with trained model. 

    Parameters
    ----------
    default_filename : str
        Filename for model.

    Returns
    -------
    sklearn model
        Trained model of sklearn.
    """
    filename = f'models/{default_filename}.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model