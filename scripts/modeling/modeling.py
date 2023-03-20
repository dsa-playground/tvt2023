# Imports
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle


def train_knn_model(X_train,y_train, k):
    knn = KNeighborsClassifier(n_neighbors = k)#, metric='minkowski')
    knn.fit(X_train,y_train)
    return knn


def calc_accuracy_score(y_test, yhat):
    return accuracy_score(y_test,yhat)

def calc_confusion_matrix(y_test, yhat):
    return pd.DataFrame(confusion_matrix(y_test, yhat))
    

def knn_experiment_with_range_for_k(X_train, y_train, X_test, y_test, lower_boundary_range, upper_boundary_range):
    df_accuracy_score = pd.DataFrame(columns=['Aantal_buren', 'Nauwkeurigheidsscore'])

    for i in range(lower_boundary_range, upper_boundary_range+1):
        knn = train_knn_model(X_train, y_train, k=i)
        yhat = knn.predict(X_test)
        acc = calc_accuracy_score(y_test,yhat)
        df_accuracy_score = pd.concat([df_accuracy_score, pd.DataFrame(data=[{'Aantal_buren':i, 'Nauwkeurigheidsscore':acc}], index=[i])])

    return df_accuracy_score


def knn_experiment(df, y_column, lower_boundary_range, upper_boundary_range):
    # Train_test_split
    list_variables = list(set(df.columns) - set([y_column]))
    X=df[list_variables]
    y=df[y_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size = 0.2, random_state=1)
    df_accuracy_score = knn_experiment_with_range_for_k(X_train, y_train, X_test, y_test, lower_boundary_range, upper_boundary_range)
    return df_accuracy_score

def knn_with_specific_neighbour(df, k, y_column):
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


def find_best_number_neighbours(X_train, y_train, X_test, y_test):
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
    # save the model to disk
    filename = f'models/{default_filename}.sav'
    pickle.dump(model, open(filename, 'wb'))


def load_model_from_pickle(default_filename):
    filename = f'models/{default_filename}.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model