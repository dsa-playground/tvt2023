# Imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle


def find_best_number_neighbours(X_train, y_train, X_test, y_test):
    acc = []

    for i in range(1,20):
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(X_train,y_train)
        yhat = knn.predict(X_test)
        acc.append(accuracy_score(y_test,yhat))
        print(f"For k = {i} : {accuracy_score(y_test,yhat)}")
        best_k = acc.index(max(acc))
        print(f"Best result for accuracy score is {max(acc)} with {best_k} neighbors")

    return best_k


def create_knn_model(df, y_column):
    
    # Train_test_split
    list_variables = list(set(df.columns) - set([y_column]))
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