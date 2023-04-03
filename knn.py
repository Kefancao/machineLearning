import numpy as np
import matplotlib.pyplot as plt


# Data loader
def load_knn_data():
  test_inputs = np.genfromtxt('test_inputs.csv', delimiter=',')
  test_labels = np.genfromtxt('test_labels.csv', delimiter=',')
  train_inputs = np.genfromtxt('train_inputs.csv', delimiter=',')
  train_labels = np.genfromtxt('train_labels.csv', delimiter=',')
  return train_inputs, train_labels, test_inputs, test_labels

def predict_knn(x, inputs, labels, k_neighbours):
    """This function implements the KNN classifier to predict the label of a data point. 
    Measure distances with the Euclidean norm (L2 norm). When there is a tie between two 
    (or more) labels, break the tie by choosing any label.

    Args:
        x: Input data point for which we'd like to predict
        inputs: matrix of data points in which neighbours will be found 
                (numpy array of N data points x M features)
        labels: vector of labels associated with the data points (numpy array of N labels)
        k_neighbours: # of nearest neighbours that will be used

    Returns:
        Predicted label
    """
    # Combine the labels with the respective differences from x.
    distances = [[labels[i], np.linalg.norm(x-inputs[i])] for i in range(len(inputs))]
    # Sort according to the differences.
    distances = sorted(distances, key=(lambda x : x[1]))
    # Get the k_neighbours closest to x.
    ma_neighbs = [distances[i][0] for i in range(k_neighbours)]
    # Find the mode of neighbours
    return max(set(ma_neighbs), key=ma_neighbs.count)


def eval_knn(inputs, labels, train_inputs, train_labels, k_neighbours):
    """
    Function that evaluates the accuracy of the KNN classifier on a dataset. 
    The dataset to be evaluated consists of (inputs, labels). 
    The dataset used to find nearest neighbours consists of (train_inputs, train_labels).
    
    Args:
        inputs: matrix of input data points to be evaluated (numpy array of N data points x M features)
        labels: vector of target labels for the inputs (numpy array of N labels)
        train_inputs: matrix of input data points in which neighbours will be found
                      (numpy array of N' data points x M features)
        train_labels: vector of labels for the training inputs (numpy array of N' labels)
        k_neighbours: # of nearest neighbours to be used (integer)

    Returns:
        accuracy: percentage of correctly labeled data points (float)
    """
    accuracy = 0
    for i in range(len(inputs)):
        pre = predict_knn(inputs[i], train_inputs, train_labels, k_neighbours)
        if (pre == labels[i]):
            accuracy += 1

    return accuracy/len(inputs)


def cross_validation_knn(k_folds, hyperparameters, inputs, labels):
    """
    This function performs k-fold cross validation to
    determine the best number of neighbours for KNN.

    Args:
        k_folds: # of folds in cross-validation (integer)
        hyperparameters: list of hyperparameters where each hyperparameter 
                         is a different # of neighbours (list of integers)
        inputs: matrix of data points to be used when searching for neighbours
                (numpy array of N data points by M features)
        labels: vector of labels associated with the inputs (numpy array of N labels)

    Returns:
        best_hyperparam: best # of neighbours for KNN (integer)
        best_accuracy: accuracy achieved with best_hyperparam (float)
        accuracies: vector of accuracies for the corresponding hyperparameters 
                    (numpy array of floats)
    """
    # dummy assignments until the function is filled in
    best_hyperparam = 0
    best_accuracy = 0
    # accuracies = np.zeros(len(hyperparameters))
    accuracies = []
    sector_size = int(len(inputs)/k_folds)
    for parm in hyperparameters:
        tmp_accuracies = []
        for i in range(1, k_folds + 1):
        # Get the test and train data for k fold.
            test_inputs = inputs[(i-1)*sector_size:i*sector_size]
            test_labels = labels[(i-1)*sector_size:i*sector_size]
            train_inputs = np.concatenate((inputs[:(i-1)*sector_size], inputs[i*sector_size:]))
            train_labels = np.concatenate((labels[:(i-1)*sector_size], labels[i*sector_size:]))
            tmp_accuracies.append(eval_knn(test_inputs, test_labels, train_inputs, train_labels, parm))
            print(f'Computing {i} fold with {parm} neighbours out of {k_folds} folds.')
        accuracies.append(np.mean(tmp_accuracies))
    # Get hyperparameter with the highest accuracy.
    best_hyperparam = np.argmax(accuracies) + 1
    best_accuracy = np.max(accuracies)
    
    return best_hyperparam, best_accuracy, accuracies


# Plots the performance of each k hyperparameter. 
def plot_knn_accuracies(accuracies,hyperparams):
  print(len(accuracies))
  print(len(hyperparams))
  plt.plot(hyperparams,accuracies)
  plt.ylabel('accuracy')
  plt.xlabel('k neighbours')
  plt.show()  
  
# Load data. 
train_inputs, train_labels, test_inputs, test_labels = load_knn_data()

# number of neighbours to be evaluated by cross validation
hyperparams = range(1,31)
k_folds = 10
best_k_neighbours, best_accuracy, accuracies = cross_validation_knn(k_folds, hyperparams, train_inputs, train_labels)

# plot results
plot_knn_accuracies(accuracies, hyperparams)
print('best # of neighbours k: ' + str(best_k_neighbours))
print('best cross validation accuracy: ' + str(best_accuracy))
# evaluate with best # of neighbours
accuracy = eval_knn(test_inputs, test_labels, train_inputs, train_labels, best_k_neighbours)
print('test accuracy: '+ str(accuracy))

