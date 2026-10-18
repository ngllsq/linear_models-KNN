from tkinter import W
import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    if predictions.ndim > 1:
      max_pred = np.max(predictions, axis=-1, keepdims=True)
      exp_pred = np.exp(predictions - max_pred)
      sum_exp = np.sum(exp_pred, axis=-1, keepdims=True)
      probs = exp_pred / sum_exp

    else:
      max_pred = np.max(predictions, axis=0, keepdims=True)
      exp_pred = np.exp(predictions - max_pred)
      sum_exp = np.sum(exp_pred, axis=0, keepdims=True)
      probs = exp_pred / sum_exp
    # raise Exception("Not implemented!")
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    loss = -np.log(probs[target_index])
    # raise Exception("Not implemented!")
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")
    max_pred = np.max(predictions, axis=-1, keepdims=True)
    exp_pred = np.exp(predictions - max_pred)
    probs = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)

    if predictions.ndim > 1:
      target_index = target_index.reshape(-1)
      truth_probs = probs[np.arange(len(target_index)), target_index]
      loss = np.mean(-np.log(truth_probs))
      dprediction = probs.copy()
      dprediction[np.arange(len(target_index)), target_index]-=1
      dprediction /= len(target_index)

    else:
      loss = -np.log(probs[target_index])
      dprediction = probs.copy()
      dprediction[target_index] -= 1

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    probs = softmax(predictions)
    truth_probs = probs[np.arange(len(target_index)), target_index]
    loss = np.mean(-np.log(truth_probs))
    target_index = target_index.reshape(-1)

    dprediction = probs.copy()
    dprediction[np.arange(len(target_index)), target_index]-=1
    dprediction /= len(target_index)
    dW= X.T @ dprediction
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            # raise Exception("Not implemented!")

            for batch in batches_indices:
                x_batch = X[batch]
                y_batch = y[batch]
                predictions = x_batch @ self.W
                max_pred = np.max(predictions, axis=-1, keepdims=True)
                logs = np.exp(predictions - max_pred)
                probs = logs / np.sum(logs, axis=-1, keepdims=True)

                truth_predictions = probs[np.arange(len(y_batch)), y_batch]
                loss = np.mean(-np.log(truth_predictions)) + reg * np.sum(self.W**2)
                loss_history.append(loss)

                dpredictions = probs.copy()
                dpredictions[np.arange(len(y_batch)), y_batch]-=1
                dpredictions /= len(y_batch)

                grad = x_batch.T @ dpredictions + 2 * reg * self.W
                self.W-= learning_rate * grad

            print("Epoch %i, loss: %f" % (epoch, loss))
                # end

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=int)

        # TODO Implement class prediction
        y_pred = X @ self.W
        y_pred = np.argmax(y_pred, axis=1)
        
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
