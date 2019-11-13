"""
Created on Sun Nov 3 12:47:37 2019

@author: 19233292
"""
import numpy as np
import matplotlib.pyplot as plt

def min_max_normalization(vec):
    min_val = np.min(vec)
    max_val = np.max(vec)
    vec = (vec-min_val) / (max_val-min_val) 
    return vec

class LogisticRegression(object):
    
    def __init__(self, input, classes,learning_rate=0.1,max_num_iter=100000,cost_threshold = 1e-8):
        """Constructor function : initializes all fields of the class. 
        All the fields will be used in the working of Logistic Regression
        
        Parameters:
        input : X used for training nd-array of dimension num_samples X num_feats
        classes : Y - list of labels of training data
        learning_rate : alpha - decides the rate at which the weights will learn default value = 0.1
        max_num_iter : specifies the maximum number of epochs for training
        cost_threshold : specifies the threshold of cost. Training will stop once the cost function will reach this threshold value
        
           """
        self.X = input
        self.classes = classes
        self.num_features = 0
        self.learning_rate = learning_rate
        self.max_num_iter = max_num_iter
        self.cost_threshold = cost_threshold
        if(self.X.ndim > 1):
            self.num_features = self.X.shape[1]
        self.unique_classes = list(set(self.classes))
        self.num_classes = len(self.unique_classes)
        indexes = [self.unique_classes.index(c) for c in self.classes]
        self.y = [[0]*len(set(self.classes)) for i in range(len(self.classes))]
        for i in range(len(self.y)):
            self.y[i][indexes[i]] = 1
        self.y = np.asarray(self.y)
        self.W = np.random.rand(self.num_features, self.num_classes) * 0.001 
        self.bias = np.zeros(self.num_classes)   
        self.all_cost = []
        self.predictions = []
    
    def softmax_func(self,X): 
        """softmax_func : 
            This function applies softmax to the given x vector passed as parameter. 
        
        Parameters:
        X : numpy array. Input to softmax function
        
        Returns:
        numpy array: Returns softmax applied to input vector x
        
           """
        vec = np.add(np.dot(X, self.W), self.bias)
        exp_vec = np.exp(vec)  
        if exp_vec.ndim == 1: #will be used for training
            return exp_vec / np.sum(exp_vec, axis=0)
        else: #will be used in classification(i.e. prediction)  
            return exp_vec / np.array([np.sum(exp_vec, axis=1)]).T  
        

    def update_params(self, alpha=0.1):
        """update_params: This function update weight vector and bias vector. 
        This function update weight and bias vector in each parameter with respect to the error calculated in each iteration
        
        Parameters:
        alpha : Learning rate. Default value = 0.1 
        
           """
        activation = self.softmax_func(self.X)
        error = self.y - activation
        self.W += alpha * np.dot(self.X.T, error)
        self.bias += alpha * np.mean(error, axis=0)

    def cost_func(self):
        """cost_func : This is the cost function
        This function calculates negative log likelihood of the input with the current weght and bias vectors values.
        The goal of this algorithm is to minimize cost function.
        
        Returns:
        int:Returning value
        
           """
        activation = self.softmax_func(self.X)
        neg_likelihood = - np.mean(np.sum(self.y * np.log(activation) + (1 - self.y) * np.log(1 - activation), axis=1))
        return neg_likelihood
    
    def calculate_accuracy(self,X_test,y_test, write_predictions_to_file = False, filename = "predictions.txt"):
        """calculate_accuracy : this function calculates accuracy 
        
        Parameters:
        X_test : numpy array of dimension num_samples X num_feats. 
        y_test : numpy array or list of labels of actual classes of X_test
        write_predictions_to_file : boolean variable, if this value is true the predictions are written to a file. Default value : False
        filename : string value, this variable will have the value of the filename to write prediction if 
        that option is selected. Default value : predictions.txt
        
        Returns:
        int:Returns Accuracy
        
           """
        count = 0
        for i in range(len(X_test)):
            prediction = self.predict(X_test[i])
            if prediction == y_test[i]:
                count+=1
                self.predictions.append([prediction,y_test[i]])
        
        if write_predictions_to_file == 1 and filename != None:
            self.write_predictions_to_file(filename)
        
        return count/len(X_test)

    def predict(self, X):
        """predict : this function predicts the class of the given X(input)
        This function applies logistic regression and predicts a class for given X
        
        Parameters:
        X : singular X which contains one sample containing all features.
        
        Returns:
        Returns predicted class.
        
           """
        prob_class = np.argmax(self.softmax_func(X))
        return self.unique_classes[prob_class]
    
    def write_predictions_to_file(self,filename):
        """write_predictions_to_file : this function writes the predicted,actual labels into a file.
        The separator used is comma.
        
        Parameters:
        filename : string value. This is filename in which the predictions are written
        
           """
        try:
            file = open(filename, "a+")
            for prediction in self.predictions:
                file.write(str(prediction[0]) + "," + str(prediction[1]) + "\n")
        except Exception as e: 
            print("Unable to write in file. Error : ", e)
        
        
        
    def train(self):
        """train : this function trains the model i.e. fits weight and bias vector. 
        The training stops once the cost threshold is reached.
        
           """
        prev_cost = 0
        for epoch in range(self.max_num_iter):
            self.update_params(alpha = self.learning_rate)
            cost = self.cost_func()
            self.all_cost.append(cost)
            if(abs(cost-prev_cost) < 1e-8):
                break
            prev_cost = cost
            if(cost<self.cost_threshold):
                break
        
    def plot_cost_function(self):
        """plot_cost_function : this takes all the cost values that was computed while training the model and
           plots it against the number of iterations. Uses matplotlib
        
           """
        if(len(self.all_cost) < 1):
            return
        x_plot = [i*1000 for i in range(1,len(self.all_cost[0::1000])+1)]
        fig, ax = plt.subplots()
        ax.plot(x_plot, self.all_cost[0::1000])

        ax.set(xlabel='Number of iterations', ylabel='Cost',
               title='Logistic Regression Cost Function Minimization')
        ax.grid()

        fig.savefig("cost_func.png")
        plt.show()
        
        