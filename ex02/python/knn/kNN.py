''' kNN.py '''

import numpy as np

class KNN(object):
    '''
    classdocs
    '''

    def __init__(self, values):
        self.values = values
        
    # classifies a new data point
    def classify(self, x, k, classes):
        sorted_dist = self.find_rows_with_nn(x, k)
        rows = sorted_dist[0:k,0].astype(np.int32)
        
        cl = self.values[rows,self.values.shape[1]-1]
        
        bins = np.append(classes, [classes.size])
        distribution = np.histogram(cl, bins)[0]
        maximum = max(distribution)
        c = np.where(distribution == maximum)   #class
        p = 1.0*maximum / np.sum(distribution)  #probability
        
        return c[0], p
        
    # determines the class by regression
    def regress(self, x, k):
        sorted_dist = self.find_rows_with_nn(x, k)
        summe = 0
        normalization = 0
        for i in range(0, k):
            summe += 1/sorted_dist[i, 1]*self.values[sorted_dist[i,0], self.values.shape[1]-1]
            normalization += 1/sorted_dist[i, 1]
        return summe/normalization
    
    # determines the rows in the values correpsonding to the k nearest neighbors
    def find_rows_with_nn(self, x, k):
        distances = np.zeros([self.values.shape[0],2])
        for i in range(0, self.values.shape[0]):
            j = self.values.shape[1]-1
            distances[i,1] = np.linalg.norm(x-self.values[i,0:j])
            distances[i,0] = i
        
        return np.sort(distances.view('i8,i8'), order=['f1'], axis=0).view(np.float)
        
        
    
