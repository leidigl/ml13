''' node.py '''

import numpy as np

class Node(object):
    '''
    class represents a node in a tree having children and a decision
    '''
    def __init__(self, features, values, current_depth, max_depth, classes):
        
        self.features = features
        self.values = values
        self.classes = classes
        
        # further slipping only if maximum depth is not reached and the data does not belong only to one class
        if current_depth < max_depth and np.unique(self.values[:, features.size-1]).size != 1:
            self.split_feature, self.split_value = self.find_decision()
            data_leq, data_g = self.split_data(self.split_feature, self.split_value)
            self.child_leq = Node(self.features, data_leq, current_depth+1, max_depth, self.classes)    # node creates its children on its own
            self.child_g = Node(self.features, data_g, current_depth+1, max_depth, self.classes)

    # determining the feature and its value for splitting optimised by the gini index
    def find_decision(self):
        bins = np.append(self.classes, [self.classes.size])
        split_feature = ''
        split_value = -100000000000000
        optimal_cost = 1000000000000000

        # try each feature and each value within
        for feature in range(0, self.features.size-1):
            for i in np.sort(self.values[:,feature]):
                current_data_leq, current_data_g = self.split_data(feature, i)
                distribution_leq = np.histogram(current_data_leq[:,self.classes.size], bins)[0]
                distribution_g = np.histogram(current_data_g[:, self.classes.size], bins)[0]
                cost_leq = self.gini(distribution_leq)
                cost_g = self.gini(distribution_g)
                total = self.values[:,0].size
                current_cost = cost_leq*np.sum(distribution_leq)/total + cost_g*np.sum(distribution_g)/total
                if current_cost < optimal_cost:
                    optimal_cost = current_cost
                    split_feature = feature
                    split_value = i
                
        return split_feature, split_value
    
    
    # split the data corresponding to a feature and its value
    def split_data(self, split_feature, split_value):

        data_leq = self.values[self.values[:, split_feature]<=split_value]
        data_g = self.values[self.values[:, split_feature]>split_value]

        return data_leq, data_g
    
    
    # computes the gini index of a given distribution
    def gini(self,v):
        sum_all = np.sum(v)
        sum_square = np.sum(np.square(v))
        if np.sum(v)==0:
            return 1.0
        else:
            return 1.0-(sum_square/((sum_all)**2.0))
        
    
    # classifies a new data point
    def classify(self, x):
        if hasattr(self, 'split_feature'):
            if x[self.split_feature] <= self.split_value:
                self.child_leq.classify(x)
            else:
                self.child_g.classify(x)
        else:
            bins = np.append(self.classes, [self.classes.size])
            distribution = np.histogram(self.values[:,self.classes.size], bins)[0]
            maximum = max(distribution)
            c = np.where(distribution == maximum)
            p = 1.0*maximum / np.sum(distribution)
            print(str(x) + ' belongs to class ' + str(c[0]) + ' with a probability of ' + str(p))
        
        
    # visualizes the decision tree
    def visualize(self, current_depth):
        if hasattr(self, 'split_feature'):
            bins = np.append(self.classes, [self.classes.size])
            distribution = np.histogram(self.values[:,self.classes.size], bins)[0]
            print (str(current_depth)  + '\t' + str(distribution)+ '\t' + self.features[self.split_feature] + '<=' + str(self.split_value) +'\t'+ str(self.gini(distribution)))
            self.child_leq.visualize(current_depth+1)
            self.child_g.visualize(current_depth+1)
        else:
            bins = np.append(self.classes, [self.classes.size])
            distribution = np.histogram(self.values[:,self.classes.size], bins)[0]
            print (str(current_depth) + '\t'+str(distribution)+'\t\t'+str(self.gini(distribution)))

    
    
    
