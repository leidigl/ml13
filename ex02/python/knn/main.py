''' main.py '''
import sys
import csv
import numpy as np

import kNN

def main(argv):
    # read csv file
    csv_file = argv[1]
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        features = np.array(reader.next())
        values = np.array([row for row in reader], dtype = 'float')

    classes = np.unique(values[:, features.size-1])     # vector with the specific classes

    knn = kNN.KNN(values)
    k = 3
    
    # Problem 3
    x1 = np.array([4.1, -0.1, 2.2])
    c1, p1 = knn.classify(x1, k, classes)

    x2 = np.array([6.1, 0.4, 1.3])
    c2, p2 = knn.classify(x2, k, classes)
    
    print(str(x1) + ' belongs to class ' + str(c1) + ' with a probability of ' + str(p1))
    print(str(x2) + ' belongs to class ' + str(c2) + ' with a probability of ' + str(p2))


    # Problem 4
    reg1 = knn.regress(x1, k)
    reg2 = knn.regress(x2, k) 
    
    print('Regression for '+ str(x1) + ' results in ' + str(reg1))
    print('Regression for '+ str(x2) + ' results in ' + str(reg2))


if __name__ == '__main__':
    main(sys.argv)
    
    
