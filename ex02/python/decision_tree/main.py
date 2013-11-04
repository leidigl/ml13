''' main.py '''

import sys
import numpy as np
import csv

import node


def main(argv):
    # read csv file
    csv_file = argv[1]
    with open(csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            features = np.array(reader.next())
            values = np.array([row for row in reader], dtype = 'float')

    classes = np.unique(values[:, features.size-1])     # vector with the specific classes

    root = node.Node(features, values, 0, 2, classes)   # create the root node, which creates its children on its own
    
    root.visualize(0)   # visualize the tree
    
    # Problem 2     
    x1 = np.array([4.1, -0.1, 2.2])
    root.classify(x1)

    x2 = np.array([6.1, 0.4, 1.3])
    root.classify(x2)


if __name__ == '__main__':
    main(sys.argv)
