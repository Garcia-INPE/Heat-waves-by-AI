import os

def count_classes(y_list):
    # y_list = y_train; class_labels = [0, 1]
    """Count the occurrence of each class in a list and return a dictionary with class labels as keys and their counts as values.

    Parameters:
    - y_list: list of class labels (e.g., [0, 1, 0, 1, 1])

    Returns:
    - class_counts: dictionary with class labels as keys and their counts as values (e.g., {0: 2, 1: 3})
    """
    class_counts = {}
    for label in y_list:
        if int(label) in class_counts:
            class_counts[int(label)] += 1
        else:
            class_counts[int(label)] = 1
    return class_counts

