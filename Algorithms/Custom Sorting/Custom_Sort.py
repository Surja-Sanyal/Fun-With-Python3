#Custom Sort: Time Complexity better than O(nlog(n/2))


import sys
import math
import random
import datetime
import numpy as np


def custom_sort(input_array, first, last, delimeter, pieces):

    if (first + pieces <= last):

        split_loc = int((last - first)/pieces)
        
        [custom_sort(input_array, first + part * split_loc, (first + (part + 1) * split_loc) if part < pieces - 1 else last, delimeter, pieces) for part in range(pieces)]

        input_array[first: last] = joined(list(input_array), first, last, delimeter, pieces)
        
    elif (pieces > 2 and first < last):
        
        custom_sort(input_array, first, last, delimeter, pieces = int(pieces/10) if last - first > 10 else pieces - 1)


def joined(input_array, first, last, delimeter, pieces):

    split_loc = int((last - first)/pieces)
    
    intermediate_array = [input_array[first + part * split_loc: (first + (part + 1) * split_loc) if part < pieces - 1 else last] + [delimeter] for part in range(pieces)]
    
    positions = np.zeros(pieces, dtype=np.int16)
    
    for k in range(first, last):
        
        current_comparison_elements = [intermediate_array[i][j] for i, j in zip(range(pieces), positions)]
        
        input_array[k] = min(current_comparison_elements)
        
        positions[current_comparison_elements.index(input_array[k])] += 1
    
    return input_array[first: last]


def main(max_val, pieces = 2):

    input_array = np.random.random_integers(0, max_val, max_val)
    
    print ("\nUnsorted=", input_array)
    
    custom_sort(input_array, 0, max_val, max(input_array) + 1, pieces)
    
    print ("\nSorted=", input_array)


if __name__=="__main__":

    start = datetime.datetime.now()
    
    max_val = 100000
#    pieces = 2
    pieces = int(max_val/10) if max_val > 10 else 3
    recursion_val = 20000 
    print ("\nSorting", max_val, "values")
    print ("\nCurrent recursion limit=", sys.getrecursionlimit())
    print ("\nSetting recursion limit=", recursion_val)
    sys.setrecursionlimit(recursion_val)
    
    main(max_val, pieces)
    
    print ("\nTotal time taken =", datetime.datetime.now() - start)


