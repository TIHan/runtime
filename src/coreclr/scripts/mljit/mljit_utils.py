import os
import re
import collections
import numpy as np
import statistics
import json
import itertools
import functools

# From https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
# Print iterations progress
def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def for_all(predicate, xs):
    return all(predicate(x) for x in xs)

def partition_yield(xs, size):
    for i in range(0, len(xs), size):
        yield list(itertools.islice(xs, i, i + size))

def partition(xs, size):
    return list(partition_yield(xs, size))

def map_dict(f, my_dictionary):
   return {k: f(k, v) for k, v in my_dictionary.items()}

def map_dict_value(f, my_dictionary):
   return {k: f(v) for k, v in my_dictionary.items()}

def flatten(xss):
    return [x for xs in xss for x in xs]