import pandas as pd

def dropNA(table):
    return (table.dropna(axis=0, how='any'))