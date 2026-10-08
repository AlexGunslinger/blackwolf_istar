import math
import re
import pandas as pd
import numpy as np


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    try:
        return math.degrees(
            math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
        )
    except Exception:
        return 0

def extract_station_ids(file_names, sorted=True):
    """extracts identifying numbers for scans from their file names.
    It is important to have all the files sharing the same format

    Args:
        file_names (list): list of file names
        sorted (bool, optional): whether to sort scan numbers. Defaults to True.

    Returns:
        names (numpy array):    scan numbers in string format (e.g. '001')
        ids (numpy int array):  scan numbers in numeric format (e.g 1)
        file_names (array):     sorted (-if requested) file_names
    """
    # extract station id numbers from a list of station names
    labels = [re.findall('[0-9]+', fn) for fn in file_names]
    df = pd.DataFrame(labels)
    for i in df.columns:
        if df[i].nunique() > 1:
            names = df[i].values
            ids = names.astype(np.int8)
            if sorted:
                ind = np.argsort(ids)
                names = names[ind]
                ids = ids[ind]
                file_names = np.array(file_names)[ind]
                return names, ids, file_names
    return np.array(file_names), np.arange(len(file_names)), file_names