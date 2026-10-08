import pandas as pd
import numpy as np
import re
from enum import IntEnum

METER2FOOT = 3.2808399
FOOT2METER = 0.3048

def extract_station_ids(file_names, sorted=True):
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

class Task(IntEnum):
    INIT = 0
    JOB_SUMMARY = 1
    CYLINDER_EXTRACTION = 2
    ROUGH_ALIGNMENT = 3
    FINE_ALIGNMENT = 4
    EVALUATION = 5
    UPLOAD_RESULTS = 6
    END = 7