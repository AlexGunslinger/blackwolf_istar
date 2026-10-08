import os
import pickle
def foo(x,y):
    return x + y

def extract_cylinders(station):
    station.extract_cylinder(refine=True)
    station.ptc = None
    return station

    # out_dir = os.path.join(data_dir,station.name + '-output')
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    # # save station without points
    # station.ptc = None
    # out_fp = os.path.join(out_dir, f'{station.name}.pkl')
    # with open(out_fp, 'wb') as f:
    #     pickle.dump(station, f)
    