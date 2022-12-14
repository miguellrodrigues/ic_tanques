from scipy.io import loadmat
import scipy.signal as sig
import numpy as np


def load_data(experiment_path, keys=['t', 'Level3CM']):
    data = loadmat(f'{experiment_path}.mat')

    t = np.array(data['t']).flatten()

    # filter parameters
    b = np.array([
        0.13575525, 0.13575525
    ])

    a = np.array([
        1.,        -0.7284895
    ])

    key_datas = []

    for key in keys:
        if key.startswith('Level'):
            key_data = np.array(data[key]).flatten()
            key_datas.append(
                sig.lfilter(b, a, key_data)
            )

    return t, key_datas
