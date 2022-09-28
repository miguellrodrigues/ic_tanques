import numpy as np
import matplotlib.pyplot as plt
import json


with open('validacao.json', 'r') as file:
    data = json.load(file)


t = np.array(data[0]['points'], dtype=np.float64)
Level3CM = np.array(data[1]['points'], dtype=np.float64)
Leval4CM = np.array(data[2]['points'], dtype=np.float64)
Pump1PC = np.array(data[3]['points'], dtype=np.float64)


# def moving_average(a, n=3):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n

# filtered_level3 = moving_average(Level3CM, n=499)
# filtered_level4 = moving_average(Leval4CM, n=499)
# t = t[:len(filtered_level3)]
# Pump1PC = Pump1PC[:len(filtered_level3)]

np.save('./data/t', t)
np.save('./data/level3', Level3CM)
np.save('./data/level4', Leval4CM)
np.save('./data/pump1', Pump1PC)

# plt.plot(t, filtered_level3, label='Level3CM')
# plt.plot(t, filtered_level4, label='Leval4CM')
# plt.plot(t, Pump1PC, label='Pump1PC')

plt.show()
