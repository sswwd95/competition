import pandas as pd
import numpy as np

submission = pd.read_csv('./solar/csv/conv2d_1.csv')


ranges = 336
hours = range(ranges)
submission = submission[ranges:ranges+ranges]

q_01 = submission['q_0.1'].values
q_02 = submission['q_0.2'].values
q_03 = submission['q_0.3'].values
q_04 = submission['q_0.4'].values
q_05 = submission['q_0.5'].values
q_06 = submission['q_0.6'].values
q_07 = submission['q_0.7'].values
q_08 = submission['q_0.8'].values
q_09 = submission['q_0.9'].values

# print(submission[0])

import matplotlib.pyplot as plt
plt.figure(figsize=(18,2.5))
plt.subplot(1,1,1)
plt.plot(hours, q_01, color='red')
plt.plot(hours, q_02, color='#aa00cc')
plt.plot(hours, q_03, color='#00ccaa')
plt.plot(hours, q_04, color='#ccaa00')
plt.plot(hours, q_05, color='#00aacc')
plt.plot(hours, q_06, color='#aacc00')
plt.plot(hours, q_07, color='#cc00aa')
plt.plot(hours, q_08, color='#000000')
plt.plot(hours, q_09, color='blue')
plt.legend()
plt.show()