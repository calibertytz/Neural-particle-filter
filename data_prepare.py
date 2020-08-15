import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data prepare
'''
frog tracking
dx_t = 3x_t(1-x_t^2)dt + dw_t

dv_t = x_tdt + sigma_v*d(beta_t)
da_t = tanh(2x_t)dt + sigma_a*d(gamma_t)
'''

class frogsystem(object):
    def __init__(self, dt = None, sigma_v = None, sigma_a = None, x0 = None):
        self.x = np.random.normal() if x0 is None else x0
        self.dt = 0.01 if dt is None else dt
        self.sigma_v = 1 if sigma_v is None else sigma_v
        self.sigma_a = 1 if sigma_a is None else sigma_a
        self.f = lambda x: x + 3 * x * (1 - x**2) * self.dt
        self.h1 = lambda x: x * self.dt
        self.h2 = lambda x: np.tanh(2 * x) * self.dt

    def generate_data(self, N):
        real_state, measurement = [], []
        for _ in range(N):
            real_state.append(self.x)
            v = self.h1(self.x) + np.random.normal(0, self.sigma_a)
            a = self.h2(self.x) + np.random.normal(0, self.sigma_v)
            measurement.append([v, a])
            self.x = self.f(self.x) + np.random.normal()

        return np.array(real_state), np.array(measurement)


if __name__ == '__main__':
    frog = frogsystem()
    real, obs = frog.generate_data(100)
    #print(obs.shape)
    '''
    plt.plot(real, label = 'real')
    plt.legend()
    plt.show()
    '''

    v, a = obs[:, 0], obs[:, 1]
    df_dic = {'real': real, 'v': v, 'a': a}
    df = pd.DataFrame(df_dic)
    df.to_csv('data.csv')


