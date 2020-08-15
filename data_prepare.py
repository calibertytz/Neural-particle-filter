import numpy as np
from sklearn.model_selection import train_test_split

# data prepare
'''
frog tracking
dx_t = 3x_t(1-x_t^2)dt + dw_t

dv_t = x_tdt + sigma_v*d(beta_t)
da_t = tanh(2x_t)dt + sigma_a*d(gamma_t)
'''

class frogsystem(object):
    def __init__(self, dt = None, sigma_v = None, sigma_a = None, x0 = None, N = None, M = None, a = None):
        self.x = np.random.normal() if x0 is None else x0
        self.dt = 0.01 if dt is None else dt
        self.sigma_v = 1 if sigma_v is None else sigma_v
        self.sigma_a = 1 if sigma_a is None else sigma_a
        self.f = lambda x: x + 3 * x * (1 - x**2) * self.dt
        self.h1 = lambda x: x * self.dt
        self.h2 = lambda x: np.tanh(2 * x) * self.dt
        self.N = 100 if N is None else N
        self.M = 100 if M is None else M
        self.a = 0.3 if a is None else a

        self.real_dataset = []
        self.measurement_dataset = []

    def generate_data(self):
        '''
        :param N: sequence length
        :param M: number of dataset copys
        :return:
        '''
        for _ in range(self.M):
            real_state, measurement = [], []
            for _ in range(self.N):
                real_state.append(self.x)
                v = self.h1(self.x) + np.random.normal(0, self.sigma_a)
                a = self.h2(self.x) + np.random.normal(0, self.sigma_v)
                measurement.append([v, a])
                self.x = self.f(self.x) + np.random.normal()
            self.real_dataset.append(np.array(real_state))
            self.measurement_dataset.append((np.array(measurement)))

        self.real_dataset, self.measurement_dataset = np.array(self.real_dataset), np.array(self.measurement_dataset)

    def split_dataset(self):
        x_train, x_test, y_train, y_test = \
            train_test_split(self.measurement_dataset, self.real_dataset, test_size=self.a)
        return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    frog = frogsystem()
    frog.generate_data()
    x_train, x_test, y_train, y_test = frog.split_dataset()
    #print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)
