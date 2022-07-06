import numpy as np


class Superellipsoid:
    def __init__(self, a, b, c, p, r0, orientation):
        self.a = a
        self.b = b
        self.c = c
        self.p = p
        self.r0 = r0
        self.orientation = orientation

    def r_relative(self, r):
        return np.dot(np.linalg.inv(self.orientation), np.subtract(r, self.r0))

    def inner_shape_function(self, r):
        r_relative = self.r_relative(r)
        return np.power((r_relative[0] / self.a), 2 * self.p) + np.power((r_relative[1] / self.b), 2 * self.p) \
                + np.power((r_relative[2] / self.c), 2 * self.p)

    def shape_function(self, r):
        return np.power(self.inner_shape_function(r), 1/self.p) - 1

    def gradient(self, r_C):
        p = self.p
        r_relative = self.r_relative(r_C)
        grad = np.array([(2 * p * (r_relative[0] / self.a) ** (2 * p - 1)) / self.a,
                         (2 * p * (r_relative[1] / self.b) ** (2 * p - 1)) / self.b,
                         (2 * p * (r_relative[2] / self.c) ** (2 * p - 1)) / self.c])
        return np.dot(self.orientation, grad)

    def nabla(self, r_C):
        inner_shape_func = self.inner_shape_function(r_C)[0]
        return (1/self.p) * inner_shape_func ** (1/self.p - 1) * self.gradient(r_C)

    def hessian(self, r_C):
        p = self.p
        r_relative = self.r_relative(r_C)
        H = np.array([((2 * p * (2 * p - 1) * (r_relative[0][0] / self.a) ** (2 * p - 2))/(self.a ** 2), 0, 0),
                      (0, (2 * p * (2 * p - 1) * (r_relative[1][0] / self.b) ** (2 * p - 2))/(self.b ** 2), 0),
                      (0, 0, (2 * p * (2 * p - 1) * (r_relative[2][0] / self.c) ** (2 * p - 2))/(self.c ** 2))])
        return np.dot(np.dot(self.orientation, H), self.orientation.transpose())

    def nabla_2(self, r_C):
        inner_shape_func = self.inner_shape_function(r_C)[0]
        gradient = self.gradient(r_C)
        p = self.p
        return np.dot(1/self.p * inner_shape_func ** (1/self.p - 1), self.hessian(r_C)) \
               + 1/p * (1/p - 1) * inner_shape_func ** (1/p - 2) * gradient * np.transpose(gradient)

