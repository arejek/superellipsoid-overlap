import numpy as np
import sympy as sp


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

        rC1 = sp.Symbol('rC1')
        rC2 = sp.Symbol('rC2')
        rC3 = sp.Symbol('rC3')

        p = self.p

        inner_shape_function_equation = (rC1 / self.a) ** (2 * p) + (rC2 / self.b) ** (2 * p) + (rC3 / self.c) ** (2 * p)

        derivative_1 = inner_shape_function_equation.diff(rC1)
        derivative_2 = inner_shape_function_equation.diff(rC2)
        derivative_3 = inner_shape_function_equation.diff(rC3)

        derivative_1 = sp.lambdify(rC1, derivative_1)
        derivative_2 = sp.lambdify(rC2, derivative_2)
        derivative_3 = sp.lambdify(rC3, derivative_3)

        r_relative = self.r_relative(r_C)

        # to jest 'wlasciwy' gradient
        grad = np.array([derivative_1(r_relative[0]), derivative_2(r_relative[1]), derivative_3(r_relative[2])])

        return np.dot(self.orientation, grad)

    def nabla(self, r_C):

        inner_shape_func = self.inner_shape_function(r_C)[0]
        x = sp.Symbol('x')
        g_func = x ** (1 / self.p)
        g_func_prime = g_func.diff(x)
        g_func_prime = sp.lambdify(x, g_func_prime)

        if inner_shape_func < 0:
            return
        else:
            return g_func_prime(inner_shape_func) * self.gradient(r_C)

    def hessian(self, r_C):

        rC1 = sp.Symbol('rC1')
        rC2 = sp.Symbol('rC2')
        rC3 = sp.Symbol('rC3')

        p = self.p

        inner_shape_function_equation = (rC1 / self.a) ** (2 * p) + (rC2 / self.b) ** (2 * p) + (rC3 / self.c) ** (2 * p)

        derivative_r1 = inner_shape_function_equation.diff(rC1)
        derivative_r2 = inner_shape_function_equation.diff(rC2)
        derivative_r3 = inner_shape_function_equation.diff(rC3)

        H = [[derivative_r1, derivative_r1, derivative_r1],
             [derivative_r2, derivative_r2, derivative_r2],
             [derivative_r3, derivative_r3, derivative_r3]]

        r_relative = self.r_relative(r_C)

        for i in range(3):
            H[i][0] = sp.lambdify(rC1, H[i][0].diff(rC1))
            H[i][1] = sp.lambdify(rC2, H[i][1].diff(rC2))
            H[i][2] = sp.lambdify(rC3, H[i][2].diff(rC3))

            H[i][0] = H[i][0](r_relative[0][0])
            H[i][1] = H[i][1](r_relative[1][0])
            H[i][2] = H[i][2](r_relative[2][0])

        # 'wlasciwy' hessjan to macierz H

        return np.dot(np.dot(self.orientation, H), self.orientation.transpose())

    def nabla_2(self, r_C):

        inner_shape_func = self.inner_shape_function(r_C)[0]
        x = sp.Symbol('x')
        g_func = x ** (1 / self.p)
        g_func_prime = g_func.diff(x)
        g_func_double_prime = g_func_prime.diff(x)
        g_func_prime = sp.lambdify(x, g_func_prime)
        g_func_double_prime = sp.lambdify(x, g_func_double_prime)

        # nabla_2 = g_prime * hessian + g_double_prime * gradient * gradientT

        gradient = self.gradient(r_C)

        return np.dot(g_func_prime(inner_shape_func), self.hessian(r_C)) \
               + g_func_double_prime(inner_shape_func) * gradient * np.transpose(gradient)
