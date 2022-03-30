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
        return np.dot(self.orientation.transpose(), np.subtract(r, self.r0))

    def inner_shape_function(self, r):
        r_relative = self.r_relative(r)
        return np.power((r_relative[0] / self.a), 2 * self.p) + np.power((r_relative[1] / self.b), 2 * self.p) \
                + np.power((r_relative[2] / self.c), 2 * self.p)

    def shape_function(self, r):
        return np.power(self.inner_shape_function(r), 1/self.p) - 1

    def gradient(self, r):

        r1 = sp.Symbol('r1')
        r2 = sp.Symbol('r2')
        r3 = sp.Symbol('r3')

        p = self.p

        inner_shape_function_equation = (r1 / self.a) ** (2 * p) + (r2 / self.b) ** (2 * p) + (r3 / self.c) ** (2 * p)

        derivative_1 = inner_shape_function_equation.diff(r1)
        derivative_2 = inner_shape_function_equation.diff(r2)
        derivative_3 = inner_shape_function_equation.diff(r3)

        derivative_1 = sp.lambdify(r1, derivative_1)
        derivative_2 = sp.lambdify(r2, derivative_2)
        derivative_3 = sp.lambdify(r3, derivative_3)

        r_relative = self.r_relative(r)

        # to jest 'wlasciwy' gradient
        grad = np.array([derivative_1(r_relative[0]), derivative_2(r_relative[1]), derivative_3(r_relative[2])])

        return np.dot(self.orientation, grad)

    def nabla(self, r):

        inner_shape_func = self.inner_shape_function(r)[0]
        x = sp.Symbol('x')
        g_func = x ** (1 / self.p)
        g_func_prime = g_func.diff(x)
        g_func_prime = sp.lambdify(x, g_func_prime)

        if inner_shape_func < 0:
            print('shape function returned negative number')
            return
        else:
            print('===')
            print('g\'')
            print(g_func_prime(inner_shape_func))
            print('===')
            return g_func_prime(inner_shape_func) * self.gradient(r)

    def hessian(self, r):

        r1 = sp.Symbol('r1')
        r2 = sp.Symbol('r2')
        r3 = sp.Symbol('r3')

        p = self.p

        inner_shape_function_equation = (r1 / self.a) ** (2 * p) + (r2 / self.b) ** (2 * p) + (r3 / self.c) ** (2 * p)

        derivative_r1 = inner_shape_function_equation.diff(r1)
        derivative_r2 = inner_shape_function_equation.diff(r2)
        derivative_r3 = inner_shape_function_equation.diff(r3)

        H = [[derivative_r1, derivative_r1, derivative_r1],
             [derivative_r2, derivative_r2, derivative_r2],
             [derivative_r3, derivative_r3, derivative_r3]]

        """
        # Sprawdzenie czy hessjan jest poprawny
        for i in range(3):
            H[i][0] = H[i][0].diff(r1)
            H[i][1] = H[i][1].diff(r2)
            H[i][2] = H[i][2].diff(r3)
        print(H)"""

        r_relative = self.r_relative(r)

        for i in range(3):
            H[i][0] = sp.lambdify(r1, H[i][0].diff(r1))
            H[i][1] = sp.lambdify(r2, H[i][1].diff(r2))
            H[i][2] = sp.lambdify(r3, H[i][2].diff(r3))

            H[i][0] = H[i][0](r_relative[0][0])
            H[i][1] = H[i][1](r_relative[1][0])
            H[i][2] = H[i][2](r_relative[2][0])

        return np.array(H)

    def nabla_2(self, r):

        inner_shape_func = self.inner_shape_function(r)[0]
        x = sp.Symbol('x')
        g_func = x ** (1 / self.p)
        g_func_prime = g_func.diff(x)
        g_func_double_prime = g_func_prime.diff(x)
        g_func_prime = sp.lambdify(x, g_func_prime)
        g_func_double_prime = sp.lambdify(x, g_func_double_prime)

        # nabla_2 = g_prime * hessian + g_double_prime * gradient * gradientT
        """
        print('g_prime * hessian:')
        print(np.dot(g_func_prime(inner_shape_func), self.hessian(r)))
        print('g_double_prime * gradient * gradientT')
        gradient = self.gradient(r)
        print(g_func_double_prime(inner_shape_func) * gradient * np.transpose(gradient))"""

        gradient = self.gradient(r)

        return np.dot(g_func_prime(inner_shape_func), self.hessian(r)) \
               + g_func_double_prime(inner_shape_func) * gradient * np.transpose(gradient)

    def overlap(self, another_superellipsoid):
        pass
        #    # tutaj bedzie magia
        #    return False
