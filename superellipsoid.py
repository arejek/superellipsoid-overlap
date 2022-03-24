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

    def overlap(self, another_superellipsoid):
        pass
        #    # tutaj bedzie magia
        #    return False


s1 = Superellipsoid(2, 2, 2, 3, np.array([[0], [0], [0]]),
                    np.array([[0.731, -0.682, 0.000], [0.682, 0.731, 0.000], [0.000, 0.000, 1.000]]))

print(s1.shape_function(np.array([[1], [1], [1]])))
print(s1.orientation)

print('r z wezykiem:')
print(s1.r_relative(np.array([[1], [1], [1]])))

print('gradient:')
print(s1.gradient(np.array([[1], [1], [1]])))

print('---')
print(s1.gradient(np.array([[1], [1], [1]])))
print('---')

s2 = Superellipsoid(2, 2, 2, 3, np.array([[-3.456], [9.345], [-8.456]]),
                    np.array([[0.993, 0.000, 0.122], [0.066, 0.839, -0.541], [-0.102, 0.545, 0.832]]))

print(s2.r_relative(np.array([[0.999], [0], [-2.345]])))

s3 = Superellipsoid(10, 10, 10, 3, np.array([[0], [0], [0]]),
                    np.array([[0.731, -0.682, 0.000], [0.682, 0.731, 0.000], [0.000, 0.000, 1.000]]))

print('Inny wektor')
print(s1.inner_shape_function(np.array([[1.8], [1.8], [1.8]])))
print(s1.gradient(np.array([[1.8], [1.8], [1.8]])))
print(s1.nabla(np.array([[1.8], [1.8], [1.8]])))
