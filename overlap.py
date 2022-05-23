import numpy as np


def matrix_M(lbd, this_sp_nabla_2, other_sp_nabla_2):
    return lbd * this_sp_nabla_2 + (1 - lbd) * other_sp_nabla_2


def delta_g(this_sp_nabla, other_sp_nabla):
    return this_sp_nabla - other_sp_nabla


def zeta_lbd_lbd(delta_g, matrix_M):
    delta_g_T = np.transpose(delta_g)
    matrix_M_inv = np.linalg.inv(matrix_M)
    return np.dot(np.dot(delta_g_T, matrix_M_inv), delta_g)


def nabla_of_both(lbd, this_sp_nabla, other_sp_nabla):
    return lbd * this_sp_nabla + (1 - lbd) * other_sp_nabla


def delta_lambda(zeta_lbd_lbd, this_sp_shape_func, other_sp_shape_func, delta_g, matrix_M, nabla_of_both):

    # (-1)/zeta_lbd_lbd * [(A.shape_func - B.shape_func) - delta_g^T * M^(-1) * nabla_of_both]

    delta_g_T = np.transpose(delta_g)
    matrix_M_inv = np.linalg.inv(matrix_M)

    delta_lambda = other_sp_shape_func - this_sp_shape_func
    delta_lambda = delta_lambda - np.dot(np.dot(delta_g_T, matrix_M_inv), nabla_of_both)
    delta_lambda = (-1)/zeta_lbd_lbd * delta_lambda

    return delta_lambda


def delta_r(matrix_M, delta_g, delta_lambda, nabla_of_both):

    # M^(-1) * (delta_g * delta_lambda - nabla_of_both)

    matrix_M_inv = np.linalg.inv(matrix_M)

    delta_r = delta_g * delta_lambda - nabla_of_both
    delta_r = np.dot(matrix_M_inv, delta_r)

    return delta_r


def overlap():
    pass