import numpy as np


def matrix_M(lbd, sp_A_nabla_2, sp_B_nabla_2):
    return lbd * sp_A_nabla_2 + (1 - lbd) * sp_B_nabla_2


def delta_g(sp_A_nabla, sp_B_nabla):
    return sp_A_nabla - sp_B_nabla


def zeta_lbd_lbd(delta_g, matrix_M):
    delta_g_T = np.transpose(delta_g)
    matrix_M_inv = np.linalg.inv(matrix_M)
    return np.dot(np.dot(delta_g_T, matrix_M_inv), delta_g)


def nabla_of_both(lbd, sp_A_nabla, sp_B_nabla):
    return lbd * sp_A_nabla + (1 - lbd) * sp_B_nabla


def delta_lambda(zeta_lbd_lbd, sp_A_shape_func, sp_B_shape_func, delta_g, matrix_M, nabla_of_both):

    # (-1)/zeta_lbd_lbd * [(A.shape_func - B.shape_func) - delta_g^T * M^(-1) * nabla_of_both]

    delta_g_T = np.transpose(delta_g)
    matrix_M_inv = np.linalg.inv(matrix_M)

    delta_lambda = sp_B_shape_func - sp_A_shape_func
    delta_lambda = delta_lambda - np.dot(np.dot(delta_g_T, matrix_M_inv), nabla_of_both)
    delta_lambda = (-1)/zeta_lbd_lbd * delta_lambda

    return delta_lambda


def delta_r_C(matrix_M, delta_g, delta_lambda, nabla_of_both):

    # M^(-1) * (delta_g * delta_lambda - nabla_of_both)

    matrix_M_inv = np.linalg.inv(matrix_M)

    delta_r_C = delta_g * delta_lambda - nabla_of_both
    delta_r_C = np.dot(matrix_M_inv, delta_r_C)

    return delta_r_C


def overlap(sp_A, sp_B):

    lbd = 0.5
    r_C = (sp_A.r0 + sp_B.r0) / 2

    dlt_lbd = 0
    dlt_r_C = np.array([[0], [0], [0]])

    num_of_iterations = 10

    for n in range(num_of_iterations):

        lbd = lbd - dlt_lbd
        r_C = r_C + dlt_r_C

        print("------------", end="")
        print(n, end="")
        print("------------")
        print("lbd:")
        print(lbd)
        print("r_C:")
        print(r_C)

        nabla_A = sp_A.nabla(r_C)
        nabla_B = sp_B.nabla(r_C)

        dlt_g = delta_g(nabla_A, nabla_B)
        M = matrix_M(lbd, sp_A.nabla_2(r_C), sp_B.nabla_2(r_C))
        zeta = zeta_lbd_lbd(dlt_g, M)
        nob = nabla_of_both(lbd, nabla_A, nabla_B)

        dlt_lbd = delta_lambda(zeta, sp_A.shape_function(r_C), sp_B.shape_function(r_C), dlt_g, M, nob)
        dlt_r_C = delta_r_C(M, dlt_g, dlt_lbd, nob)

    pw_potential = lbd * sp_A.shape_function(r_C) + (1-lbd) * sp_B.shape_function(r_C)

    return pw_potential
