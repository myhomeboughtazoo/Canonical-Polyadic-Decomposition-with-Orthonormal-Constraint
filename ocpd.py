import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import polar
import copy

def T_circ_beta_l(T:np.ndarray, beta_l:list):
    axises = []
    for i in range(len(T.shape)):
        if i not in beta_l:
            axises.append(i)
    axises = axises + beta_l
    temp_T = T.transpose(axises)
    if len(beta_l) == 2:
        temp_T = temp_T.reshape(-1, temp_T.shape[-2], temp_T.shape[-1])
    elif len(beta_l) == 1:
        temp_T = temp_T.reshape(-1, temp_T.shape[-1])
    else:
        raise ValueError()
    return temp_T

def otimes_U(U, U_new, r, beta_l):
    res = np.array([1])
    s_1 = res.shape[0]
    s_2 = res.shape[0]
    for i in range(beta_l[0]):
        res = np.kron(res, U_new[i][r].reshape(-1))
        s_1 = res.shape[0]
    for i in range(beta_l[0]+len(beta_l), len(U)):
        res = np.kron(res, U[i][r].reshape(-1))
        s_2 = int(res.shape[0]/s_1)
    return res.reshape(s_1, s_2)

def U_initial(K, Miu, Rank, L_i):
    U = []
    for l in range(K):
        if l < K-Miu:
            u_i = np.random.randn(Rank, L_i[l])
            for i in range(u_i.shape[0]):
                u_i[i] /= np.linalg.norm(u_i[i])
        else:
            assert Rank <= L_i[l], 'rank must be samiler than the revelent dimension length'
            u_i = np.eye(Rank, L_i[l], dtype=np.float32)
        U.append(u_i)
    return U

def ocpd(T, Rank, Miu, P_max):
    L_i = T.shape
    K = len(L_i)
    U = U_initial(K, Miu, Rank, L_i)
    U_new = copy.deepcopy(U)
    lambda_lr = np.zeros([K, Rank])

    # 开始ocpd
    Tau = K - Miu - 1
    if (K - Miu)%2:     # 如果K-Miu为奇数
        Tau = K - Miu -2
    for p in range(P_max):
        for l in range(0, Tau, 2):
            beta_l = [l, l+1]
            T_beta_l = T_circ_beta_l(T, beta_l)
            T_beta_l = T_beta_l.reshape(-1, L_i[beta_l[0]]*L_i[beta_l[1]])
            for r in range(Rank):
                u_o = otimes_U(U, U_new, r, beta_l).reshape(-1)
                C_l_r_p = np.dot(u_o, T_beta_l).reshape(L_i[beta_l[0]], L_i[beta_l[1]])
                u, s, v = svds(C_l_r_p, k = 1)
                if u[0,0]<0:
                    u = -u
                    v = -v
                U_new[l][r] = u.reshape(-1)
                lambda_lr[l, r] = s

                U_new[l+1][r] = v.reshape(-1)
                lambda_lr[l+1, r] = s
                # if (K-Miu)%2==0:
                #     U_new[l+1][r] = v
                #     lambda_lr[l+1, r] = s
                # else:
                #     U_new[K-Miu-2][r] = v
                #     lambda_lr[K-Miu-2, r] = s
        if Tau == K - Miu - 2:
            beta_k_miu_1 = [K-Miu-2, K-Miu-1]
            T_beta_k_miu = T_circ_beta_l(T, beta_k_miu_1)
            T_beta_k_miu = T_beta_k_miu.reshape(-1, L_i[beta_k_miu_1[0]]*L_i[beta_k_miu_1[1]])
            for r in range(Rank):
                u_o = otimes_U(U, U_new, r, beta_k_miu_1).reshape(-1)
                C_k_miu_r_p = np.dot(u_o, T_beta_k_miu).reshape(L_i[beta_k_miu_1[0]], L_i[beta_k_miu_1[1]])
                u, s, v = svds(C_k_miu_r_p, 1)
                if u[0, 0]<0:
                    u = -u
                    v = -v
                U_new[K-Miu-2][r] = u.reshape(-1)
                U_new[K-Miu-1][r] = v.reshape(-1)
                lambda_lr[K-Miu-2, r] = s
                lambda_lr[K-Miu-1, r] = s
        for l in range(K-Miu, K):
            T_beta_l = T_circ_beta_l(T, [l])
            V = np.zeros([Rank, L_i[l]])
            lambda_hat = np.zeros([Rank, Rank])
            for r in range(Rank):
                u_o = otimes_U(U, U_new, r, [l]).reshape(-1)
                V[r] = np.dot(u_o, T_beta_l).reshape(-1)
                lambda_hat[r, r] = np.dot(V[r], U[l][r])
            temp_U, temp_S = polar(np.matmul(V, lambda_hat))
            U_new[l] = temp_U
            for r in range(Rank):
                lambda_lr[l, r] = temp_S[r, r]
        U = copy.deepcopy(U_new)
    
    vecs = []
    for r in range(Rank):
        vec = np.array([1])
        for l in range(K):
            vec = np.kron(vec, U_new[l][r].reshape(-1))
        vecs.append(vec)
    vec_T = T.reshape(-1)
    lambdas = np.zeros(Rank)
    for r in range(Rank):
        lambdas[r] = np.dot(vec_T, vecs[r])
    
    return U_new, vecs, lambdas
