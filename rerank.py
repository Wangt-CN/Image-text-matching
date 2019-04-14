import numpy as np
import utils


def i2t_rerank(sim, K1, K2):

    size_i = sim.shape[0]
    size_t = sim.shape[1]
    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)
    sort_i2t_re = np.copy(sort_i2t)[:, :K1]
    address = np.array([])

    for i in range(size_i):
        for j in range(K1):
            result_t = sort_i2t[i][j]
            query = sort_t2i[:, result_t]
            # query = sort_t2i[:K2, result_t]
            address = np.append(address, np.where(query == i)[0][0])

        sort = np.argsort(address)
        sort_i2t_re[i] = sort_i2t_re[i][sort]
        address = np.array([])

    sort_i2t[:,:K1] = sort_i2t_re

    return sort_i2t


def i2t_rerank_gai(sim, K1, K2, K3):

    size_i = sim.shape[0]
    size_t = sim.shape[1]
    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)
    sort_i2t_re = np.copy(sort_i2t)[:, :K1]
    address = np.array([])

    for i in range(size_i):
        for j in range(K1):
            result_t = sort_i2t[i][j]
            query = sort_t2i[:, result_t]
            # query = sort_t2i[:K2, result_t]
            address = np.append(address, np.where(query == i)[0][0])

        if address[0] == address[1] == 0:

            new_query1 = sort_i2t[sort_t2i[1, sort_i2t[i][0]], :K3]
            new_query2 = sort_i2t[sort_t2i[1, sort_i2t[i][1]], :K3]
            sim1, sim2 = sparsity_sentence_sim(sort_i2t[i, :K3], new_query1, new_query2)
            if sim1 < sim2:
                address[0] += 0.1

        if address[0] == address[1] == 1:

            new_query1 = sort_i2t[sort_t2i[0, sort_i2t[i][0]], :K3]
            new_query2 = sort_i2t[sort_t2i[0, sort_i2t[i][1]], :K3]
            sim1, sim2 = sparsity_sentence_sim(sort_i2t[i, :K3], new_query1, new_query2)
            if sim1 < sim2:
                address[0] += 0.1

        sort = np.argsort(address)
        sort_i2t_re[i] = sort_i2t_re[i][sort]
        address = np.array([])

    sort_i2t[:,:K1] = sort_i2t_re

    return sort_i2t


def sparsity_sentence_sim(anchor, s1, s2):
    # anchor = anchor[:k]
    # s1 = s1[:k]
    # s2 = s2[:k]
    sum1 = 0
    sum2 = 0
    s1 = s1.tolist()
    s2 = s2.tolist()
    for i in range(anchor.shape[0]):
        if anchor[i] in s1:
            sum1 += 1
        elif anchor[i] in s2:
            sum2 += 1
    return sum1, sum2


def t2i_rerank(sim, K1, K2):

    size_i = sim.shape[0]
    size_t = sim.shape[1]
    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)
    sort_t2i_re = np.copy(sort_t2i)[:K1, :]
    address = np.array([])

    for i in range(size_t):
        for j in range(K1):
            result_i = sort_t2i[j][i]
            query = sort_i2t[result_i, :]
            # query = sort_t2i[:K2, result_t]
            address = np.append(address, np.where(query == i)[0][0])

        sort = np.argsort(address)
        sort_t2i_re[:, i] = sort_t2i_re[:, i][sort]
        address = np.array([])

    sort_t2i[:K1, :] = sort_t2i_re

    return sort_t2i


def t2i_rerank_gai(sim, s_sort, K1, K2, K_near):

    size_i = sim.shape[0]
    size_t = sim.shape[1]
    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)
    sort_t2i_re = np.copy(sort_t2i)[:K1, :]
    address = np.array([])

    for i in range(size_t):
        for j in range(K1):
            result_i = sort_t2i[j][i]
            k_near = s_sort[i][:K_near].tolist()
            query = sort_i2t[result_i, :]
            # query = sort_t2i[:K2, result_t]
            ranks = 1e20
            for k in k_near:
                tmp = np.where(query == k)[0][0]
                if tmp < ranks:
                    ranks = tmp
            address = np.append(address, ranks)

        sort = np.argsort(address)
        sort_t2i_re[:, i] = sort_t2i_re[:, i][sort]
        address = np.array([])

    sort_t2i[:K1, :] = sort_t2i_re

    return sort_t2i

def t2i_rerank1(sim, K1, K2):

    size_i = sim.shape[0]
    size_t = sim.shape[1]
    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)
    sort_t2i_re = np.copy(sort_t2i)[:K1, :]
    address = np.array([])

    for i in range(size_t):
        for j in range(K1):
            result_i = sort_t2i[j][i]
            query = sort_i2t[result_i, :]
            # query = sort_t2i[:K2, result_t]
            ranks = 1e20
            for k in range(5):
                tmp = np.where(query == i//5 * 5 + k)[0][0]
                if tmp < ranks:
                    ranks = tmp
            address = np.append(address, ranks)

        sort = np.argsort(address)
        sort_t2i_re[:, i] = sort_t2i_re[:, i][sort]
        address = np.array([])

    sort_t2i[:K1, :] = sort_t2i_re

    return sort_t2i


def acc_i2t2(input):
    """Computes the precision@k for the specified values of k of i2t"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = input[index]
        # Score
        if index == 197:
            print('s')
        rank = 1e20
        for i in range(5 * index, min(5 * index + 5, image_size*5), 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i2(input):
    """Computes the precision@k for the specified values of k of t2i"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)

    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = input[5 * index + i]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


d = np.load('stage_1_test.npy')
s_sort = np.load('sentence_sim_test.npy')

# sort_rerank = i2t_rerank_gai(d, s_sort, 6, 1, 2)
sort_rerank = i2t_rerank(d, 6, 1)
a = np.argsort(-d, 0)

(r1i, r5i, r10i, medri, meanri), _ = acc_i2t2(np.argsort(-d, 1))
(r1i2, r5i2, r10i2, medri2, meanri2), _ = acc_i2t2(sort_rerank)

print(r1i, r5i, r10i, medri, meanri)
print(r1i2, r5i2, r10i2, medri2, meanri2)

# sort_rerank = t2i_rerank_gai(d, s_sort, 7, 1, 3)
# # sort_rerank = t2i_rerank1(d, 20, 1)
# (r1t, r5t, r10t, medrt, meanrt), _ = acc_t2i2(np.argsort(-d, 0))
# (r1t2, r5t2, r10t2, medrt2, meanrt2), _ = acc_t2i2(sort_rerank)
#
# print(r1t, r5t, r10t, medrt, meanrt)
# print(r1t2, r5t2, r10t2, medrt2, meanrt2)
