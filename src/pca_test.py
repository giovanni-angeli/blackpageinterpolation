

import random
import time

import numpy as np
from sklearn.decomposition import PCA


# ~ sites = [
    # ~ [0.0, 0.0, 0.0], 
    # ~ [0.0, 0.13541666666666666, 0.14130434782608695], 
    # ~ [0.3382352941176471, 0.4270833333333333, 0.3804347826086957], 
    # ~ [0.022058823529411766, 0.09375, 0.13043478260869565], 
    # ~ [0.16911764705882354, 0.3854166666666667, 0.45652173913043476], 
    # ~ [0.22794117647058823, 0.4270833333333333, 0.5], 
    # ~ [0.16176470588235295, 0.3229166666666667, 0.31521739130434784], 
    # ~ [0.47794117647058826, 0.3854166666666667, 0.17391304347826086], 
    # ~ [0.3897058823529412, 0.14583333333333334, 0.05434782608695652], 
    # ~ [0.5, 0.5, 0.3804347826086957]
# ~ ]

# ~ CENTER = np.array([.5, .5, .5])
CENTER = np.array([.0, .0, .0])


def rnd(factor):
    return factor * (random.random() - 0.5)

def generate_cloud_3(a, b, c, N=100):

    sites = []
    v_2 = np.array([a ** 2, b ** 2, c ** 2])
    for i in range(2 * N):
        x = np.array([rnd(2 * a), rnd(2 * b), rnd(2 * c)])
        s = np.sum(x / v_2)
        if s < 1:
            sites.append(CENTER + x)

            if len(sites) >= N:
                break

    sites = np.array(sites)
    return sites

def generate_cloud_2():

    sites = [ CENTER +  np.array([rnd(1), rnd(1), rnd(10),]) for i in range(100)]
    sites = np.array(sites)
    return sites

def generate_cloud_1():

    sites = [
        [0.5, 0.5, 0.5], 
        [0.0, 0.0, 0.0], 
        [1, 0, 0], 
        [0, 1, 0], 
        [0, 0, 1], 
        [0, 1, 1], 
        [1, 0, 1], 
        [1, 1, 0], 
        [1, 1, 1], 
    ]

    sites = 0.25 + .5 * np.array(sites)
    return sites

def is_inside(p, cloud):

    t0 = time.time()
    cntrs = [0, 0]
    for q in cloud:
        p_q = (p - q)
        p_q_2 = np.dot(p_q, p_q)
        cntr = 0
        for x in cloud:
            if not x is q:
                d = np.dot((x - q), p_q) - p_q_2
                if d >= 0:
                    cntrs[0] += 1
                else:
                    cntrs[1] += 1
    dt = time.time() - t0
    r = cntrs[0] / max(1, cntrs[1])
    print (f"p:{p}, r:{r:.3f}, cntrs:{cntrs}, dt:{dt}")
    return r

def one():

    pca = PCA(n_components=3)

    # ~ sites = generate_cloud_1()
    sites = generate_cloud_2()

    # ~ print(f"sites:{sites}")
    pca.fit(sites)
    # ~ print(dir(pca))
    print(f"pca.components_:{pca.components_}")
    print(f"pca.explained_variance_:{pca.explained_variance_}")
    # ~ print(f"sites:{sites}, sites_:{pca.transform(sites)}")

    for test_site in [
            CENTER ,
            CENTER + [10, 1., 5.],
            CENTER + [.1, 1., 5.],
            CENTER + [.1, 1., 10.],
            CENTER + [.1, 1., 20.],
            CENTER + [.1, 5., 20.],
            CENTER + [1., 5., 20.],
            CENTER + [1., 0., 0.],
        ]:
        test_site = np.matrix(test_site)
        test_site_ = pca.transform(test_site)
        print(f"test_site:{test_site}, test_site_:{test_site_}")

def two():

    # ~ sites = generate_cloud_1()
    # ~ cloud = generate_cloud_2()
    cloud = generate_cloud_3(20, 20, 10, 10)

    test_site = [CENTER + [0, 0, i] for i in range(0, 20, 2)]

    for p in test_site:
        is_inside(p, cloud)

two()
