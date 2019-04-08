from sklearn.mixture import GaussianMixture
import numpy as np
import pdb
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
from scipy.stats import multivariate_normal

def main():

    path = 'X_new.txt'
    X = np.loadtxt(path)
    N = X.shape[0]
    gmm = GaussianMixture(n_components=3, covariance_type='spherical', tol=0.001, reg_covar=1e-06, max_iter=200, n_init=10,init_params='random')

    gmm.fit(X)
    print (gmm.get_params())
    print (gmm.means_)
    print (gmm.covariances_)

    labels = gmm.predict(X)
    mu = gmm.means_

    sorted_gs = np.argsort(gmm.means_[:, 0])
    sorted_colors = ['b','g','r']
    color_dict ={}
    for i,g in enumerate(sorted_gs):
        color_dict[g] = sorted_colors[i]
    c = np.array([color_dict[i] for i in labels])


    plt.scatter(X[:,0],X[:,1],c = c)
    plt.scatter(mu[:,0],mu[:,1],c='k')
    plt.show()


    sorted_gs = np.argsort(gmm.means_[:, 2])
    color_dict ={}
    for i,g in enumerate(sorted_gs):
        color_dict[g] = sorted_colors[i]
    c = np.array([color_dict[i] for i in labels])


    plt.scatter(X[:,2],X[:,3],c = c)
    plt.scatter(mu[:, 2], mu[:, 3],c= 'k')
    plt.show()

    sorted_gs = np.argsort(gmm.means_[:, 3])
    color_dict ={}
    for i,g in enumerate(sorted_gs):
        color_dict[g] = sorted_colors[i]
    c = np.array([color_dict[i] for i in labels])


    plt.scatter(X[:,3],X[:,4],c = c)
    plt.scatter(mu[:, 3], mu[:, 4],c= 'k')
    # plt.clf()
    plt.show()
    # pdb.set_trace()

    # mus = np.random.rand(3)
    stds = np.sqrt(gmm.covariances_.copy())
    pis = gmm.weights_.copy()
    exps = np.zeros([N,3])

    iter_max =150


    ##EM algorithm without library functions##
    ##initialize means##
    init_inds= np.random.randint(N, size=3)
    mus = X[init_inds]
    print ("init mus:",mus)

    covs = [np.identity(5)*(stds[i]**2) for i in range(3)]
    pis = np.random.rand(3)
    pis = pis/sum(pis)
    new_mus = np.zeros_like(mus)
    tol = 0.001


    for i in range(iter_max):
        n = [multivariate_normal(mus[j],covs[j]) for j in range(3)]

        for j in range(N):
            for k in range(3):

                exps[j,k] = pis[k]* (n[k].pdf(X[j]) + 1e-15)
            # print (exps[j,:])
            exps[j,:] = exps[j,:]/np.sum(exps[j,:])


        ##MAXIMIZE!!!##
        for k in range(3):
            pis[k] = np.sum(exps[:,k])/N
            new_mus[k] = np.sum(exps[:,k].reshape(-1,1)*X,axis =0) / np.sum(exps[:,k])  #exps = Nxk , X = nxd , mus = d

        if(np.linalg.norm(new_mus - mus) < tol):
            print ("Breaking at iter ",i)
            break
        else:
            mus = new_mus.copy()


    print ("post optimization:")
    print (mus)
    print (covs)
    print (pis)
    gmm.means_ = np.array(mus)

    labels = gmm.predict(X)
    # mu = gmm.means_

    sorted_gs = np.argsort(gmm.means_[:, 0])
    sorted_colors = ['b','g','r']
    color_dict ={}
    for i,g in enumerate(sorted_gs):
        color_dict[g] = sorted_colors[i]
    c = np.array([color_dict[i] for i in labels])


    plt.scatter(X[:,0],X[:,1],c = c)
    plt.scatter(mus[:,0],mus[:,1],c='k')
    plt.show()



















if __name__ == '__main__':
    main()