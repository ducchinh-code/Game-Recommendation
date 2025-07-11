import pandas as pd
import numpy as np
from cffi.backend_ctypes import xrange
from pywin.mfc.object import Object
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class CF(Object):
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        self.uuFC = uuCF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None

        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def add(self, new_data):
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)

    def normalize_Y(self):
        users = self.Y_data[:, 0]
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in xrange(self.n_users):
            ids = np.where(users==n)[0].astype(np.int32)
            items_ids = self.Y_data[ids,1]
            ratings = self.Y_data[ids,2]
            m = np.isnan(ratings)
            if np.isnan(m):
                m = 0

            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
                                       (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

    def refresh(self):
        self.normalize_Y()
        self.similarity()

    def fit(self):
        self.refresh()

    def _pred(self, u, i, normalized = 1):
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        sim = self.S[u,users_rated_i]
        a = np.argsort(sim)[-self.k:]
        nearest_s = sim[a]
        r = self.Ybar[i, users_rated_i[a]]

        if normalized:
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)

        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def pred(self, u , i, normalized = 1):

        if self.uuFC:
            return self._pred(u, i, normalized)
        return self._pred(i, u, normalized)

    def recommend(self, u ,normalized = 1):
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids,1].tolist()
        recomment_items = []
        for i in xrange(self.n_items):
            if i not in items_rated_by_u:
                rating = self._pred(u,i)
                if rating > 0:
                    recomment_items.append(i)
        return recomment_items

    def print_recommendation(self):
        print
        'Recommendation: '
        for u in xrange(self.n_users):
            recommended_items = self.recommend(u)
            if self.uuCF:
                print('Recommend item(s):', recommended_items, 'to user', u)
            else:
                print('Recommend item', u, 'to user(s) : ', recommended_items)




