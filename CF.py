import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


class CF:
    def __init__(self, Y_data, k=2, dist_func=cosine_similarity, uuCF=1):
        self.uuCF = uuCF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None

        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def add(self, new_data):
        self.Y_data = np.concatenate((self.Y_data, new_data), axis=0)

    def normalize_Y(self):
        users = self.Y_data[:, 0]
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))

        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)
            user_ratings = self.Y_data[ids, 2]

            if len(user_ratings) == 0:
                m = 0
            else:
                m = np.mean(user_ratings)

            self.mu[n] = m
            self.Ybar_data[ids, 2] = user_ratings - self.mu[n]

        self.Ybar = sparse.coo_matrix(
            (self.Ybar_data[:, 2], (self.Ybar_data[:, 1], self.Ybar_data[:, 0])),
            shape=(self.n_items, self.n_users)
        ).tocsr()

    def similarity(self):
        self.S = cosine_similarity(self.Ybar.T, dense_output=False)

    def refresh(self):
        self.normalize_Y()
        self.similarity()

    def fit(self):
        self.refresh()

    def _pred(self, u, i, normalized=1):
        # Kiểm tra bounds
        if u >= self.n_users or i >= self.n_items:
            return 0.0

        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)

        if len(ids) == 0:
            # Không có user nào đánh giá item này
            return 0.0

        users_rated_i = self.Y_data[ids, 0].astype(np.int32)

        # Lấy similarity scores
        sim = self.S[u, users_rated_i]

        # Chuyển đổi sparse matrix thành dense array nếu cần
        if hasattr(sim, 'toarray'):
            sim = sim.toarray().flatten()
        elif hasattr(sim, 'A1'):
            sim = sim.A1  # Cho sparse matrix
        else:
            sim = np.array(sim).flatten()

        # Tìm k nearest neighbors
        if len(sim) == 0:
            return 0.0

        k = min(self.k, len(sim))
        a = np.argsort(sim)[-k:]
        nearest_s = sim[a]

        # Lấy ratings
        r = self.Ybar[i, users_rated_i[a]]
        if hasattr(r, 'toarray'):
            r = r.toarray().flatten()
        elif hasattr(r, 'A1'):
            r = r.A1
        else:
            r = np.array(r).flatten()

        # Tính toán prediction
        sum_sim = np.abs(nearest_s).sum()
        if sum_sim == 0:
            return 0.0

        pred_score = (r @ nearest_s) / (sum_sim + 1e-8)

        if normalized:
            return float(pred_score)
        else:
            # Lấy mean rating của user u
            mu_u = self.mu[u]
            # Đảm bảo mu_u là scalar
            if isinstance(mu_u, np.ndarray):
                mu_u = mu_u.item() if mu_u.size == 1 else mu_u[0]

            return float(pred_score + mu_u)

    def pred(self, u, i, normalized=1):
        if self.uuCF:
            return self._pred(u, i, normalized)
        return self._pred(i, u, normalized)

    def recommend(self, u, normalized=1):
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommended_items = []

        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self._pred(u, i, normalized)

                # Đảm bảo rating là scalar
                if isinstance(rating, np.ndarray):
                    rating = rating.item() if rating.size == 1 else rating[0]

                if rating > 0:
                    recommended_items.append((i, float(rating)))

        recommended_items.sort(key=lambda x: -x[1])
        return recommended_items

    def print_recommendation(self):
        print('Recommendations:')
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            if self.uuCF:
                print(f'Recommend item(s): {recommended_items} to user {u}')
            else:
                print(f'Recommend item {u} to user(s): {recommended_items}')