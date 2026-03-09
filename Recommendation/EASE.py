import numpy as np

class EASE:
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.B = None

    def fit(self, X):
        # prepare optimization
        I_ids = np.arange(0, X.shape[1], 1, dtype=np.int64)
        I = np.eye(X.shape[1])
        
        # G : gram matrix
        G = X.T@X
        
        # P : precision matrix
        P = np.linalg.inv(G + self.lambda_ * I)

        # if i != j,
        self.B = - P / np.diag(P)
        # if i == j,
        self.B[I_ids, I_ids] = 0


    def forward(self, X):
        S = X@self.B
        return S

class weightedEASE:
    def __init__(self, alpha, lambda_):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.B = None
        self.C = None

    def fit(self, X):
        # prepare optimization
        I_ids = np.arange(0, X.shape[1], 1, dtype=np.int64)
        I = np.eye(X.shape[1])

        # G : gram matrix
        G = X.T@X

        # pop : item count
        # C : popularity-based confidence 
        pop = np.sum(X, axis=0).astype(G.dtype)
        self.C = pop[:, None] + pop[None, :]
        self.C = 1 + self.alpha * np.log1p(self.C)

        # update G with C
        G = self.C * G
        
        # P : precision matrix
        P = np.linalg.inv(G + self.lambda_ * I)

        # if i != j,
        self.B = - P / np.diag(P)
        # if i == j,
        self.B[I_ids, I_ids] = 0


    def forward(self, X):
        S = X@self.B
        return S

    def fowrard_with_C(self, X):
        S = X@(C*self.B)
        return S