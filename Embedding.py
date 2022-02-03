import torch
import torch.nn as nn
import pickle

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class Node_feature(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        # m1_1 means matrix 1 of layer 1
        # layer 1 - node feature/self feature

        # initialize with standard normal distribution
        self.m1_1 = nn.Parameter(torch.randn(k, d)).to(device)

    def forward(self, x_c):
        # x_c is a d-dimensional vector [usually initialized using Tucker embedding algorithm]
        # m1_1 @ x_c returns a vector of size k (since x_c is a vector of size d)
        # return the sum of all elements (a scalar)
        return (self.m1_1 @ x_c).sum().item()


class Pair_feature(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        # m1_2 means matrix 1 of layer 2, m2_2 means matrix 2 of layer 2
        # layer 2- pair feature
        self.m1_2 = nn.Parameter(torch.randn(k, d)).to(device)
        self.m2_2 = nn.Parameter(torch.randn(k, d)).to(device)

    def forward(self, x_c, x_t):
        # calculate matrix multiplications between the following pairs: (m1_2, x_c) and (m2_2 @ x_t)
        # these matmul operations return two vectors, each of size k
        # return the dot product of these two vectors (a scalar)
        return (self.m1_2 @ x_c) @ (self.m2_2 @ x_t)


class Triplet_feature(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        # m1_3 means matrix 1 of layer 3
        # layer 3 - triplet feature
        self.m1_3 = nn.Parameter(torch.randn(k, d)).to(device)
        self.m2_3 = nn.Parameter(torch.randn(k, d)).to(device)  # matrix 2 of layer 3
        self.m3_3 = nn.Parameter(torch.randn(k, d)).to(device)  # matrix 3 of layer 3

    def forward(self, x_c, x_t, x_i):
        # calculate matrix multiplications of the following pairs: (m1_3, x_c), (m2_3, x_t), (m3_3, x_i)
        # each matmul returns a vector of size k (let's call them A, B, C respectively)

        # option a: calculate an outer product of A and B (shape (k, k)) and then calculate a dot product of the outer product and C
        # a vector of size k would be created; return the sum of the elements of the vector (a scalar)
        # return (torch.outer((self.m1_3 @ x_c), (self.m2_3 @ x_t)) @ (self.m3_3 @ x_i)).sum()

        # option b: calculate an element wise product of A, B, C and return the sum
        return ((self.m1_3 @ x_c) * (self.m2_3 @ x_t) * (self.m3_3 @ x_i)).sum()


class Gibbs_dist(nn.Module):
    def __init__(d, k, L_init):
        super().__init__()
        # L_init is the initial lookup table
        # it is an embedding calculated using tucker model
        E1 = Node_feature(d, k)
        E2 = Pair_feature(d, k)
        E3 = Triplet_feature(d, k)
        L = nn.Parameter(L_init).to(device)  # should be a parameter

    def forward(idx_c, idx_t, idx_intermediate):
        x_c = L(idx_c)
        x_t = L(idx_t)

        energy = E1(x_c) + E2(x_c, x_t)

        for idx in idx_intermediate:
            # x_i = L(idx)
            energy += E3(x_c, x_t, L(idx))
        return torch.sigmoid(energy)