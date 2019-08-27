#! /usr/bin/env python3

import torch
from torch.nn import Linear, Module
from torch.distributions import MultivariateNormal as MVN
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)

import numpy as np

def np_softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0)


def np_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def build_toy_dataset(N, K=None, D=None, pi=None):
    pi = np.array([0.4, 0.6])
    mus = [[2, 2], [-2, -2]]
    stds = [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]]
    x = np.zeros((N, 3), dtype=np.float32)
    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, 0:2] = np.random.multivariate_normal(mus[k], stds[k])
        x[n, 2] = k
    return x

sigmoid = torch.nn.Sigmoid()
softmax = torch.nn.Softmax(dim=0)
softplus = torch.nn.Softplus()

def make_scale_tril(gammas, dim, full_cov=True):
    if not(full_cov):
        return torch.diag_embed(gammas)
    d_vals = gammas[:, 0:dim]
    od_vals = gammas[:, dim:]
    d = d_vals.shape[-1]
    diags = torch.arange(d) + torch.cumsum(torch.arange(d), dim=0)
    tester = torch.zeros(d_vals.shape[0], d*(d+1)//2)
    tester[:, diags] = d_vals
    od0, od1 = (tester == 0).nonzero().transpose(0, 1)
    tester[od0, od1] = od_vals.flatten()
    i,j = torch.tril_indices(d,d)
    tril = torch.zeros(*d_vals.shape, d)
    tril[:, i, j] = tester
    return tril

def make_scale_tril_not_convoluted(diags, off_diags=None):
    if off_diags is None:
        return torch.diag_embed(diags)
    k = diags.shape[0]
    d = diags.shape[-1]
    d_idx0, d_idx1 = torch.cat(
        (torch.arange(d).unsqueeze(0),
         torch.arange(d).unsqueeze(0))
    )
    o_idx = torch.tril_indices(d, d)[0] != torch.tril_indices(d, d)[1]
    o_idx0, o_idx1 = torch.tril_indices(d, d)[:, o_idx]
    tril = torch.zeros(k, d, d)
    tril[:, d_idx0, d_idx1] = diags
    tril[:, o_idx0, o_idx1] = off_diags
    return tril



class GaussianMixtureModel(Module):
    def __init__(self, num_mixtures=2, num_dim=2, unity=True, full_cov=True, cuda_dev="cuda:0"):
        super().__init__()

        # model structure
        self.k = num_mixtures
        self.d = num_dim
        self.full_cov = full_cov
        self.unity = unity

        if unity:
            self.weights_bijector = softmax
        else:
            self.weights_bijector = sigmoid

        # initialize parameters
        self.weights = torch.nn.Parameter(
            torch.randn(self.k), requires_grad=True
        )
        self.mus = torch.nn.Parameter(
            torch.randn(self.k, self.d), requires_grad=True
        )
        self.diags = torch.nn.Parameter(
            torch.randn(self.k, self.d), requires_grad=True
        )

        if full_cov:
            self.off_diags = torch.nn.Parameter(
                torch.randn(
                    self.k, self.d * (self.d - 1) // 2
                ), requires_grad=True
            )

        use_cuda = torch.cuda.is_available()
        self.device = torch.device(cuda_dev if use_cuda else "cpu")

        if use_cuda:
            self.cuda()

    def forward(self, inputs):
        diags = softplus(self.diags)
        weights = self.weights_bijector(self.weights)
        chols = make_scale_tril_not_convoluted(diags, self.off_diags)
        probs = []
        for i in range(self.k):
            mvn = MVN(self.mus[i], scale_tril=chols[i])
            probs.append(weights[i] * mvn.log_prob(inputs).exp().unsqueeze(1))

        # mvns = MVN(self.mus, scale_tril=chols)
        # probs = mvns.log_prob(inputs).exp()
        # print(probs.shape)

        l2 = []
        covs = torch.bmm(chols, torch.transpose(chols, dim0=1, dim1=2))
        for i in range(self.k):
            for j in range(self.k):
                diff = MVN(
                    self.mus[i] - self.mus[j],
                    covariance_matrix=covs[i]+covs[j],
                    validate_args=True
                )
                p = diff.log_prob(torch.zeros(self.d)).exp() * weights[i] * weights[j]
                l2.append(p)

        return torch.Tensor(l2), torch.cat(probs, dim=1).sum(dim=1)


    def fit(self, inputs, nepochs=500):

        optimizer = torch.optim.Adam
        opt = optimizer(self.parameters(), lr=0.1)

        for i in range(nepochs):
            opt.zero_grad()
            outputs = self.forward(inputs)
            loss = outputs[0].sum() - 2 * outputs[1].mean()
            loss.backward()
            # if i % 10 == 0:
                # print(outputs[0].shape)
                # print(i, loss.item(), outputs[0].sum(), -2*outputs[1].mean(), outputs[1].shape)
            opt.step()
        print("fit finished")


np.random.seed(4)
data = build_toy_dataset(500)
GMM = GaussianMixtureModel()
print(np_softmax(GMM.weights.detach().numpy()))
c = torch.Tensor(data[:, 0:2])
GMM.fit(c)
print(np_softmax(GMM.weights.detach().numpy()))

