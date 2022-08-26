import oneflow as flow
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import warnings
from oneflow.optim import LBFGS

warnings.filterwarnings('ignore')

np.random.seed(1234)

# CUDA support
if flow.cuda.is_available():
    device = flow.device('cuda')
else:
    device = flow.device('cpu')


# the deep neural network
class DNN(flow.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = flow.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, flow.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), flow.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = flow.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X, u, layers, lb, ub):

        # boundary conditions
        self.lb = flow.tensor(lb).float().to(device)
        self.ub = flow.tensor(ub).float().to(device)

        # data
        self.x = flow.tensor(X[:, 0:1], requires_grad=True).float().to(device)

        self.t = flow.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.u = flow.tensor(u).float().to(device)

        # settings
        self.lambda_1 = flow.tensor([0.0], requires_grad=True).to(device)
        self.lambda_2 = flow.tensor([-6.0], requires_grad=True).to(device)

        self.lambda_1 = flow.nn.Parameter(self.lambda_1)
        self.lambda_2 = flow.nn.Parameter(self.lambda_2)

        # deep neural networks
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lambda_1', self.lambda_1)
        self.dnn.register_parameter('lambda_2', self.lambda_2)

        # optimizers: using the same settings
        self.optimizer = LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )

        self.optimizer_Adam = flow.optim.Adam(self.dnn.parameters())
        self.iter = 0

    def net_u(self, x, t):
        u = self.dnn(flow.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        lambda_1 = self.lambda_1
        lambda_2 = flow.exp(self.lambda_2)
        u = self.net_u(x, t)

        u_t = flow.autograd.grad(
            u, t,
            grad_outputs=flow.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = flow.autograd.grad(
            u, x,
            grad_outputs=flow.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = flow.autograd.grad(
            u_x, x,
            grad_outputs=flow.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = u_xx.to(device)
        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        return f

    def loss_func(self):
        u_pred = self.net_u(self.x, self.t)

        f_pred = self.net_f(self.x, self.t)

        loss = flow.mean((self.u - u_pred) ** 2) + flow.mean(f_pred ** 2)

        self.optimizer.zero_grad()
        loss.backward()

        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Loss: %e, l1: %.5f, l2: %.5f' %
                (
                    loss.item(),
                    self.lambda_1.item(),
                    flow.exp(self.lambda_2.detach()).item()
                ), flush=True
            )
        return loss

    def train(self, nIter):
        self.dnn.train()

        for epoch in range(nIter):
            u_pred = self.net_u(self.x, self.t)
            f_pred = self.net_f(self.x, self.t)
            loss = flow.mean((self.u - u_pred) ** 2) + flow.mean(f_pred ** 2)
            loss.backward()

            if epoch % 100 == 0:
                print(
                    'It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f' %
                    (
                        epoch,
                        loss.item(),
                        self.lambda_1.item(),
                        flow.exp(self.lambda_2).item()
                    ), flush=True
                )

        # Backward and optimize
        self.optimizer.step(self.loss_func)

    def predict(self, X):
        x = flow.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = flow.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()

        u = self.net_u(x, t)
        f = self.net_f(x, t)
        x = x.detach().cpu().numpy()
        x = x.reshape(25600)
        t = t.detach().cpu().numpy()
        t = t.reshape(25600)
        u = u.detach().cpu().numpy()
        u = u.reshape(25600)
        f = f.detach().cpu().numpy()
        plt.figure()
        ax = plt.gca(projection='3d')
        ax.plot_trisurf(x, t, u, linewidth=0.3, antialiased=True, cmap='rainbow', alpha=0.8)
        plt.show()
        return u, f


# Configurations
nu = 0.01 / np.pi

N_u = 2000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

data = scipy.io.loadmat('data/burgers_shock.mat')

t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)

# Training on Non-noisy Data
noise = 0.0

# create training set
idx = np.random.choice(X_star.shape[0], N_u, replace=False)
X_u_train = X_star[idx, :]
u_train = u_star[idx, :]

# training
model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
model.train(0)

# evaluations
u_pred, f_pred = model.predict(X_star)
