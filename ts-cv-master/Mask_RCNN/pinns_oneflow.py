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
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()

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

error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

lambda_1_value = model.lambda_1.detach().cpu().numpy()
lambda_2_value = model.lambda_2.detach().cpu().numpy()
lambda_2_value = np.exp(lambda_2_value)

error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
error_lambda_2 = np.abs(lambda_2_value - nu) / nu * 100

print('Error u: %e' % (error_u))
print('Error l1: %.5f%%' % (error_lambda_1))
print('Error l2: %.5f%%' % (error_lambda_2))
# print('done')
####### Row 0: u(t,x) ##################

fig, ax = plt.subplots(figsize=(9, 5))
h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min().item(), t.max().item(), x.min().item(), x.max().item()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=15)

ax.plot(
    X_u_train[:, 1],
    X_u_train[:, 0],
    'kx', label='Data (%d points)' % (u_train.shape[0]),
    markersize=4,  # marker size doubled
    clip_on=False,
    alpha=.5
)

line = np.linspace(x.min().item(), x.max().item(), 2)[:, None]
ax.plot(t[25].item() * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[50].item() * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[75].item() * np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.set_title('$u(t,x)$', fontsize=20)  # font size doubled
ax.tick_params(labelsize=15)
print('done')
plt.show()
###### Row 1: u(t,x) slices ##################

""" The aesthetic setting has changed. """

fig, ax = plt.subplots(figsize=(14, 10))

gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = 0.25$', fontsize=15)
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = 0.50$', fontsize=15)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_title('$t = 0.75$', fontsize=15)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.show()
