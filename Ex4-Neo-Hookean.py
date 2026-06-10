# Prepare with libraries and device configuration

import os
import re

os.environ["DDEBACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
import time
from deepxde.backend import torch
from deepxde.nn import activations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dde.config.set_random_seed(2024)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Define a custom multi-output physics-informed neural network (MPFNN) with two branches for displacements and stresses

class MPFNN(dde.nn.NN):
    def __init__(self, first_layer_sizes, second_layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.first_layer_sizes = first_layer_sizes
        self.second_layer_sizes = second_layer_sizes
        self.firstFNN = dde.nn.PFNN(first_layer_sizes, activation, kernel_initializer)
        self.secondFNN = dde.nn.PFNN(second_layer_sizes, activation, kernel_initializer)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        x_firstFNN = self.firstFNN(x)
        x_secondFNN = self.secondFNN(x)
        x = torch.cat((self.firstFNN(x), self.secondFNN(x)), dim=1)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x

# Read and organize input data
data = np.load("\...\.....npy", allow_pickle=True)
coors, gt_disp = data.item()["coordinates"], data.item()["displacements"]
time_value = 1.0 


coors_t0 = np.hstack((coors, np.zeros((len(coors), 1)))) # ()
coors_t1 = np.hstack((coors, time_value * np.ones((len(coors), 1)))) # 


# Compute mean and std of displacements for normalization
ux_mean, uy_mean, uz_mean = (
    np.mean(gt_disp[:, 0]),
    np.mean(gt_disp[:, 1]),
    np.mean(gt_disp[:, 2]),
)
# Standard deviation of displacements for normalization
ux_std, uy_std, uz_std = (
    np.std(gt_disp[:, 0]),
    np.std(gt_disp[:, 1]),
    np.std(gt_disp[:, 2]),
)


# Randomly sample 2500 points from each time step for PDE residual training
idx1 = np.random.choice(len(coors_t0), 2500, replace=False)
idx2 = np.random.choice(len(coors_t1), 2500, replace=False)
pde_pts = np.vstack((coors_t0[idx1, :], coors_t1[idx2, :]))
pde_pts_disp = np.vstack((np.zeros((len(coors_t0[idx1, :]), 3)), gt_disp[idx2, :]))

geomtime = dde.geometry.PointCloud(points=pde_pts) # Domain for PDE residual training

loss = [
    dde.PointSetBC(
        coors_t0[idx1, :], np.zeros((len(coors[idx1, :]), 3)), component=[0, 1, 2]
    ),
    dde.PointSetBC(
        coors_t1[idx2, :], coors_t1[idx2, :3] + gt_disp[idx2, :], component=[9, 10, 11]
    ),
]

# Model variables
E_ = dde.Variable(1.0)
nu_ = dde.Variable(1.0)


def stress(x, y):
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3] # Displacement components from the first FNN output

    # Compute deformation gradients

    duxdx = dde.grad.jacobian(Nux, x, i=0, j=0)
    duxdy = dde.grad.jacobian(Nux, x, i=0, j=1)
    duxdz = dde.grad.jacobian(Nux, x, i=0, j=2)

    duydx = dde.grad.jacobian(Nuy, x, i=0, j=0)
    duydy = dde.grad.jacobian(Nuy, x, i=0, j=1)
    duydz = dde.grad.jacobian(Nuy, x, i=0, j=2)

    duzdx = dde.grad.jacobian(Nuz, x, i=0, j=0)
    duzdy = dde.grad.jacobian(Nuz, x, i=0, j=1)
    duzdz = dde.grad.jacobian(Nuz, x, i=0, j=2)

    Fxx = duxdx + 1.0
    Fxy = duxdy
    Fxz = duxdz

    Fyx = duydx
    Fyy = duydy + 1.0
    Fyz = duydz

    Fzx = duzdx
    Fzy = duzdy
    Fzz = duzdz + 1.0

    # Compute determinant and inverse of deformation gradient
    detF = (
        Fxx * (Fyy * Fzz - Fyz * Fzy)
        - Fxy * (Fyx * Fzz - Fyz * Fzx)
        + Fxz * (Fyx * Fzy - Fyy * Fzx)
    )

    detF = torch.where(torch.le(detF, 0), 0.001, detF) # Prevent non-positive determinant for numerical stability

    adjFxx = Fyy * Fzz - Fyz * Fzy
    adjFxy = -(Fxy * Fzz - Fxz * Fzy)
    adjFxz = Fxy * Fyz - Fxz * Fyy

    adjFyx = -(Fyx * Fzz - Fyz * Fzx)
    adjFyy = Fxx * Fzz - Fxz * Fzx
    adjFyz = -(Fxx * Fyz - Fxz * Fyx)

    adjFzx = Fyx * Fzy - Fzx * Fyy
    adjFzy = -(Fxx * Fzy - Fxy * Fzx)
    adjFzz = Fxx * Fyy - Fxy * Fyx

    invFxx = adjFxx / detF
    invFxy = adjFxy / detF
    invFxz = adjFxz / detF

    invFyx = adjFyx / detF
    invFyy = adjFyy / detF
    invFyz = adjFyz / detF

    invFzx = adjFzx / detF
    invFzy = adjFzy / detF
    invFzz = adjFzz / detF

    E = (torch.tanh(E_) + 1.0) * 400
    nu = (torch.tanh(nu_) + 1.0) / 4

    lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))  # mu/2 = c10


    lnF = torch.log(detF)

    Pxx = mu * Fxx + (lmbd * lnF - mu) * invFxx
    Pxy = mu * Fxy + (lmbd * lnF - mu) * invFyx
    Pxz = mu * Fxz + (lmbd * lnF - mu) * invFzx
    Pyx = mu * Fyx + (lmbd * lnF - mu) * invFxy
    Pyy = mu * Fyy + (lmbd * lnF - mu) * invFyy
    Pyz = mu * Fyz + (lmbd * lnF - mu) * invFzy
    Pzx = mu * Fzx + (lmbd * lnF - mu) * invFxz
    Pzy = mu * Fzy + (lmbd * lnF - mu) * invFyz
    Pzz = mu * Fzz + (lmbd * lnF - mu) * invFzz

    # Cauchy stress
    sxx = (Pxx * Fxx + Pxy * Fxy + Pxz * Fxz) / detF
    sxy = (Pxx * Fyx + Pxy * Fyy + Pxz * Fyz) / detF
    sxz = (Pxx * Fzx + Pxy * Fzy + Pxz * Fzz) / detF
    syy = (Pyx * Fyx + Pyy * Fyy + Pyz * Fyz) / detF
    syz = (Pyx * Fzx + Pyy * Fzy + Pyz * Fzz) / detF
    szz = (Pzx * Fzx + Pzy * Fzy + Pzz * Fzz) / detF

    return sxx, sxy, sxz, syy, syz, szz

def pde(x, y):
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    Nsxx, Nsxy, Nsxz, Nsyy, Nsyz, Nszz = (
        y[:, 3:4],
        y[:, 4:5],
        y[:, 5:6],
        y[:, 6:7],
        y[:, 7:8],
        y[:, 8:9],
    )

    sxx, sxy, sxz, syy, syz, szz = stress(x, y)

    sxx_x = dde.grad.jacobian(Nsxx, x, i=0, j=0)
    sxy_y = dde.grad.jacobian(Nsxy, x, i=0, j=1)
    sxz_z = dde.grad.jacobian(Nsxz, x, i=0, j=2)

    sxy_x = dde.grad.jacobian(Nsxy, x, i=0, j=0)
    syy_y = dde.grad.jacobian(Nsyy, x, i=0, j=1)
    syz_z = dde.grad.jacobian(Nsyz, x, i=0, j=2)

    sxz_x = dde.grad.jacobian(Nsxz, x, i=0, j=0)
    syz_y = dde.grad.jacobian(Nsyz, x, i=0, j=1)
    szz_z = dde.grad.jacobian(Nszz, x, i=0, j=2)

    rho = 1e-2 # Density in KPa*s^2/mm^3, adjust as needed for unit consistency
    d2x_dt2 = dde.grad.hessian(Nux, x, i=3, j=3)
    d2y_dt2 = dde.grad.hessian(Nuy, x, i=3, j=3)
    d2z_dt2 = dde.grad.hessian(Nuz, x, i=3, j=3)

    mx = sxx_x + sxy_y + sxz_z - rho * d2x_dt2
    my = sxy_x + syy_y + syz_z - rho * d2y_dt2
    mz = sxz_x + syz_y + szz_z - rho * d2z_dt2

    stress_xx = sxx - Nsxx
    stress_yy = syy - Nsyy
    stress_zz = szz - Nszz
    stress_xy = sxy - Nsxy
    stress_xz = sxz - Nsxz
    stress_yz = syz - Nsyz

    return [
        mx,
        my,
        mz,
        stress_xx,
        stress_yy,
        stress_zz,
        stress_xy,
        stress_xz,
        stress_yz,
    ]


def output_transform(x, y):
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    Nux = Nux * ux_std + ux_mean
    Nuy = Nuy * uy_std + uy_mean
    Nuz = Nuz * uz_std + uz_mean

    Nsxx, Nsxy, Nsxz, Nsyy, Nsyz, Nszz = (
        y[:, 3:4],
        y[:, 4:5],
        y[:, 5:6],
        y[:, 6:7],
        y[:, 7:8],
        y[:, 8:9],
    )

    Nux_new = Nux + x[:, 0:1]
    Nuy_new = Nuy + x[:, 1:2]
    Nuz_new = Nuz + x[:, 2:3]

    return torch.concat(
        [Nux, Nuy, Nuz, Nsxx, Nsxy, Nsxz, Nsyy, Nsyz, Nszz, Nux_new, Nuy_new, Nuz_new],
        axis=1,
    )


def hausdorff_distance(y_true, y_pred):
    distances = torch.cdist(y_pred[:, :3], y_true[:, :3], p=2)
    avg_distances_1 = torch.mean(
        torch.min(distances, dim=1).values
    )  # Max of min distances from 1 to 2
    avg_distances_2 = torch.mean(
        torch.min(distances, dim=0).values
    )  # Max of min distances from 2 to 1
    error = 0.5 * (avg_distances_1 + avg_distances_2)

    return error


first_layer_sizes  = [4] + [[32, 32, 32]] + [[16,16,16]] + [[8,8,8]] + [3]
second_layer_sizes = [4] + [[32, 32, 32, 32, 32, 32]] + [[16,16,16,16,16,16]] + [[8,8,8,8,8,8]] + [6]




net = MPFNN(first_layer_sizes, second_layer_sizes, "swish", "Glorot normal")
net.apply_output_transform(output_transform)
data = dde.data.PDE(
    geomtime,
    pde,
    loss,
    anchors=pde_pts,
)

model = dde.Model(data, net)
loss_type = ["MSE"] * 10 + [hausdorff_distance]
model = dde.Model(data, net)
external_trainable_variables = [E_, nu_]
variables = dde.callbacks.VariableValue(
    external_trainable_variables, period=1000, filename="variables.dat"
)

model.compile(
    "adam",
    loss=loss_type,
    lr=1e-3,
    loss_weights=[1] * 10 + [1] * 1,
    external_trainable_variables=external_trainable_variables,
)

losshistory, train_state = model.train(epochs=200000, callbacks=[variables])
dde.saveplot(losshistory, train_state, issave=True, isplot=False)



lines = open("variables.dat", "r").readlines()
vkinfer = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

l, c = vkinfer.shape
E_pred = (np.tanh(vkinfer[:, 0]) + 1) * 400
nu_pred = (np.tanh(vkinfer[:, 1]) + 1) / 4
E_True = 450
nu_True = 0.35
epochs = np.arange(len(nu_pred)) * 1000
loss_array = np.array(losshistory.loss_train)  
total_loss = loss_array.sum(axis=1)

print(
    "nu prediction: ",
    nu_pred[-1],
    "percentage error (%): ",
    np.linalg.norm(nu_True - nu_pred[-1]) / np.linalg.norm(nu_True) * 100,
)

print(
    "E prediction: ",
    E_pred[-1],
    "percentage error (%): ",
    np.linalg.norm(E_True - E_pred[-1]) / np.linalg.norm(E_True) * 100,
)


# Plot nu vs epochs
plt.figure(figsize=(8, 5))
plt.plot(epochs, nu_pred, color="C1")
plt.axhline(y=nu_True, color="k", linestyle="--", label=f"True nu = {nu_True}")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("nu prediction")
plt.title("Ex4: nu vs Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("\...\....png", dpi=150)
plt.close()
print("Plot saved to \...\....png")

# Plot E vs epochs
plt.figure(figsize=(8, 5))
plt.plot(epochs, E_pred, color="C1")
plt.axhline(y=E_True, color="k", linestyle="--", label=f"True E = {E_True}")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("E prediction")
plt.title("Ex4: E vs Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("\...\....png", dpi=150)
plt.close()
print("Plot saved to \...\....png")


# Plot total training loss vs epochs
plt.figure(figsize=(8, 5))
plt.plot(total_loss, color="C2")
plt.yscale("log")  
plt.xlabel("Epochs")
plt.ylabel("Total Training Loss")
plt.title("Ex4: Training Loss vs Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("\...\....png", dpi=150)
plt.close()
print("Plot saved to \...\....png")
