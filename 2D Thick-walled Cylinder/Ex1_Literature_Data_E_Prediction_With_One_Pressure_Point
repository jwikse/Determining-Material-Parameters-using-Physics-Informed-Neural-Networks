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
        self.firstFNN  = dde.nn.PFNN(first_layer_sizes,  activation, kernel_initializer)
        self.secondFNN = dde.nn.PFNN(second_layer_sizes, activation, kernel_initializer)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        x = torch.cat((self.firstFNN(x), self.secondFNN(x)), dim=1)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x



# Read and organize input data
data = np.load("..../...../hollow_cylinder_data.npy", allow_pickle="TRUE")

coors, gt_disp = data.item()["coordinates"], data.item()["displacements"]

bnd_coors, bond_gt_disp = (data.item()["boundary_coordinates"],data.item()["boundary_displacements"],)
radius = data.item()["radius"]


# Normdimensionalize displacements (*Does not effect physics?)

scale = 1e4
gt_disp = gt_disp * scale
bond_gt_disp = bond_gt_disp * scale

ux_mean, uy_mean = np.mean(gt_disp[:, 0]), np.mean(gt_disp[:, 1])
ux_std, uy_std = np.std(gt_disp[:, 0]), np.std(gt_disp[:, 1])



# Extract sampling points
idx1 = np.random.choice(np.where(radius < 2)[0], 500, replace=False)
idx2 = np.random.choice(np.where((radius > 2) & (radius < 3))[0], 400, replace=False)
idx3 = np.random.choice(np.where((radius > 3) & (radius < 4))[0], 300, replace=False)
idx4 = np.random.choice(np.where((radius > 4) & (radius <= 5))[0], 200, replace=False)
idx5 = np.random.choice(np.where(bnd_coors)[0], 150, replace=False)

pde_pts = np.vstack(
    (coors[idx1, :], coors[idx2, :], coors[idx3, :], coors[idx4, :], bnd_coors[idx5, :])
)
pde_pts_disp = np.vstack(
    (
        gt_disp[idx1, :],
        gt_disp[idx2, :],
        gt_disp[idx3, :],
        gt_disp[idx4, :],
        bond_gt_disp[idx5, :],
    )
)

geom = dde.geometry.PointCloud(points=pde_pts)

p_in = 1e-5 * scale

pressure_point = np.array([[1.0, 0.0]])
pressure_value = np.array([[-p_in]])

losses = [
    dde.PointSetBC(pde_pts, pde_pts_disp, component=[0, 1]),
    dde.PointSetBC(pressure_point, pressure_value, component=5),
]

# Model variables

E_ = dde.Variable(1.0)


# Compute strain tensor from displacements
def strain(x, y):
    ux, uy = y[:, 0:1], y[:, 1:2]

    exx = dde.grad.jacobian(ux, x, i=0, j=0)
    eyy = dde.grad.jacobian(uy, x, i=0, j=1)
    exy = 0.5 * (
        dde.grad.jacobian(ux, x, i=0, j=1) + dde.grad.jacobian(uy, x, i=0, j=0)
    )

    return exx, eyy, exy


# Compute stess tensor from displacements
def stress(x, y):
    exx, eyy, exy = strain(x, y)

    E = (torch.tanh(E_) + 1.0) / 10 # Interval range for E: (0, 0.2)  
    nu = 0.3

    sxx = E / (1 - nu**2) * (exx + nu * eyy)
    syy = E / (1 - nu**2) * (eyy + nu * exx)
    sxy = E / (1 + nu) * exy

    return sxx, syy, sxy


def pde(x, y):
    Nsxx, Nsyy, Nsxy, Nsrr = (
        y[:, 2:3],
        y[:, 3:4],
        y[:, 4:5],
        y[:, 5:6],
    )

    sxx_x = dde.grad.jacobian(Nsxx, x, i=0, j=0)
    syy_y = dde.grad.jacobian(Nsyy, x, i=0, j=1)
    sxy_y = dde.grad.jacobian(Nsxy, x, i=0, j=1)
    sxy_x = dde.grad.jacobian(Nsxy, x, i=0, j=0)

    mx = sxx_x + sxy_y
    my = sxy_x + syy_y

    sxx, syy, sxy = stress(x, y)

    # Safer radial normal vector
    rad = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2 + 1e-12)
    nx = x[:, 0:1] / rad
    ny = x[:, 1:2] / rad

    # Radial stress from Cartesian stress
    srr = sxx * nx**2 + syy * ny**2 + 2 * sxy * nx * ny

    stress_x = sxx - Nsxx
    stress_y = syy - Nsyy
    stress_xy = sxy - Nsxy
    stress_rr = srr - Nsrr

    return mx, my, stress_x, stress_y, stress_xy, stress_rr


data = dde.data.PDE(
    geom,
    pde,
    losses,
    anchors=pde_pts,
)


# Nondimensionalize network output variables
def output_transform(x, y):
    Nux, Nuy = y[:, 0:1], y[:, 1:2]
    Nsxx, Nsyy, Nsxy, Nsrr = y[:, 2:3], y[:, 3:4], y[:, 4:5], y[:, 5:6]

    Nux = x[:, 0:1] * (Nux * ux_std + ux_mean)
    Nuy = x[:, 1:2] * (Nuy * uy_std + uy_mean)

    rad = torch.sqrt(torch.square(x[:, 0:1]) + torch.square(x[:, 1:2]))

    # Keep Cartesian stresses as network outputs
    Nsxx = Nsxx
    Nsyy = Nsyy
    Nsxy = Nsxy


    return torch.concat([Nux, Nuy, Nsxx, Nsyy, Nsxy, Nsrr], axis=1)



first_layer_sizes  = [2] + [[45, 45]] * 5 + [2]
second_layer_sizes = [2] + [[45, 45, 45, 45]] * 5 + [4]

net = MPFNN(first_layer_sizes, second_layer_sizes, "swish", "Glorot normal")

net.apply_output_transform(output_transform)

model = dde.Model(data, net)
external_trainable_variables = [E_]
variables = dde.callbacks.VariableValue(
    external_trainable_variables, period=1000, filename="variables.dat"
)

model.compile(
    "adam",
    lr=1e-4,
    loss_weights=[1] * 2 + [2] * 4 + [1]+[4],
    external_trainable_variables=external_trainable_variables,
)
losshistory, train_state = model.train(epochs=50000, callbacks=[variables])
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
E_true = 1.35e-1
E_pred = (np.tanh(vkinfer[:, 0]) + 1.0) / 10
epochs = np.arange(len(E_pred)) * 1000
loss_array = np.array(losshistory.loss_train) 
total_loss = loss_array.sum(axis=1)

print(
    "E prediction: ",
    E_pred[-1],
    "percentage error (%): ",
    np.linalg.norm(E_true - E_pred[-1]) / np.linalg.norm(E_true) * 100,
)

# Plot E vs epochs
plt.figure(figsize=(8, 5))
plt.plot(epochs, E_pred, color="C0")
plt.axhline(y=E_true, color="k", linestyle="--", label=f"True E = {E_true}")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("E prediction (N/μm^2)")
plt.title("Ex1 Const Nu NoPressure: E vs Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("....../....../.png", dpi=150)
plt.close()
print("E plot saved to ..../......png")


# Plot total training loss vs epochs
plt.figure(figsize=(8, 5))
plt.plot(total_loss, color="C2")
plt.yscale("log")  
plt.xlabel("Epochs")
plt.ylabel("Total Training Loss")
plt.title("Ex1 Const Nu NoPressure: Training Loss vs Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("..../......png", dpi=150)
plt.close()
