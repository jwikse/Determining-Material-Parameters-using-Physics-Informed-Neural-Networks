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
data = np.load("....../....../Ex1_Validation_E1000MPa_Nu0.35_P1MPa.npy", allow_pickle="TRUE")

coors, gt_disp = data.item()["coordinates"], data.item()["displacements"]


# Normdimensionalize displacements
scale = 1e3
gt_disp = gt_disp * scale

ux_mean, uy_mean = np.mean(gt_disp[:, 0]), np.mean(gt_disp[:, 1])
ux_std, uy_std = np.std(gt_disp[:, 0]), np.std(gt_disp[:, 1])


# Extract sampling points
idx1 = np.random.choice(np.arange(len(coors)), 1750, replace=False)

pde_pts = coors[idx1, :]
pde_pts_disp = gt_disp[idx1, :]

geom = dde.geometry.PointCloud(points=pde_pts)

losses = [dde.PointSetBC(pde_pts, pde_pts_disp, component=[0, 1]),]

# Model variables
p_in = 1
E_ = dde.Variable(0.8)


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

    E = (torch.tanh(E_) + 1.0) # Interval range for E: (0, 2)  
    nu = 0.35

    sxx = E / (1 - nu**2) * (exx + nu * eyy)
    syy = E / (1 - nu**2) * (eyy + nu * exx)
    sxy = E / (1 + nu) * exy

    return sxx, syy, sxy


# Define the governing equations to constrain the networks
def pde(x, y):
    Nsxx, Nsyy, Nsxy, Nsrr = y[:, 2:3], y[:, 3:4], y[:, 4:5], y[:, 5:6]

    sxx_x = dde.grad.jacobian(Nsxx, x, i=0, j=0)
    syy_y = dde.grad.jacobian(Nsyy, x, i=0, j=1)
    sxy_y = dde.grad.jacobian(Nsxy, x, i=0, j=1)
    sxy_x = dde.grad.jacobian(Nsxy, x, i=0, j=0)

    mx = sxx_x + sxy_y
    my = sxy_x + syy_y

    sxx, syy, sxy = stress(x, y)

    nx = torch.cos(torch.arctan(x[:, 1:2] / x[:, 0:1]))
    ny = torch.sin(torch.arctan(x[:, 1:2] / x[:, 0:1]))
    srr = sxx * torch.square(nx) + syy * torch.square(ny) + sxy * 2 * nx * ny

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

    Nsrr = ((rad - 1) * Nsrr - 1) * p_in / -4 * (rad - 5)

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
    loss_weights=[0.1] * 2 + [0.1] * 4 + [0.9],
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
E_true = 1000 # MPa
E_pred = (np.tanh(vkinfer[:, 0]) + 1.0)*scale 
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
plt.axhline(y=E_true, color="k", linestyle="--", label=f"True E = {E_true} MPa")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("E prediction (N/mm^2)")
plt.title("Ex1 Const Nu Abaqus Data: E vs Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("..../...../.png", dpi=150)
plt.close()
print("E plot saved to ..../...../.png")


# Plot total training loss vs epochs
plt.figure(figsize=(8, 5))
plt.plot(total_loss, color="C2")
plt.yscale("log")  
plt.xlabel("Epochs")
plt.ylabel("Total Training Loss")
plt.title("Ex1 Const Nu Abaqus Data: Training Loss vs Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("..../...../.png", dpi=150)
plt.close()
