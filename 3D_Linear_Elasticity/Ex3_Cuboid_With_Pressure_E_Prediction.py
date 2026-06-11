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
data = np.load("/......./Ex3_Validation_E400_Nu04.npy", allow_pickle="TRUE")

coors, gt_disp = data.item()["coordinates"], data.item()["displacements"]


sigma_ref = 0.01 # MPa, - approximate pressure from Abaqus
E_ref = 400
L_ref = 2.0 
u_ref = sigma_ref * L_ref / E_ref   # np.max(np.abs(gt_disp))  another method
E_thilde = E_ref / sigma_ref # not used
coors = coors / L_ref
gt_disp = gt_disp / u_ref 


Pressure = 0.01

P_in = Pressure/sigma_ref


ux_mean, uy_mean, uz_mean = np.mean(gt_disp[:, 0]), np.mean(gt_disp[:, 1]), np.mean(gt_disp[:, 2])
ux_std, uy_std, uz_std = np.std(gt_disp[:, 0]), np.std(gt_disp[:, 1]), np.std(gt_disp[:, 2])


# Extract sampling points
idx1 = np.random.choice(np.arange(len(coors)), 9000, replace=False)
pde_pts = coors[idx1, :]
pde_pts_disp = gt_disp[idx1, :]

tol= 1e-6

def face_pts(axis, val):
    """Return all nodes where coors[:,axis] ≈ val."""
    mask = np.abs(coors[:, axis] - val) < tol
    return coors[mask]
 
# Pressure face: z=2, σzz = -p/σ₀  (component 8)
z2_pts = face_pts(2, np.max(coors[:,2]))

pressure_pts   = z2_pts                                    # all 441 nodes
pressure_value = -P_in * np.ones((len(z2_pts), 1))


# Free face: x=1, σxx=σxy=σxz=0 (components 3,4,5)
x1_pts = face_pts(0, np.max(coors[:,0]))



geom = dde.geometry.PointCloud(points=pde_pts)

losses = [
    dde.PointSetBC(pde_pts, pde_pts_disp, component=[0, 1, 2]),          # index 9
    dde.PointSetBC(pressure_pts, pressure_value, component=8),            # index 10
    # x=1 free face: σxx, σxy, σxz = 0, with can be added or removed, code converge
    dde.PointSetBC(x1_pts, np.zeros((len(x1_pts), 1)), component=3),     # index 11
    dde.PointSetBC(x1_pts, np.zeros((len(x1_pts), 1)), component=4),     # index 12
    dde.PointSetBC(x1_pts, np.zeros((len(x1_pts), 1)), component=5),     # index 13

]

# Model variables
E_true = 400 # MPa 
E_ = dde.Variable(0.2)
nu_ = 0.4

# Compute strain tensor from displacements
def strain(x, y):
    ux, uy, uz = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    exx = dde.grad.jacobian(ux, x, i=0, j=0)
    exy = 0.5 * (
        dde.grad.jacobian(ux, x, i=0, j=1) + dde.grad.jacobian(uy, x, i=0, j=0)
    )
    exz = 0.5 * (
        dde.grad.jacobian(ux, x, i=0, j=2) + dde.grad.jacobian(uz, x, i=0, j=0)
    )
    eyy = dde.grad.jacobian(uy, x, i=0, j=1)
    eyz = 0.5 * (
        dde.grad.jacobian(uy, x, i=0, j=2) + dde.grad.jacobian(uz, x, i=0, j=1)
    )
    ezz = dde.grad.jacobian(uz, x, i=0, j=2)

    return exx, exy, exz, eyy, eyz, ezz

# Compute stress tensor from displacements
def stress(x, y):
    exx, exy, exz, eyy, eyz, ezz = strain(x, y)

    E = (torch.tanh(E_) + 1) * 1
    nu = 0.4

    c1 = E / (1 + nu)
    c2 = nu / (1 - 2 * nu)

    sxx = c1 * (exx + c2 * (exx + eyy + ezz))
    sxy = c1 * exy
    sxz = c1 * exz
    syy = c1 * (eyy + c2 * (exx + eyy + ezz))
    syz = c1 * eyz
    szz = c1 * (ezz + c2 * (exx + eyy + ezz))

    return sxx, sxy, sxz, syy, syz, szz

# Define the governing equations to constrain the networks
def pde(x, y):
    sxx, sxy, sxz, syy, syz, szz = stress(x, y)
    Nsxx, Nsxy, Nsxz, Nsyy, Nsyz, Nszz = (
        y[:, 3:4],
        y[:, 4:5],
        y[:, 5:6],
        y[:, 6:7],
        y[:, 7:8],
        y[:, 8:9],
    )

    sxx_x = dde.grad.jacobian(Nsxx, x, i=0, j=0)
    sxy_y = dde.grad.jacobian(Nsxy, x, i=0, j=1)
    sxz_z = dde.grad.jacobian(Nsxz, x, i=0, j=2)

    sxy_x = dde.grad.jacobian(Nsxy, x, i=0, j=0)
    syy_y = dde.grad.jacobian(Nsyy, x, i=0, j=1)
    syz_z = dde.grad.jacobian(Nsyz, x, i=0, j=2)

    sxz_x = dde.grad.jacobian(Nsxz, x, i=0, j=0)
    syz_y = dde.grad.jacobian(Nsyz, x, i=0, j=1)
    szz_z = dde.grad.jacobian(Nszz, x, i=0, j=2)

    mx = sxx_x + sxy_y + sxz_z
    my = sxy_x + syy_y + syz_z
    mz = sxz_x + syz_y + szz_z

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

data = dde.data.PDE(
    geom,
    pde,
    losses,
    anchors=pde_pts,
)


def output_transform(x, y):
    Nux, Nuy, Nuz = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    Nux = Nux 
    Nuy = Nuy 
    Nuz = Nuz 

    Nsxx = y[:, 3:4]
    Nsxy = y[:, 4:5] 
    Nsxz = y[:, 5:6] 
    Nsyy = y[:, 6:7] 
    Nsyz = y[:, 7:8] 
    Nszz = y[:, 8:9] 

    return torch.cat([Nux, Nuy, Nuz, Nsxx, Nsxy, Nsxz, Nsyy, Nsyz, Nszz], dim=1)

first_layer_sizes  = [3] + [[64, 64, 64]]  + [3]      
second_layer_sizes = [3] + [[64, 64, 64, 64, 64, 64]] + [6]

net = MPFNN(first_layer_sizes, second_layer_sizes, "swish", "Glorot normal")

net.apply_output_transform(output_transform)




model = dde.Model(data, net)
external_trainable_variables = [E_]
variables = dde.callbacks.VariableValue(
    external_trainable_variables, period=1000, filename="variables.dat"
)



target_weights = [1]*3 + [2]*6 + [1, 4] + [2]*3


model.compile("adam", lr=1e-3, loss_weights=target_weights,
              external_trainable_variables=external_trainable_variables)
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
E_true = 400 # MPa
E_pred = (np.tanh(vkinfer[:, 0]) + 1) * 1 *E_ref
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
plt.ylabel("E prediction")
plt.title("E vs Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("......../..../.png", dpi=150)
plt.close()
print("E plot saved to ....../...../.png")


# Plot total training loss vs epochs
plt.figure(figsize=(8, 5))
plt.plot(total_loss, color="C2")
plt.yscale("log")  
plt.xlabel("Epochs")
plt.ylabel("Total Training Loss")
plt.title("Ex3 Const E Original Data: Training Loss vs Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("......../..../.png", dpi=150)
plt.close()
