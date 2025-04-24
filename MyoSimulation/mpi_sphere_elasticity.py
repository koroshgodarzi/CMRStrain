import dolfinx as fe
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem, assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from dolfinx.nls.petsc import NewtonSolver
import numpy as np
import os
from mpi4py import MPI
import ufl
from petsc4py import PETSc
import pyvista
import pandas as pd
import meshio
from scipy.spatial import cKDTree
from scipy.special import jv, yv, iv, kv
from itertools import product
import h5py
import matplotlib.pyplot as plt
import argparse
import json

from natural_freq import natural_freq

import sys
sys.path.append("/Users/korosh/anaconda3/pkgs/petsc-3.20.6-real_he4ff8f5_100/lib/petsc/bin")
petsc_dir = "/Users/korosh/anaconda3/pkgs/petsc-3.20.6-real_he4ff8f5_100" 
os.environ["PETSC_DIR"] = petsc_dir

import PetscBinaryIO

import logging

logging.basicConfig(filename='error_compared_to_exact.log', level=logging.INFO, format='%(asctime)s:%(message)s')


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data.astype(np.int32)]})
    return out_mesh


parser = argparse.ArgumentParser(description="Parameters of ellipsoid.")
parser.add_argument('-maxp', '--maximum_pressure', type=float, default=1.0)
parser.add_argument('-minp', '--minimum_pressure', type=float, default=0.0)
parser.add_argument('-E', '--elasticity', type=int, default=10000)
parser.add_argument('-r', '--rho', type=float, default=1000.0)
parser.add_argument('-n', '--nu', type=float, default=0.3)
parser.add_argument('-ed', '--element_degree', type=int, default=1)
parser.add_argument('-sn', '--step_numbers', type=int, default=100)
parser.add_argument('-dt', '--dtype', type=type, default=np.float64) 
parser.add_argument('-v', '--visualize', type=bool, default=False)
parser.add_argument('-e', '--exact', type=bool, default=True)
parser.add_argument('-c', '--center', type=np.array, default=np.array([0, 0, 0]))
parser.add_argument('-m', '--mesh', type=str, default='mesh_tuned_dim_0.msh')
parser.add_argument('-o', '--output_file', type=str, default='test')
parser.add_argument('-tim', '--time_integration_method', type=str, default='0.01 0.01')
parser.add_argument('-gm', '--gamma_pressure', type=float, default=1)

args = parser.parse_args()

proc = MPI.COMM_WORLD.rank
proc_size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

output_file = args.time_integration_method

mesh_file = os.path.join("..", "Mesh", "Output", "sphere", args.mesh)

if proc == 0 and not os.path.exists(os.path.join("output", 'sphere', 'time_integration_study', output_file)):
    os.makedirs(os.path.join("output", 'sphere', output_file))

output_file = os.path.join("output", 'sphere', 'time_integration_study', output_file)

maximum_pressure = fe.default_scalar_type(args.maximum_pressure)
minimum_pressure = fe.default_scalar_type(args.minimum_pressure)
rho = fe.default_scalar_type(args.rho)
E = fe.default_scalar_type(args.elasticity)
nu = fe.default_scalar_type(args.nu)
element_degree = args.element_degree
num_steps = args.step_numbers
dtype = args.dtype
visualize = args.visualize

############################################Importing the mesh############################################
##########################################################################################################
# mesh, cell_markers, facet_markers = fe.io.gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)

#Wrinting the mesh as XDMF file to read from that later
if proc == 0:
    # Read in mesh
    msh = meshio.read(mesh_file)

    # Create and save one file for the mesh, and one file for the facets
    tetra_mesh = create_mesh(msh, "tetra", prune_z=False)
    triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
    # facets = create_mesh(msh, "triangle", prune_z=False)
    meshio.write(os.path.join(output_file, "mt.xdmf"), triangle_mesh)
    meshio.write(os.path.join(output_file, "mesh.xdmf"), tetra_mesh)

MPI.COMM_WORLD.barrier()

#Reading the xdmf files
with fe.io.XDMFFile(MPI.COMM_WORLD, os.path.join(output_file, "mesh.xdmf"), "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    cell_markers = xdmf.read_meshtags(mesh, name="Grid")

gdim = mesh.topology.dim
mesh.topology.create_connectivity(gdim, gdim - 1)
with fe.io.XDMFFile(MPI.COMM_WORLD, os.path.join(output_file, "mt.xdmf"), "r") as xdmf:
    facet_markers = xdmf.read_meshtags(mesh, name="Grid")


local_correspondence = {}
radius_to_check = [0.05, 0.06]
index_of_radius_to_check = []
coord_of_index_to_check = []


for r in radius_to_check:
    if proc == 0:
        points = mesh.geometry.x
        for i, p in enumerate(points - args.center):
            if np.linalg.norm(p) == r:
                index_of_radius_to_check.append(i)
                coord_of_index_to_check.append(p)
                # local_correspondence[0] = i
                break
    else:
        coord_of_index_to_check = None

MPI.COMM_WORLD.Barrier()
# print(points.shape)

######################################################################################################
#######################creating function space and boundary condition#################################
element = ufl.VectorElement("CG", mesh.ufl_cell(), element_degree)
V = fe.fem.functionspace(mesh, element)

# num_steps = 10
gamma_p = args.gamma_pressure # 2*np.pi     ################MIND THE GAMMA 0.01
end = 2*np.pi / gamma_p
# t = np.concatenate((np.linspace(0, end/15, int(num_steps/4)), np.linspace(end/15, 14 * end/15, int(num_steps/2)), np.linspace(14*end/15, end, int(num_steps/4))))
t = np.linspace(0, end, num_steps)
dt = fe.fem.Constant(mesh, t[1] - t[0])

pressure = maximum_pressure / 2 * (1 - np.cos(gamma_p * t))
p = fe.fem.Constant(mesh, pressure[0])     ####The pressure will be applied in the formulation


######################################################################################################
######################################################################################################

######################################################################################################
#########################################Formulation##################################################
u = ufl.TrialFunction(V)           
v  = ufl.TestFunction(V)          
old_u  = fe.fem.Function(V, dtype=dtype)
old_velocity  = fe.fem.Function(V, dtype=dtype)
old_acceleration  = fe.fem.Function(V, dtype=dtype)

d = len(u)
I = ufl.variable(ufl.Identity(d))             # Identity tensor
F = ufl.variable(I + ufl.grad(u))             # Deformation gradient
C = ufl.variable(F.T*F)                   # Right Cauchy-Green tensor
J = ufl.variable(ufl.det(F))

####Check out for the metadata
metadata = {"quadrature_degree": 4}   ### The quadrature degree has a great impact on the steady state problem, but apparantely none in the dynamic casse.
ds = ufl.Measure('ds', domain=mesh, subdomain_data=facet_markers, metadata=metadata)
dx = ufl.Measure("dx", domain=mesh, metadata=metadata)

# Generalized-alpha method parameters
tim = args.time_integration_method.split()
tim = list(map(float, tim))
if len(tim) == 2:
    alpha_m = fe.fem.Constant(mesh, -0.1429)  #   0.2  -0.0526
    alpha_f = fe.fem.Constant(mesh, 0.2857)   #  0.4  0.4211
    gamma   = fe.fem.Constant(mesh, fe.default_scalar_type(0.5+alpha_f-alpha_m))
    beta    = fe.fem.Constant(mesh, fe.default_scalar_type((gamma+0.5)**2/4.))
elif len(tim) == 4:
    alpha_m = fe.fem.Constant(mesh, 0.0)  # No numerical dissipation for mass
    alpha_f = fe.fem.Constant(mesh, 0.0)  # No numerical dissipation for forces
    gamma   = fe.fem.Constant(mesh, fe.default_scalar_type(0.5))  # Central difference scheme
    beta    = fe.fem.Constant(mesh, fe.default_scalar_type(0.25))  # Standard Newmark parameter
else:
    print("Invalid Time Integration Parameters!!")
    exit()

# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)


def update_fields(u_new, u_old, v_old, a_old):
    """Update fields at the end of each time step."""
    # Get vectors (references)
    u_vec, u0_vec  = u_new.x.array[:], u_old.x.array[:]
    v0_vec, a0_vec = v_old.x.array[:], a_old.x.array[:]

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

    # Update (u_old <- u)
    v_old.x.array[:], a_old.x.array[:] = v_vec, a_vec
    u_old.x.array[:] = u_new.x.array[:]


def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new

normal = -ufl.FacetNormal(mesh)
# X, Y, Z = ufl.SpatialCoordinate(mesh)
# normal_analytic = ufl.as_vector((X, Y, Z))

mu = fe.fem.Constant(mesh, E / (2 * (1 + nu)))
lmbda = fe.fem.Constant(mesh, E * nu / ((1 + nu) * (1 - 2 * nu)))

def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def S(u):
    return 2.0 * mu * epsilon(u) + lmbda * ufl.nabla_div(u) * I  # ufl.sym(ufl.grad(u)) # ufl.tr(ufl.sym(ufl.grad(u)))

acceleration = update_a(u, old_u, old_velocity, old_acceleration, ufl=False)    #### Changing the ufl to True would not work. It makes the acceleration type not suitable to be added to other variables in the formulation!
velocity = update_v(acceleration, old_u, old_velocity, old_acceleration, ufl=True)

formulation = rho * ufl.inner(avg(old_acceleration, acceleration, alpha_m), v) * dx \
  + ufl.inner(epsilon(v), S(avg(old_u, u, alpha_f))) * dx \
  - ufl.dot(v, p * normal) * ds(112)     #  ufl.inner(v, p * normal)

bilinear_form = fe.fem.form(ufl.lhs(formulation))
linear_form = fe.fem.form(ufl.rhs(formulation))
######################################################################################################
######################################################################################################

######################################################################################################
###############################################Solving################################################
A = assemble_matrix(bilinear_form, bcs=[])
A.assemble()
b = create_vector(linear_form)

solver = PETSc.KSP().create(mesh.comm)
solver.setInitialGuessNonzero(True)
solver.setOperators(A)
solver.getPC().setType(PETSc.PC.Type.SOR)
# fe.log.set_log_level(fe.log.LogLevel.INFO)
# solver.view()

a = ufl.inner(S(u), epsilon(v)) * dx
L = ufl.dot(p * normal, v) * ds(112)

displacement = fe.fem.Function(V, dtype=dtype)

if visualize == True:
    ##############Initializing the plot###################
    pyvista.start_xvfb()
    plotter = pyvista.Plotter()
    plotter.open_gif(os.path.join(output_file,"deformation.gif"), fps=3)

    topology, cell_types, geometry = fe.plot.vtk_mesh(mesh, 3)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    values = np.zeros((geometry.shape[0], 3))
    values[:, :len(u)] = displacement.x.array.astype(float).reshape(geometry.shape[0], len(displacement))
    grid.point_data["u"] = values
    grid.set_active_vectors("u")

    # Warp mesh by deformation
    warped = grid.warp_by_vector("u", factor=10)
    warped.set_active_vectors("u")

    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False, clim=[0, 1])

    Vs = fe.fem.FunctionSpace(mesh, ("Lagrange", 1))
    magnitude = fe.fem.Function(Vs)
    us = fe.fem.Expression(ufl.sqrt(sum([displacement[i]**2 for i in range(len(displacement))])), Vs.element.interpolation_points())
    magnitude.interpolate(us)
    warped["mag"] = magnitude.x.array

    plotter.write_frame()

class exact_solution():
    def __init__(self, rho, inner_radius, k, lmbda, miu, p_0, gamma, num_steps):  # p = -p_0 * (1-cos(gammma*tau))
        self.a = inner_radius
        self.k = k  # outer_radius = k * a
        self.betta = 0
        self.phi = -(1 + self.betta)/2
        second_lame_constant = (lmbda/2*(lmbda+miu))
        self.n = np.sqrt(self.phi * self.phi - 2 * (second_lame_constant * self.betta/(1- second_lame_constant) - 1))
        self.C_11_0 = lmbda + 2 * miu #E * (1 - miu) / ((1 + miu) * (1 - 2*miu))
        self.C_12_0 = lmbda #E * miu / ((1 + miu) * (1 - 2*miu))
        self.m = self.C_12_0 / self.C_11_0
        self.S_1 = self.n - 2 * self.m - self.phi
        self.alpha = natural_freq(self.n, self.S_1, self.k) 
        self.alpha.sort()
        # print(self.alpha)
        self.p_0 = p_0
        self.gamma = gamma
        self.tau = 0 # 6 * 2 * np.pi
        self.c = np.sqrt(self.C_11_0 / rho)
        # self.dtau =  2*np.pi / self.gamma /num_steps # dt /(2*np.pi)  # * self.c
        self.dtau = 2*np.pi / self.gamma /num_steps


    def return_tau(self):
        return self.tau


    def step_one_period(self):
        self.tau += self.gamma * 2*np.pi # self.c 


    def step_in_time(self):
        self.tau += self.dtau
        # print(f"tau: {self.tau}")


    def r(self, x):   # distance to the center 
        return np.sqrt(np.power(x[0], 2) + np.power(x[1], 2) + np.power(x[2], 2))


    def theta_cood(self, x):
        return np.arccos(x[2]/self.r(x))


    def phi_cood(self, x):
        return np.sign(x[1]) * np.arccos(x[0]/np.sqrt(np.power(x[0], 2) + np.power(x[1], 2)))


    def F(self, p, x):
        k = self.k
        n = self.n
        S_1 = self.S_1
        a = self.a
        return (kv(n, p*k + 0j) * S_1 + p*k * kv(n-1, p*k + 0j)) * iv(n, p*self.r(x)/a + 0j)\
               - (iv(n, p*k + 0j) * S_1 - p*k * iv(n-1, p*k + 0j)) * kv(n, p*self.r(x)/a + 0j)


    def G(self, p):
        k = self.k
        n = self.n
        S_1 = self.S_1
        a = self.a
        return (kv(n, p*k + 0j) * S_1 + p*k * kv(n-1, p*k + 0j)) * (iv(n, p + 0j) * S_1 - p * iv(n-1, p + 0j))\
               - (kv(n, p + 0j) * S_1 + p * kv(n-1, p + 0j)) * (iv(n, p*k + 0j) * S_1 - p*k * iv(n-1, p*k + 0j))


    def __call__(self, x, radial=False):

        p_0 = self.p_0
        a = self.a
        phi = self.phi
        C_11 = self.C_11_0 * np.power(self.r(x), self.betta)
        k = self.k
        n = self.n
        S_1 = self.S_1
        gamma = self.gamma
        alpha = self.alpha

        first_term = -(p_0 * np.power(self.r(x)/a, phi) / C_11)\
                    * (-np.power(k, 2*n) * (-2*n + S_1) + S_1 * np.power(self.r(x)/a, 2*n))\
                    / ((-1+np.power(k, 2*n)) * (-2*n + S_1) * S_1 * np.power(self.r(x)/a, n))
        second_term = -(p_0 * np.power(self.r(x)/a, phi) / C_11)\
                    * self.F(np.array([1j * gamma]), x) * np.cos(gamma * self.tau)\
                    / self.G(np.array([1j * gamma]))
        series = 0
        for s in range(len(alpha)):
            first_parantheses = S_1 * jv(n, alpha[s] + 0j) - alpha[s] * jv(n-1, alpha[s] + 0j)
            second_parantheses = np.power(S_1, 2) - 2 * S_1 * n + np.power(alpha[s], 2)
            third_parantheses = np.power(S_1 * jv(n, alpha[s] * k + 0j) - alpha[s] * k * jv(n-1, alpha[s] * k + 0j), 2)
            forth_parantheses = S_1 * jv(n, alpha[s] * k + 0j) - alpha[s] * k * jv(n-1, alpha[s] * k + 0j)
            fifth_parantheses = np.power(S_1, 2) - 2 * S_1 * n + np.power(alpha[s] * k, 2)
            sixth_parantheses = np.power(S_1 * jv(n, alpha[s] + 0j) - alpha[s] * jv(n-1, alpha[s] + 0j), 2)

            series += 2 * self.F(np.array([1j * alpha[s]]), x)/(np.power(gamma, 2) - np.power(alpha[s], 2))\
                                    * first_parantheses * forth_parantheses / (second_parantheses * third_parantheses - fifth_parantheses * sixth_parantheses)\
                                    * np.cos(alpha[s] * (self.tau))

        third_term = (p_0 * np.power(self.r(x)/a, phi) * np.power(gamma, 2) / C_11) * series

        u = np.zeros((1, x.shape[1]))
        u = first_term + second_term + third_term

        u *= a

        displacement = np.zeros(x.shape)
        mag = np.linalg.norm(x, axis=0)
        for i in range(mag.shape[0]):
            displacement[:, i] = u[i].real * x[:, i] / mag[i]

        if radial==True:
            return u.real
        
        return displacement #u.real # [0]


u_exact = exact_solution(rho, 0.05, 6/5, lmbda.value, mu.value, maximum_pressure/2, gamma_p, num_steps)  

extended_time_ratio = 1

exact_radial_disp = t * extended_time_ratio

for r in radius_to_check:
    radial_disp = []
    if args.exact == True:
        for i in range(int(num_steps * extended_time_ratio)):
            radial_disp.append(u_exact.__call__(np.array([[r], [0], [0]]), radial=True))
            u_exact.step_in_time()

        if proc == 0:
            plt.plot(np.linspace(0, extended_time_ratio, int(num_steps * extended_time_ratio)), radial_disp, label=f'exact_radius_{r}')
            radial_disp = [x.tolist() if isinstance(x, np.ndarray) else x for x in radial_disp]
            with open(os.path.join(output_file, f'exact_radius_{r}.json'), 'w') as f:
                json.dump(radial_disp, f)
    
    u_exact.tau = 0


def error_calculation(exact, calculated):
    error = np.zeros((1, 4))

    for cood in [0, 1, 2]:
        non_zero_mask = np.argwhere(np.abs(exact[:, cood]) > 1e-5)
        err_ = np.divide(np.power(exact[non_zero_mask, cood] - calculated[non_zero_mask, cood], 2), np.power(exact[non_zero_mask, cood], 2), dtype=np.float128) # [non_zero_mask, 1]
        err_ = np.mean(err_)
        error[0, cood] = err_

    # exact = np.linalg.norm(exact, axis=1)
    # calculated = np.linalg.norm(calculated, axis=1)
    err_ = np.divide(np.power(exact - calculated, 2), np.power(exact, 2), dtype=np.float128) # [non_zero_mask, 1]
    # err_ = np.divide(np.power(exact - calculated, 2), np.power(exact, 2))
    err_ = np.mean(err_)
    error[0, 3] = err_
    return error

displacement_exact = fe.fem.Function(V, dtype=dtype)    

error_table = np.zeros((int(num_steps  * extended_time_ratio), 4))
radial_disp_1 = []
radial_disp_2 = []
error = []
rel_error = []


# Stepping through time and solving for displacements and strains
for i in range(int(num_steps  * extended_time_ratio)): #  * 3/2
    # print(i)
    if i < num_steps:
        p.value = pressure[i]

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [])

    # Solve linear problem
    solver.solve(b, displacement.vector)
    displacement.x.scatter_forward()

    displacement_exact.interpolate(u_exact)

    error_table[i, :] = error_calculation(displacement_exact.x.array.reshape(-1, 3), displacement.x.array.reshape(-1, 3))

    error_L2 = MPI.COMM_WORLD.allreduce(
        fe.fem.assemble.assemble_scalar(fe.fem.form(((displacement - displacement_exact))** 2 * ufl.dx)),
                                        op=MPI.SUM)**0.5

    exact_L2_norm = MPI.COMM_WORLD.allreduce(
        fe.fem.assemble.assemble_scalar(fe.fem.form((displacement_exact)**2 * ufl.dx)),
        op=MPI.SUM)**0.5

    relative_error = error_L2 / exact_L2_norm

    if proc == 0:
        error.append(error_L2)
        rel_error.append(relative_error)
        radial_disp_1.append(np.linalg.norm(displacement.x.array.reshape((-1, 3))[index_of_radius_to_check[0], :]))
        radial_disp_2.append(np.linalg.norm(displacement.x.array.reshape((-1, 3))[index_of_radius_to_check[1], :]))
        # print(np.linalg.norm(disp))

    # Update old fields with new quantities
    update_fields(displacement, old_u, old_velocity, old_acceleration)
    u_exact.step_in_time()


    if visualize == True:
        warped["u"][:, :len(displacement)] = displacement.x.array.reshape(geometry.shape[0], len(displacement))
        us = fe.fem.Expression(ufl.sqrt(sum([displacement[i]**2 for i in range(len(displacement))])), Vs.element.interpolation_points())
        magnitude.interpolate(us)
        warped.set_active_scalars("mag")
        warped_n = warped.warp_by_vector(factor=10)
        plotter.update_coordinates(warped_n.points.copy(), render=False)
        plotter.update_scalar_bar_range([0, 10])
        plotter.update_scalars(magnitude.x.array)
        plotter.write_frame()

# mesh.comm.allreduce()
if visualize == True:
    plotter.close()


num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

with open(os.path.join(output_file, 'info.txt'), 'w') as f:
    f.write(f'Pressure Frequency (gamma): {gamma_p} \n')
    f.write(f'Time integration with alpha_m = {alpha_m.value}, alpha_f = {alpha_f.value}, gamma = {gamma.value}, beta = {beta.value}. \n')
    f.write(f'Modulous of Elasticity: {E} \n')
    f.write(f'Poission Ratio: {nu} \n')
    f.write(f'Density: {rho} \n')
    f.write(f'Maximum pressure: {maximum_pressure} \n')
    f.write(f'DOF number: {num_dofs_global} \n')

pd_file = pd.DataFrame(error_table)
pd_file.to_csv(os.path.join(output_file, f"relative_error_for_proc_{rank}.csv"), header=["x", 'y', 'z', 'norm'], index_label='time step')
# np.savetxt(os.path.join(output_file, f"relative_error_for_proc_{rank}.csv"), error_table, delimiter = ",", header=)


if proc == 0:
    with open(os.path.join(output_file, 'L2_error.json'), 'w') as f:
        json.dump(error, f)
    with open(os.path.join(output_file, 'relative_error.json'), 'w') as f:
        json.dump(rel_error, f)
    with open(os.path.join(output_file, 'radial_disp_1.json'), 'w') as f:
        json.dump(radial_disp_1, f)
    with open(os.path.join(output_file, 'radial_disp_2.json'), 'w') as f:
        json.dump(radial_disp_2, f)

    plt.plot(np.linspace(0, extended_time_ratio, int(num_steps * extended_time_ratio)), radial_disp_1, label=f'calculated_radius_1')
    plt.plot(np.linspace(0, extended_time_ratio, int(num_steps * extended_time_ratio)), radial_disp_2, label=f'calculated_radius_2')
    plt.xlabel('time(s)')
    plt.ylabel('displacement(m)')
    plt.savefig(os.path.join(output_file ,"radial_disp_without_legend.jpg"))
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(output_file ,"radial_disp.jpg"))

MPI.COMM_WORLD.barrier()


