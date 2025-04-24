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
import time
import random

from natural_freq import natural_freq
# print(ans)

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


def generate_point_on_outer_surface(a, b, c):
    u = np.random.uniform(0, 2 * np.pi)  # Random azimuthal angle
    v = np.random.uniform(np.arccos(0.75), np.arccos(0.125))  # Random polar angle (z > 0)
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    p = np.array([x, y, z])
    print(p)
    return p


start_time = time.time()

proc = MPI.COMM_WORLD.rank
proc_size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

parser = argparse.ArgumentParser(description="Parameters of ellipsoid.")
parser.add_argument('-maxp', '--maximum_pressure', type=float, default=16000.0)
parser.add_argument('-minp', '--minimum_pressure', type=float, default=0.0)
parser.add_argument('-E', '--elasticity', type=int, default=10000)
parser.add_argument('-r', '--rho', type=float, default=1000.0)
parser.add_argument('-n', '--nu', type=float, default=0.3)
parser.add_argument('-ed', '--element_degree', type=int, default=1)
parser.add_argument('-sn', '--step_numbers', type=int, default=100)
parser.add_argument('-dt', '--dtype', type=type, default=np.float64) 
parser.add_argument('-v', '--visualize', type=bool, default=False)
parser.add_argument('-ac', '--anomaly_center', type=str, default='0 0 0')
parser.add_argument('-ar', '--anomaly_radius', type=float, default=0.2)
parser.add_argument('-Ear', '--elasticity_anomaly_ratio', type=float, default=1)
parser.add_argument('-m', '--mesh', type=str, default='geom_test/ellip.msh')
parser.add_argument('-o', '--output_file', type=str, default='test')

args = parser.parse_args()

mesh_file = os.path.join("..", "Mesh", "Output", "ellipsoid", args.mesh) 
output_file = 'case_' + str(args.output_file)

if proc == 0  and not os.path.exists(os.path.join("output", 'ellipsoid', output_file)):
    os.makedirs(os.path.join("output", 'ellipsoid', output_file))

output_file = os.path.join("output", 'ellipsoid', output_file) # 'ellipsoid', output_file

maximum_pressure = fe.default_scalar_type(args.maximum_pressure)
minimum_pressure = fe.default_scalar_type(args.minimum_pressure)
rho = fe.default_scalar_type(args.rho)
E = fe.default_scalar_type(args.elasticity)
nu = fe.default_scalar_type(args.nu)
element_degree = args.element_degree
num_steps = args.step_numbers
dtype = args.dtype
visualize = args.visualize
anomaly_radius = args.anomaly_radius
data_strings = args.anomaly_center.split(' ')
anomaly_center = np.array(list(map(lambda x: float(x), data_strings)))
E_anomaly = fe.default_scalar_type(args.elasticity_anomaly_ratio * E)

count = 0

for m in ['geom_1', 'geom_2', 'geom_3', 'geom_4', 'geom_5', 'geom_6']: # 
    for p in [0.3 , 0.4, 0.2]:
        for r in [950, 1000, 1050, 900, 1100]:
            for mp in [11000, 15000, 18000]:
                # for ac in ['-0.0332 0.0124 0.0323', '0.0260 0.0289 0.0095', '0.0172 -0.0150 0.0611']:
                # for ar in [0.004, 0.005]:
                for aer in [0.5, 0.7, 1, 1.5, 3.5, 0.3, 4, 4.5]:

                    count += 1

                    # if count < 510:
                    #     MPI.COMM_WORLD.barrier()
                    #     continue

                    E = fe.default_scalar_type(12000)
                    mesh_file = os.path.join("..", "Mesh", "Output", "ellipsoid", m, 'ellip_128.msh') 
                    output_file = 'case_' + str(count)

                    if proc == 0  and not os.path.exists(os.path.join("output", 'ellipsoid', 'sixth_try', output_file)):
                        os.makedirs(os.path.join("output", 'ellipsoid', 'sixth_try', output_file))

                    MPI.COMM_WORLD.barrier()

                    output_file = os.path.join("output", 'ellipsoid', 'sixth_try', output_file)

                    nu = fe.default_scalar_type(p)
                    rho = fe.default_scalar_type(r)
                    maximum_pressure = fe.default_scalar_type(mp)

                    # data_strings = ac.split(' ')
                    # anomaly_center = np.array(list(map(lambda x: float(x), data_strings)))

                    anomaly_center = np.zeros((3, 1))
                    anomaly_radius = 0

                    if proc == 0:
                        info_file = os.path.join("..", "Mesh", "Output", "ellipsoid", m, 'info.txt') 
                        with open(info_file, 'r') as f:
                            lines = f.readlines()
                            a = lines[0].strip()
                            if a.startswith('semi-axis length along x:'):
                                a = a.split(":")[1].strip()
                            c = lines[2].strip()
                            if c.startswith('semi-axis length along z (vertical):'):
                                c = c.split(":")[1].strip()
                            t = lines[3].strip()
                            if t.startswith('Wall thickness:'):
                                t = t.split(":")[1].strip()

                        anomaly_center = generate_point_on_outer_surface(float(a), float(a), float(c))
                        anomaly_radius = float(t) * random.uniform(1.1, 1.5)

                    MPI.COMM_WORLD.barrier()

                    anomaly_center = MPI.COMM_WORLD.bcast(anomaly_center, root=0)
                    anomaly_radius = MPI.COMM_WORLD.bcast(anomaly_radius, root=0)

                    E_anomaly = fe.default_scalar_type(aer * E)

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
                        # apex = create_mesh(msh, 'vertex')
                        facets = create_mesh(msh, "triangle", prune_z=False)
                        meshio.write(os.path.join(output_file, "mt.xdmf"), facets)
                        meshio.write(os.path.join(output_file, "mesh.xdmf"), tetra_mesh)
                        # meshio.write(os.path.join(output_file, "pt.xdmf"), apex)

                    MPI.COMM_WORLD.barrier()

                    #Reading the xdmf files
                    with fe.io.XDMFFile(MPI.COMM_WORLD, os.path.join(output_file, "mesh.xdmf"), "r") as xdmf:
                        mesh = xdmf.read_mesh(name="Grid")
                        cell_markers = xdmf.read_meshtags(mesh, name="Grid")

                    gdim = mesh.topology.dim
                    mesh.topology.create_connectivity(gdim, gdim - 1)
                    with fe.io.XDMFFile(MPI.COMM_WORLD, os.path.join(output_file, "mt.xdmf"), "r") as xdmf:
                        facet_markers = xdmf.read_meshtags(mesh, name="Grid")

                    # with fe.io.XDMFFile(MPI.COMM_WORLD, os.path.join(output_file, "mt.xdmf"), "r") as xdmf:
                    #     facets = xdmf.read_mesh(name="Grid")

                    # gdim = mesh.topology.dim
                    # mesh.topology.create_connectivity(gdim, gdim - 2)
                    # with fe.io.XDMFFile(MPI.COMM_WORLD, os.path.join(output_file, "pt.xdmf"), "r") as xdmf:
                    #     apex_markers = xdmf.read_meshtags(mesh, name="Grid")
                    ######################################################################################################
                    #######################creating function space and boundary condition#################################
                    base_marker = 2
                    endo_marker = 1
                    element = ufl.VectorElement("CG", mesh.ufl_cell(), element_degree)
                    V = fe.fem.functionspace(mesh, element)

                    # apex = {}
                    # found_point = 0

                    # def find_points(coords):
                    #     apex = []
                    #     pnt = []
                    #     for i, p in enumerate(coords):
                    #         if p[0] == 0 and p[1] == 0 and p[2]:
                    #             apex.append(i)
                    #             pnt.append(p)
                    #     if len(apex) > 0:
                    #         return apex, pnt
                    #     else:
                    #         return None, None

                    # found_point, desired_point = find_points(mesh.geometry.x)

                    # mpi_data_distrubution = {'rank': rank
                    #                         ,'found_point': found_point
                    #                         ,'coordinates': desired_point}



                    # MPI.COMM_WORLD.barrier()

                    ###boundary conditions
                    base_facets = facet_markers.find(base_marker)
                    base_dofs = fe.fem.locate_dofs_topological(V.sub(2), 2, base_facets)
                    bc = fe.fem.dirichletbc(fe.default_scalar_type(0), base_dofs, V.sub(2))

                    # base_dofs = fe.fem.locate_dofs_topological(V, 2, base_facets)
                    # bc = fe.fem.dirichletbc(fe.default_scalar_type((dtype(0), dtype(0), dtype(0))), base_dofs, V)

                    # num_steps = 10
                    gamma_p =  2*np.pi    ################MIND THE GAMMA   
                    t = np.linspace(0, 2*np.pi / gamma_p, num_steps)
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
                    metadata = {"quadrature_degree": 2}   ### The quadrature degree has a great impact on the steady state problem, but apparantely none in the dynamic casse.
                    ds = ufl.Measure('ds', domain=mesh, subdomain_data=facet_markers, metadata=metadata)
                    dx = ufl.Measure("dx", domain=mesh, metadata=metadata)

                    # Generalized-alpha method parameters
                    alpha_m = fe.fem.Constant(mesh, 0.2)  #   0.2
                    alpha_f = fe.fem.Constant(mesh, 0.4)   #  0.4
                    gamma   = fe.fem.Constant(mesh, fe.default_scalar_type(0.5+alpha_f-alpha_m))
                    beta    = fe.fem.Constant(mesh, fe.default_scalar_type((gamma+0.5)**2/4.))

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

                    normal_to_facet = -ufl.FacetNormal(mesh)

                    # mu = fe.fem.Constant(mesh, E / (2 * (1 + nu)))
                    # lmbda = fe.fem.Constant(mesh, E * nu / ((1 + nu) * (1 - 2 * nu)))

                    mechanical_module = fe.fem.functionspace(mesh, ("DG", 0))

                    mu = fe.fem.Function(mechanical_module)      ### Poission's ratio can be chosen for anolmaly as well.
                    lmbda = fe.fem.Function(mechanical_module)

                    random_factors = 1/10 * np.random.rand(1,6)

                    def eval_mu(x):
                        center = np.repeat([anomaly_center], x.shape[1], axis=0)
                        values = np.zeros(x.shape[1], dtype=fe.default_scalar_type)
                        # Create a boolean array indicating which dofs (corresponding to cell centers)
                        # that are in each domain
                        anomaly_coords = np.linalg.norm(x - center.T, axis=0) < anomaly_radius
                        normal_coords = np.linalg.norm(x - center.T, axis=0) > anomaly_radius
                        values[anomaly_coords] = np.full(sum(anomaly_coords), E_anomaly / (2 * (1 + nu)))
                        values[normal_coords] = np.full(sum(normal_coords), E / (2 * (1 + nu)))
                        # values += 1/50 * E * np.sin(random_factors[0, 0] * x[0] + random_factors[0, 1] * x[1] + random_factors[0, 2] * x[2])
                        return values

                    def eval_lmbda(x):
                        center = np.repeat([anomaly_center], x.shape[1], axis=0)
                        values = np.zeros(x.shape[1], dtype=fe.default_scalar_type)
                        # Create a boolean array indicating which dofs (corresponding to cell centers)
                        # that are in each domain
                        anomaly_coords = np.linalg.norm(x - center.T, axis=0) < anomaly_radius
                        normal_coords = np.linalg.norm(x - center.T, axis=0) > anomaly_radius
                        values[anomaly_coords] = np.full(sum(anomaly_coords), E_anomaly * nu / ((1 + nu) * (1 - 2 * nu)))
                        values[normal_coords] = np.full(sum(normal_coords), E * nu / ((1 + nu) * (1 - 2 * nu)))
                        # values += 1/50 * E * np.sin(random_factors[0, 3] * x[0] + random_factors[0, 4] * x[1] + random_factors[0, 5] * x[2])
                        return values

                    mu.interpolate(eval_mu)
                    lmbda.interpolate(eval_lmbda)


                    def epsilon(u):
                        return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

                    def S(u):
                        return 2.0 * mu * epsilon(u) + lmbda * ufl.nabla_div(u) * I  # ufl.sym(ufl.grad(u)) # ufl.tr(ufl.sym(ufl.grad(u)))

                    acceleration = update_a(u, old_u, old_velocity, old_acceleration, ufl=False)    #### Changing the ufl to True would not work. It makes the acceleration type not suitable to be added to other variables in the formulation!
                    velocity = update_v(acceleration, old_u, old_velocity, old_acceleration, ufl=True)

                    formulation = rho * ufl.inner(avg(old_acceleration, acceleration, alpha_m), v) * dx \
                      + ufl.inner(epsilon(v), S(avg(old_u, u, alpha_f))) * dx \
                      - ufl.dot(v, p * normal_to_facet) * ds(endo_marker)     #  ufl.inner(v, p * normal)

                    bilinear_form = fe.fem.form(ufl.lhs(formulation))
                    linear_form = fe.fem.form(ufl.rhs(formulation))

                    ######################################################################################################
                    ######################################################################################################

                    ######################################################################################################
                    ###############################################Solving################################################
                    A = assemble_matrix(bilinear_form, bcs=[bc])
                    A.assemble()
                    b = create_vector(linear_form)

                    solver = PETSc.KSP().create(mesh.comm)
                    solver.setInitialGuessNonzero(True)
                    solver.setOperators(A)
                    solver.getPC().setType(PETSc.PC.Type.SOR)
                    # fe.log.set_log_level(fe.log.LogLevel.INFO)
                    # solver.view()

                    displacement = fe.fem.Function(V, dtype=dtype)
                    points = mesh.geometry.x

                    ###############Strain#################
                    el_strain = ufl.TensorElement("Lagrange", mesh.ufl_cell(), 1, shape=(3, 3))
                    Q_strain = fe.fem.FunctionSpace(mesh, el_strain)
                    lagrange_finite_strain = ufl.dot((ufl.Identity(d) + ufl.grad(displacement)).T, ufl.Identity(d) + ufl.grad(displacement)) - ufl.Identity(d)
                    strain_expr = fe.fem.Expression(lagrange_finite_strain, Q_strain.element.interpolation_points())
                    strain = fe.fem.Function(Q_strain, dtype=dtype)     #### 2E = C - I
                    strain.name = "lagrange_finite_strain"

                    if visualize == True:
                        ##############Initializing the plot###################
                        pyvista.start_xvfb()
                        plotter = pyvista.Plotter()
                        plotter.open_gif(os.path.join(output_file, f"deformation_proc_{proc}.gif"), fps=3)

                        topology, cell_types, geometry = fe.plot.vtk_mesh(mesh, 3)
                        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

                        values = np.zeros((geometry.shape[0], 3))
                        values[:, :len(u)] = displacement.x.array.astype(float).reshape(geometry.shape[0], len(displacement))
                        grid.point_data["u"] = values
                        grid.set_active_vectors("u")

                        # Warp mesh by deformation
                        warped = grid.warp_by_vector("u", factor=1)
                        warped.set_active_vectors("u")

                        renderer = plotter.add_mesh(warped, show_edges=True, lighting=False, clim=[0, 1])

                        Vs = fe.fem.FunctionSpace(mesh, ("Lagrange", 1))
                        magnitude = fe.fem.Function(Vs)
                        us = fe.fem.Expression(ufl.sqrt(sum([displacement[i]**2 for i in range(len(displacement))])), Vs.element.interpolation_points())
                        magnitude.interpolate(us)
                        warped["mag"] = magnitude.x.array

                        plotter.write_frame()

                    io = PetscBinaryIO.PetscBinaryIO(complexscalars=True)

                    #Here the correspondence between nodes of the dolfinx mesh and the facets are found for later.
                    ### facets is needed for saving the stl files.
                    points = mesh.geometry.x

                    if proc == 0:
                        list2 = facets.points #geometry.x
                    else:
                        list2 = None
                    
                    MPI.COMM_WORLD.barrier()
                    list2 = MPI.COMM_WORLD.bcast(list2, root=0)

                    kdtree = cKDTree(points)
                    correspondence_btw_facets_points_and_mesh = []     ### (index of dolfinx mesh, index of facets mesh)
                    for i, cell in enumerate(list2):
                        dist, idx = kdtree.query(cell)
                        if dist == 0:
                            correspondence_btw_facets_points_and_mesh.append((i, idx))

                    # Pairwise comparison to remove duplicates
                    for other_rank in range(proc_size):
                        if other_rank != rank:
                            # Send and receive lists with other processor
                            other_numbers = MPI.COMM_WORLD.sendrecv(correspondence_btw_facets_points_and_mesh, dest=other_rank, source=other_rank)

                            # Identify shared numbers
                            shared_numbers = set([i[0] for i in correspondence_btw_facets_points_and_mesh]).intersection([i[0] for i in other_numbers])

                            # Remove shared numbers only if this process has the higher rank
                            if rank > other_rank:
                                correspondence_btw_facets_points_and_mesh = [num for num in correspondence_btw_facets_points_and_mesh if num[0] not in shared_numbers]

                    # apex_disp = {}
                    # if mpi_data_distrubution['found_point'] is not None:
                    #     num_dt = len(mpi_data_distrubution['found_point'])
                    #     for j in range(num_dt):
                    #         apex_disp[j] = []


                    # Stepping through time and solving for displacements and strains
                    for i in range(int(num_steps)): #  * 3/2
                        # print(i)
                        if i < num_steps:
                            p.value = pressure[i]

                        # Update the right hand side reusing the initial vector
                        with b.localForm() as loc_b:
                            loc_b.set(0)
                        assemble_vector(b, linear_form)

                        # Apply Dirichlet boundary condition to the vector
                        apply_lifting(b, [bilinear_form], [[bc]])
                        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
                        set_bc(b, [bc])
                        
                        # Solve linear problem
                        solver.solve(b, displacement.vector)
                        displacement.x.scatter_forward()
                        # print(displacement.x.array[:])

                        if i % 10 == 0:
                            displacementAsPetscBiIOVec = displacement.vector.array_w.view(PetscBinaryIO.Vec)
                            io.writeBinaryFile(os.path.join(output_file, f"displacement_time_step_{i}_proc_{rank}.dat"), [displacementAsPetscBiIOVec])
                            strain.interpolate(strain_expr)
                            strainAsPetscBiIOVec = strain.vector.array_w.view(PetscBinaryIO.Vec)
                            io.writeBinaryFile(os.path.join(output_file, f"strain_time_step_{i}_proc_{rank}.dat"), [strainAsPetscBiIOVec])

                        mesh.geometry.x[:, :mesh.geometry.dim] += displacement.x.array.reshape(points.shape)

                        # with fe.io.XDMFFile(mesh.comm, os.path.join(output_file, f"warped_mesh_time_step_{i}.xdmf"), "w") as xdmf:
                        #     xdmf.write_mesh(mesh)

                        mesh.geometry.x[:, :mesh.geometry.dim] -= displacement.x.array.reshape(points.shape)

                        for proc_num in range(proc_size):
                            list2[[i[0] for i in correspondence_btw_facets_points_and_mesh]] += displacement.x.array.reshape(points.shape)[[i[1] for i in correspondence_btw_facets_points_and_mesh]]
                            list2 = MPI.COMM_WORLD.bcast(list2, root=proc_num)

                        if proc == 0 and i % 10 == 0:
                            facets.points = list2
                            meshio.write(os.path.join(output_file, f"warped_facet_time_step_{i}.stl"), facets)

                        MPI.COMM_WORLD.barrier()
                        for proc_num in range(proc_size):
                            list2[[i[0] for i in correspondence_btw_facets_points_and_mesh]] -= displacement.x.array.reshape(points.shape)[[i[1] for i in correspondence_btw_facets_points_and_mesh]]
                            list2 = MPI.COMM_WORLD.bcast(list2, root=proc_num)

                        if proc == 0:
                            facets.points = list2

                        MPI.COMM_WORLD.barrier()

                        # if mpi_data_distrubution['found_point'] is not None:
                        #     num_dt = len(mpi_data_distrubution['found_point'])
                        #     for j in range(num_dt):
                        #         apex_disp[j].append(np.linalg.norm(displacement.x.array.reshape((-1, 3))[mpi_data_distrubution['found_point'][j], :]))
                        #     # apex_2_disp.append(np.linalg.norm(displacement.x.array.reshape((-1, 3))[mpi_data_distrubution['found_point'][1], :]))

                        # Update old fields with new quantities
                        update_fields(displacement, old_u, old_velocity, old_acceleration)

                        if visualize == True:
                            warped["u"][:, :len(displacement)] = displacement.x.array.reshape(geometry.shape[0], len(displacement))
                            us = fe.fem.Expression(ufl.sqrt(sum([displacement[i]**2 for i in range(len(displacement))])), Vs.element.interpolation_points())
                            magnitude.interpolate(us)
                            warped.set_active_scalars("mag")
                            warped_n = warped.warp_by_vector(factor=1)
                            plotter.update_coordinates(warped_n.points.copy(), render=False)
                            plotter.update_scalar_bar_range([0, 10])
                            plotter.update_scalars(magnitude.x.array)
                            plotter.write_frame()

                    # mesh.comm.allreduce()
                    if visualize == True:
                        plotter.close()

                    end_time = time.time()

                    num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

                    if proc == 0:
                        elapsed_time = end_time - start_time
                        print(f"Execution time: {elapsed_time:.2f} seconds")
                        with open(os.path.join(output_file, 'info.txt'), 'w') as f:
                            f.write(f'Mesh: {m} \n')
                            f.write(f'Modulus of Elasticity: {E} \n')
                            f.write(f'Poisson Ratio: {nu} \n')
                            f.write(f'Density: {rho} \n')
                            f.write(f'Maximum pressure: {maximum_pressure} \n')
                            f.write(f'Minimum pressure: {minimum_pressure} \n')
                            f.write(f'Element degree: {args.element_degree} \n')
                            f.write(f'Time step number: {args.step_numbers} \n')
                            f.write(f'Anomaly center: {anomaly_center} \n')
                            f.write(f'Anomaly radius: {anomaly_radius} \n')
                            f.write(f'Elasticity anomaly ratio: {aer} \n')


                    # np.save(os.path.join(output_file, f'point_coord_for_proc_{rank}'), points)

                    # if mpi_data_distrubution['found_point'] is not None:
                    #     num_dt = len(mpi_data_distrubution['found_point'])
                    #     for j in range(num_dt):
                    #         plt.plot(t, apex_disp[j], label=f'mesh_{mpi_data_distrubution["coordinates"][j]}')
                        
                    #     comsol_derived = np.loadtxt("/Users/korosh/Desktop/COMSOL/ellipsoid_normal_mesh.txt", dtype=np.float64, skiprows=8) 

                    #     plt.plot(comsol_derived[:,0], comsol_derived[:,1], label='comsol_derived_dist_2.6')
                    #     plt.plot(comsol_derived[:,0], comsol_derived[:,2], label='comsol_derived_dist_3')    

                    #     plt.legend()
                    #     # plt.show()
                    #     plt.savefig(os.path.join(output_file ,f"radial_disp_proc_{proc}.jpg"))


                    MPI.COMM_WORLD.barrier()

                    # plt.show()


