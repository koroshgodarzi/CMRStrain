import trimesh
import dolfinx as fe
import ufl
import meshio
import os
from mpi4py import MPI
import numpy as np
from plane_cutter import plane_spacing, node_to_voxel_mapper, mean_of_myo, strain_on_plane
import cv2
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import warnings
warnings.filterwarnings("ignore", category=np.ComplexWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys

sys.path.append("/Users/korosh/anaconda3/pkgs/petsc-3.20.6-real_he4ff8f5_100/lib/petsc/bin")
petsc_dir = "/Users/korosh/anaconda3/pkgs/petsc-3.20.6-real_he4ff8f5_100" 
os.environ["PETSC_DIR"] = petsc_dir
import PetscBinaryIO

sys.path.append(os.path.join("..", "Mesh"))
from boundary_selection import longitudinal_direction_detection

# def reading_the_data():
#     info_file = ''
#     with open(info_file, 'r') as f:
#         lines = f.readlines()
#         anomaly_line = lines[-1].strip()
#         if anomaly_line.startswith("Elasticity anomaly ratio:"):
#             anomaly_ratio = anomaly_line.split(":")[1].strip()

rank = MPI.COMM_WORLD.Get_rank()

number_of_time_steps = 6 
number_of_slices = 7

root_dir = os.path.join('..', 'myo_simulation', 'output', 'ellipsoid', 'fifth_try', 'first_part')

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):

        # with open(os.path.join(folder_path, 'info.txt'), 'r') as f:
        #     lines = f.readlines()
        #     print(lines[-3::])
            # if 'geom_1' in lines[0] or 'geom_4' in lines[0] or 'geom_7' in lines[0]:
            #     MPI.COMM_WORLD.barrier()
            #     continue

        output_path = os.path.join("output", 'fifth_try', "synth_image_"+folder)  # 

        if rank == 0 and not os.path.exists(output_path):
            os.makedirs(output_path)
        if rank == 0:
            shutil.copy(os.path.join(folder_path, "info.txt"), output_path)

        undeformed_mesh_file = os.path.join(folder_path, "mesh.xdmf")
        with fe.io.XDMFFile(MPI.COMM_WORLD, undeformed_mesh_file, "r") as xdmf:
            undeformed_mesh = xdmf.read_mesh(name="Grid")

        element = ufl.VectorElement("CG", undeformed_mesh.ufl_cell(), 1)
        V = fe.fem.functionspace(undeformed_mesh, element)
        displacement = fe.fem.Function(V, dtype=np.float64)

        el_strain = ufl.TensorElement("Lagrange", undeformed_mesh.ufl_cell(), 1, shape=(3, 3))
        Q_strain = fe.fem.FunctionSpace(undeformed_mesh, el_strain)
        strain = fe.fem.Function(Q_strain, dtype=np.float64)

        for time_step in range(number_of_time_steps):
            strain_file = os.path.join(folder_path, f"strain_time_step_{10*time_step}.dat")
            stl_mesh_file = os.path.join(folder_path, f"warped_facet_time_step_{10*time_step}.stl")

            io = PetscBinaryIO.PetscBinaryIO(complexscalars=True)
            strain.vector[...] = io.readBinaryFile(os.path.join(folder_path, f"strain_time_step_{10*time_step}_proc_{MPI.COMM_WORLD.rank}.dat"))
            displacement.vector[...] = io.readBinaryFile(os.path.join(folder_path, f"displacement_time_step_{10*time_step}_proc_{MPI.COMM_WORLD.rank}.dat"))

            points = undeformed_mesh.geometry.x

            gathered_points = MPI.COMM_WORLD.gather(points, root=0)
            gathered_strain = MPI.COMM_WORLD.gather(strain.x.array.reshape((-1, 3, 3)), root=0)
            gathered_displacement = MPI.COMM_WORLD.gather(displacement.x.array.reshape((-1, 3)), root=0)
            
            if rank == 0:
                # Concatenate arrays from all processes
                result = np.concatenate(gathered_points)
                unique_points, unique_indices = np.unique(result, axis=0, return_index=True)

                result = np.concatenate(gathered_strain, axis=0)
                unique_strain = result[unique_indices, :, :]

                result = np.concatenate(gathered_displacement, axis=0)
                unique_displacement = result[unique_indices, :]


            MPI.COMM_WORLD.barrier()


            if rank == 0:
                gdim = undeformed_mesh.topology.dim
                nodes = unique_points + unique_displacement

                tri_mesh = trimesh.load(stl_mesh_file)

                plane_normal = np.array([0, 0, 1]) + np.random.normal(0, 0.01, size=(3,))
                plane_origin, heights, slice_thickness, myo_length = plane_spacing(nodes, plane_normal, number_of_slices)

                lines, to_3d, face_indx = trimesh.intersections.mesh_multiplane(tri_mesh, plane_origin, plane_normal, heights)

                minimum_x = float('inf')
                minimum_y = float('inf')
                maximum_x = -float('inf')
                maximum_y = -float('inf')
                for l in lines:
                    minimum_x = min(minimum_x, np.min(np.min(l, axis=1), axis=0)[0])
                    minimum_y = min(minimum_y, np.min(np.min(l, axis=1), axis=0)[1])
                    maximum_x = max(maximum_x, np.max(np.max(l, axis=1), axis=0)[0])
                    maximum_y = max(maximum_y, np.max(np.max(l, axis=1), axis=0)[1])

                margin = 1.5 + 1/4 * np.random.rand()
                col_dist = (maximum_y - minimum_y) * margin
                row_dist = (maximum_x - minimum_x) * margin

                larger_dist = max([row_dist, col_dist])# + 100

                image_size = 224
                background_value = 0
                foreground_value = 255

                scale = image_size / larger_dist

                mean_x = (maximum_x + minimum_x)/2
                mean_y = (maximum_y + minimum_y)/2

                transformation = np.array([mean_x*scale - image_size/2, mean_y*scale - image_size/2]) + np.random.rand(2)

                # centering = 0
                # transformation = [minimum_x - centering, minimum_y - centering]

                #####I must find a point on each plane instead of plane_origin
                node_voxel_correspondence, voxel_node_correspondence = node_to_voxel_mapper(nodes, plane_normal, plane_origin, slice_thickness, slice_thickness/5, to_3d, transformation, scale)

                segmented_image = np.zeros((len(lines), image_size, image_size))
                radial_strain_img = np.zeros((len(lines), image_size, image_size))
                circumfrential_strain_img = np.zeros((len(lines), image_size, image_size))

                for n, line_in_slice in enumerate(lines):
                    image = np.full((image_size, image_size), background_value, dtype=np.uint8)

                    for points in line_in_slice:
                        start_point = np.round(points[0] * (scale) - transformation)
                        end_point = np.round(points[1] * (scale) - transformation)
                        # print(start_point)
                        # print(end_point)
                        image = cv2.line(image, start_point.astype(int), end_point.astype(int), foreground_value, 1)
                        # for p in points:
                        #     [i, j] = (p * (scale) - transformation)
                        #     image[round(i), round(j)] = foreground_value

                    first_flag = False
                    flag = False
                    d = 0
                    if n < number_of_slices-1:
                        try:
                            while not (flag * first_flag):
                                d += 1
                                if not first_flag and image[int(image_size/2) + d, int(image_size/2)] != background_value:
                                    pix_on_inner_boundary = np.array([int(image_size/2) + d, int(image_size/2)])
                                    first_flag = True

                                if not flag and image[image_size - d, int(image_size/2)] != background_value:
                                    pix_on_outer_boundary = np.array([image_size - d, int(image_size/2)])
                                    flag = True

                        except:
                            print(f'folder_{folder}_time_step_{time_step}_slice_{n}.')
                        seed_point = (pix_on_inner_boundary + pix_on_outer_boundary)/2

                        # seed_point = (int(image_size/2), int(image_size/2))
                        cv2.floodFill(image, None, [seed_point[1].astype(int), seed_point[0].astype(int)], foreground_value)
                    else:
                        cv2.floodFill(image, None, [int(image_size/2), int(image_size/2)], foreground_value)
                    segmented_image[n, :, :] = image
                    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

                    # segmented_image[n, :, :] = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=10)
                    # # segmented_image[n, :, :] = cv2.morphologyEx(segmented_image[n, :, :], cv2.MORPH_OPEN, kernel, iterations=1)                   

                    im = Image.fromarray(segmented_image[n, :, :].astype(np.uint8))
                    im.save(os.path.join(output_path, f"time_step_{time_step}_segmented_slice_{n}.png"))

                    # if time_step == 0:
                    #     seg_check = np.zeros((image_size, image_size, 3))
                    #     seg_check[:, :, 0] = image
                    #     seg_check[:, :, 1] = segmented_image[n, :, :]
                    #     seg_check[:, :, 2] = image
                    #     plt.imshow(seg_check)
                    #     plt.imsave(os.path.join(output_path, f'seg_check_slice_{n}.png'), seg_check/foreground_value)

                    myo_center_pixel = np.mean(np.argwhere(image==foreground_value), axis=0)
                    center_of_myocardium = myo_center_pixel / image_size * larger_dist + transformation

                    # counter = np.zeros((256, 256))

                    nodes_projected = list(voxel_node_correspondence[n].keys())
                    tree = cKDTree(nodes_projected)
                    for i, row in enumerate(segmented_image[n, :, :]):
                        for j, pixel in enumerate(row):
                            if pixel == foreground_value:
                                distances, indices = tree.query([i, j], k=4)
                                # if len(indices) == 0:
                                #     print('no nearby points!')

                                # weight_dists = [1/(np.linalg.norm(np.asarray([i, j]) - np.asarray(nodes_projected[x])) + 1e-5) for x in indices]
                                node_number = [voxel_node_correspondence[n][y] for y in [nodes_projected[x] for x in indices]]
                                weight_dists = [1/x for x in distances]

                                strain_for_this_pixel = np.zeros((3, 3))
                                total_weight = 0
                                for k in range(len(weight_dists)):
                                    strain_for_this_pixel += unique_strain[node_number[k], :, :] * weight_dists[k]
                                    total_weight += weight_dists[k]
                                # print(total_weight)
                                strain_for_this_pixel /= total_weight

                                radial, circumfrential, _ = strain_on_plane(strain_for_this_pixel, np.asarray([i, j]), to_3d[n, :, :], center_of_myocardium, plane_normal, transformation, scale)
                                radial_strain_img[n, i, j] = np.linalg.norm(radial)
                                circumfrential_strain_img[n, i, j] = np.linalg.norm(circumfrential)


                    # for node_number in node_voxel_correspondence[n].keys():       ####Check if the n conforms to the node number used in node_to_voxel_mapper function
                    #     [p, q] = node_voxel_correspondence[n][node_number]
                    #     radial, circumfrential, _ = strain_on_plane(unique_strain[node_number, :, :], np.asarray([p, q]), to_3d[n, :, :], center_of_myocardium, plane_normal, transformation, larger_dist)
                    #     radial_strain_img[n, p, q] += np.linalg.norm(radial)
                    #     circumfrential_strain_img[n, p, q] += np.linalg.norm(circumfrential)
                    #     counter[p, q] += 1

                    # counter = np.where(counter == 0, 1, counter)

                    # radial_strain_img[n, :, :] = np.divide(radial_strain_img[n, :, :], counter)
                    # circumfrential_strain_img[n, :, :] = np.divide(circumfrential_strain_img[n, :, :], counter)

                    # for k in [5, 9, 15]: #, 31, 51
                    #     radial_strain_img[n, :, :] = radial_strain_img[n, :, :]/np.max(radial_strain_img[n, :, :])*255
                    #     circumfrential_strain_img[n, :, :] = circumfrential_strain_img[n, :, :]/np.max(circumfrential_strain_img[n, :, :])*255

                    #     radial_strain_img[n, :, :] = cv2.blur(radial_strain_img[n, :, :].astype(np.uint8), ksize = (k,k)) # , sigmaX = 1
                    #     circumfrential_strain_img[n, :, :] = cv2.blur(circumfrential_strain_img[n, :, :].astype(np.uint8), ksize = (k,k)) # , sigmaX = 1

                    radial_strain_img[n, :, :] = radial_strain_img[n, :, :]/np.max(radial_strain_img[n, :, :])*foreground_value
                    circumfrential_strain_img[n, :, :] = circumfrential_strain_img[n, :, :]/np.max(circumfrential_strain_img[n, :, :])*foreground_value

                    # plt.imshow(circumfrential_strain_img[n, :, :], cmap='grey', label=f'slice_{n}')
                    # plt.show()

                    # mean_kernel = np.ones((5,5),np.float32)/25
                    # radial_strain_img[n, :, :] = cv2.filter2D(radial_strain_img[n, :, :],-1, mean_kernel)
                    # radial_strain_img[n, :, :] = cv2.filter2D(radial_strain_img[n, :, :],-1, mean_kernel)

                    im = Image.fromarray(radial_strain_img[n, :, :].astype(np.uint8))
                    im.save(os.path.join(output_path, f"time_step_{time_step}_radial_strain_slice_{n}.png"))    

                    im = Image.fromarray(circumfrential_strain_img[n, :, :].astype(np.uint8))
                    im.save(os.path.join(output_path, f"time_step_{time_step}_circumfrential_strain_slice_{n}.png")) 

            MPI.COMM_WORLD.barrier()

