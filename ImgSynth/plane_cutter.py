import trimesh
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import dolfinx as fe
from mpi4py import MPI
import meshio
import cv2
import math
import scipy.signal as sig
from unused_function import boundary_compeletion
import warnings

import sys
sys.path.append("/Users/korosh/anaconda3/pkgs/petsc-3.20.6-real_he4ff8f5_100/lib/petsc/bin")
petsc_dir = "/Users/korosh/anaconda3/pkgs/petsc-3.20.6-real_he4ff8f5_100" 
os.environ["PETSC_DIR"] = petsc_dir
import PetscBinaryIO

sys.path.append(os.path.join("..", "Mesh"))
from boundary_selection import longitudinal_direction_detection


def mean_of_myo(img, foreground_value):
  foreground_pxls = np.argwhere(img==foreground_value)
  mean = np.mean(foreground_pxls, axis=0)
  return mean


def pix_dist_to_mean(pixel, mean):
  return np.linalg.norm(pixel - mean)


def flood_fill_image(image, seed_point, background_value, foreground_value):
  """Fills the interior of a binary image starting from a seed point.

  Args:
      image: The binary image (numpy array).
      seed_point: A tuple (x, y) representing the seed point coordinates.
      background_value: The value considered background (e.g., 0).
      foreground_value: The value assigned to the filled region (e.g., 255).

  Returns:
      A new image with the interior filled with the foreground value.
  """
  cv2.floodFill(image, None, seed_point, foreground_value, cv2.FLOODFILL_FIXED_RANGE, cv2.FLOODFILL_FIXED_RANGE, background_value)
  return image


def plane_spacing(cood, logitudinal_direction, number_of_planes):
  minimum_length = float('inf')
  maximum_length = -float('inf')
  for node in cood:
    x = np.dot(node, logitudinal_direction)
    if x < minimum_length:
      minimum_length = x
      minimum_cood = node
    if x > maximum_length:
      maximum_length = x
      maximum_cood = node

  myo_length = maximum_length - minimum_length
  slice_thickness = myo_length / number_of_planes
  plane_origin = minimum_cood + slice_thickness/2 * logitudinal_direction
  heights = [(n)*slice_thickness for n in range(number_of_planes)]
  return plane_origin, heights, slice_thickness, myo_length

def strain_on_plane(strain, pixel, to_3d, center_of_myocardium, normal_to_plane, transformation, scale):
  center_of_myocardium_3d = np.dot(to_3d, np.asarray([center_of_myocardium[0], center_of_myocardium[1], 0, 1]))[:3]
  pixel_cood_2d = pixel / scale + transformation
  pixel_cood_3d = np.dot(to_3d, np.asarray([pixel_cood_2d[0], pixel_cood_2d[1], 0, 1]))[:3]
  
  try:
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      radial_vector_3d = (pixel_cood_3d - center_of_myocardium_3d)/np.linalg.norm(pixel_cood_3d - center_of_myocardium_3d)
  except:
    print("CHECK OUT HERE: strain_on_plane")
    radial_vector_3d = (pixel_cood_3d - center_of_myocardium_3d)/(np.linalg.norm(pixel_cood_3d - center_of_myocardium_3d) + 1e-5)
  circumfrential_vector_3d = np.cross(radial_vector_3d, normal_to_plane)

  strain_radial = np.dot(strain, radial_vector_3d)
  strian_circumfrential = np.dot(strain, circumfrential_vector_3d)
  strain_longitudinal = np.dot(strain, normal_to_plane)

  return strain_radial, strian_circumfrential, strain_longitudinal


def node_to_voxel_mapper(nodes_coordinate, normal_to_image_plane, point_on_plane, plane_spacing, slice_coverage, to_3D, transformation_on_image_plane, scale):
  corresponding_pixels_for_nodes = {}
  corresponding_nodes_for_pixels = {}

  counter = 0

  for node_number, node in enumerate(nodes_coordinate):
    v = node - point_on_plane
    dist = np.dot(v, normal_to_image_plane)
    projected_point = node - dist * normal_to_image_plane

    if (abs(abs(dist)/plane_spacing - round(abs(dist)/plane_spacing))) * plane_spacing < slice_coverage:
      corresponding_plane_number = round(abs(dist)/plane_spacing)
    else:
      counter += 1
      continue

    if corresponding_plane_number not in corresponding_pixels_for_nodes.keys():
      corresponding_pixels_for_nodes[corresponding_plane_number] = {}
      corresponding_nodes_for_pixels[corresponding_plane_number] = {}

    try:
      to_2d = np.linalg.inv(to_3D[corresponding_plane_number])
    except:
      to_2d = np.linalg.inv(to_3D[corresponding_plane_number-1])
    projected_point = trimesh.transformations.transform_points(projected_point.reshape((-1, 3)), to_2d)
    [i, j] = np.squeeze(projected_point[:, :2] * scale  - transformation_on_image_plane )    ##### scale = 256/larg_dist
    corresponding_pixels_for_nodes[corresponding_plane_number][node_number]= np.asarray([round(i), round(j)])
    corresponding_nodes_for_pixels[corresponding_plane_number][(i,j)]= node_number  # np.array([i, j])
  # print(f"{counter} nodes not included in strain_image.")
  return corresponding_pixels_for_nodes, corresponding_nodes_for_pixels


def main(mesh_file, strain_file, stl_mesh_file, number_of_slices, time_step, output_path):

  with fe.io.XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
    dolfinx_mesh = xdmf.read_mesh(name="Grid")
  # with fe.io.XDMFFile(MPI.COMM_WORLD, os.path.join("..", "myo_simulation", "output", "strain.xdmf"), "r") as xdmf:
  #   dolfinx_mesh = xdmf.read_mesh(name="Grid")
  #   # cell_markers = xdmf.read_meshtags(dolfinx_mesh, name="Grid")

  gdim = dolfinx_mesh.topology.dim
  nodes = dolfinx_mesh.geometry.x

  io = PetscBinaryIO.PetscBinaryIO(complexscalars=True)
  strain = io.readBinaryFile(strain_file)[0].reshape(-1, 3, 3)

  tri_mesh = trimesh.load(stl_mesh_file)

  plane_normal = longitudinal_direction_detection(nodes)
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

  margin = 15
  col_dist = maximum_y - minimum_y + margin
  row_dist = maximum_x - minimum_x + margin

  larger_dist = max([row_dist, col_dist])

  image_size = (256, 256)
  background_value = 0
  foreground_value = 255

  centering = 5
  transformation = [minimum_x - centering, minimum_y - centering]

  #####I must find a point on each plane instead of plane_origin
  node_voxel_correspondence = node_to_voxel_mapper(nodes, plane_normal, plane_origin, slice_thickness, myo_length/50, to_3d, transformation, 256/larger_dist)

  segmented_image = np.zeros((len(lines), 256, 256))
  radial_strain_img = np.zeros((len(lines), 256, 256))
  circumfrential_strain_img = np.zeros((len(lines), 256, 256))

  for n, line_in_slice in enumerate(lines):
    image = np.full(image_size, background_value, dtype=np.uint8)

    for points in line_in_slice:
      for p in points:
        [i, j] = (p - transformation) * 256 / larger_dist
        image[round(i), round(j)] = foreground_value

    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    # for iterate in range(100):
    #   image = boundary_compeletion(image, foreground_value)

    # plt.imshow(image)
    # plt.show()

    # if n == 1:
    #   print(image[200:205, 45:52])

    seed_point = (60, 60)     ##### An interactive method for choosing the seed point is needed

    cv2.floodFill(image, None, seed_point, foreground_value)


    # image_copy = image.copy()

    # def on_mouse_click(event, x, y, flags, param):
    #   if event == cv2.EVENT_LBUTTONDOWN:
    #     seed_point = (x, y)
    #     filled_image = flood_fill_image(image_copy, seed_point, background_value, foreground_value)  # Assuming foreground is 255
    #     cv2.imshow("Image", filled_image)

    # cv2.namedWindow("Image")
    # cv2.setMouseCallback("Image", on_mouse_click)

    # cv2.imshow("Image", image_copy)
    # cv2.waitKey(0)  # Wait for user input (e.g., any key press)
    # cv2.destroyAllWindows()


    segmented_image[n, :, :] = cv2.erode(image, kernel)

    myo_mean = mean_of_myo(segmented_image[n, :, :], foreground_value)

    # plt.imshow(segmented_image[n, :, :], cmap='gray')
    # plt.imsave(os.path.join(output_path, f"time_step_{time_step}_segmented_slice_{n}.jpg"), segmented_image[n, :, :], cmap='gray')
    im = Image.fromarray(segmented_image[n, :, :].astype(np.uint8))
    im.save(os.path.join(output_path, f"time_step_{time_step}_segmented_slice_{n}.png"))

    myo_center_pixel = np.mean(np.argwhere(image==foreground_value), axis=0)
    center_of_myocardium = myo_center_pixel / 256 * larger_dist + transformation

    pixel_dist = {}
    for node_number in node_voxel_correspondence[n].keys():       ####Check if the n conforms to the node number used in node_to_voxel_mapper function
      [p, q] = node_voxel_correspondence[n][node_number]
      radial, circumfrential, _ = strain_on_plane(strain[node_number, :, :], np.asarray([p, q]), to_3d[n, :, :], center_of_myocardium, plane_normal, transformation, larger_dist)
      radial_strain_img[n, p, q] = np.linalg.norm(radial)
      circumfrential_strain_img[n, p, q] += np.linalg.norm(circumfrential)
      if pix_dist_to_mean(np.asarray([p, q]), np.asarray(myo_mean)) not in pixel_dist:
        pixel_dist[pix_dist_to_mean(np.asarray([p, q]), np.asarray(myo_mean))] = 0
      pixel_dist[pix_dist_to_mean(np.asarray([p, q]), np.asarray(myo_mean))] += np.linalg.norm(circumfrential)
      # print(pixel_dist)
    
    # plt.hist(pixel_dist.keys(), bins=30, color='skyblue', edgecolor='black')
    # plt.show()

    if n in [3, 4, 5]:
      plt.scatter(pixel_dist.keys(), pixel_dist.values())
      plt.show()
    
    radial_strain_img[n, :, :] = radial_strain_img[n, :, :]/np.max(radial_strain_img[n, :, :])*255
    circumfrential_strain_img[n, :, :] = circumfrential_strain_img[n, :, :]/np.max(circumfrential_strain_img[n, :, :])*255

    mean_kernel = np.ones((5,5),np.float32)/25
    radial_strain_img[n, :, :] = cv2.filter2D(radial_strain_img[n, :, :],-1, mean_kernel)
    radial_strain_img[n, :, :] = cv2.filter2D(radial_strain_img[n, :, :],-1, mean_kernel)

    # plt.imshow(radial_strain_img[n, :, :], cmap='gray')
    # plt.imsave(os.path.join(output_path, f"time_step_{time_step}_radial_strain_slice_{n}.jpg"), radial_strain_img[n, :, :], cmap='gray')
    im = Image.fromarray(radial_strain_img[n, :, :].astype(np.uint8))
    im.save(os.path.join(output_path, f"time_step_{time_step}_radial_strain_slice_{n}.png"))    
    # plt.imshow(circumfrential_strain_img[n, :, :], cmap='gray')
    # plt.imsave(os.path.join(output_path, f"time_step_{time_step}_circumfrential_strain_slice_{n}.jpg"), radial_strain_img[n, :, :], cmap='gray')
    im = Image.fromarray(circumfrential_strain_img[n, :, :].astype(np.uint8))
    im.save(os.path.join(output_path, f"time_step_{time_step}_circumfrential_strain_slice_{n}.png")) 


if __name__ == "__main__":
  number_of_time_steps = 10
  number_of_slices = 7

  root_dir = os.path.join("..","myo_simulation","output")


##################################
  i = 2
  mesh_file = os.path.join(root_dir, '0012', f"warped_mesh_time_step_{i}.xdmf")
  strain_file = os.path.join(root_dir, '0012', f"strain_time_step_{i}.dat")
  stl_mesh_file = os.path.join(root_dir, '0012', f"warped_mesh_time_step_{i}.stl")

  output_path = 'output'

  main(mesh_file, strain_file, stl_mesh_file, number_of_slices, i, output_path)

'''  
  case = []
  for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
      case.append(folder_path)

      output_path = os.path.join("output", "synth_image_"+folder)

      if not os.path.exists(output_path):
        os.mkdir(output_path)
  

      for i in range(1, number_of_time_steps):
        mesh_file = os.path.join(folder_path, f"warped_mesh_time_step_{i}.xdmf")
        strain_file = os.path.join(folder_path, f"strain_time_step_{i}.dat")
        stl_mesh_file = os.path.join(folder_path, f"warped_mesh_time_step_{i}.stl")

        main(mesh_file, strain_file, stl_mesh_file, number_of_slices, i, output_path)
'''


