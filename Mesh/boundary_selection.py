import gmsh
import numpy as np
import os
import vtk
from vtk.util import numpy_support as VN
import pickle
import itertools as itr
import matplotlib.pyplot as plt


def finding_normal_to_a_direction(direction):
    direction = direction / np.linalg.norm(direction)
    random_vec = np.random.rand(3)
    normal_vec = np.cross(direction, random_vec)
    
    # If the cross product is a zero vector (which happens if random_vec was collinear with direction),
    # try another random vector
    if np.linalg.norm(normal_vec) == 0:
        return random_normal_vector(direction)

    normal_vec = normal_vec / np.linalg.norm(normal_vec)
    return normal_vec


def anomaly_volume(center, radius):
    o = gmsh.model.geo.addPoint(center[0], center[1], center[2])

    p1 = gmsh.model.geo.addPoint(center[0], center[1], center[2] + radius)
    p2 = gmsh.model.geo.addPoint(center[0], center[1], center[2] - radius)
    p3 = gmsh.model.geo.addPoint(center[0] + radius, center[1], center[2])
    p4 = gmsh.model.geo.addPoint(center[0] - radius, center[1], center[2])
    p5 = gmsh.model.geo.addPoint(center[0], center[1] + radius, center[2])
    p6 = gmsh.model.geo.addPoint(center[0], center[1] - radius, center[2])

    a1 = gmsh.model.geo.addCircleArc(p3, o, p5)
    a2 = gmsh.model.geo.addCircleArc(p5, o, p4)
    a3 = gmsh.model.geo.addCircleArc(p4, o, p6)
    a4 = gmsh.model.geo.addCircleArc(p6, o, p3)
    a5 = gmsh.model.geo.addCircleArc(p5, o, p1)
    a6 = gmsh.model.geo.addCircleArc(p1, o, p6)
    a7 = gmsh.model.geo.addCircleArc(p6, o, p2)
    a8 = gmsh.model.geo.addCircleArc(p2, o, p5)

    loop1 = gmsh.model.geo.addCurveLoop([a4, a1, a5, a6])
    loop2 = gmsh.model.geo.addCurveLoop([a2, a3, -a6, -a5])
    loop3 = gmsh.model.geo.addCurveLoop([a4, a1, -a8, -a7])
    loop4 = gmsh.model.geo.addCurveLoop([a2, a3, a7, a8])

    s1 = gmsh.model.geo.addSurfaceFilling([loop1])
    s2 = gmsh.model.geo.addSurfaceFilling([loop2])
    s3 = gmsh.model.geo.addSurfaceFilling([loop3])
    s4 = gmsh.model.geo.addSurfaceFilling([loop4])

    surface_loop = gmsh.model.geo.addSurfaceLoop([s1, s2, s3, s4])
    volume = gmsh.model.geo.addVolume([surface_loop])

    return surface_loop


def anomaly_selection(coord, maximum_height_coord, minimum_height_coord, endo_coord, epi_coord):
    fraction = 0.5
    height_of_anomaly = minimum_height_coord + fraction * (maximum_height_coord - minimum_height_coord)
    longitudinal_direction = longitudinal_direction_detection(coord)
    normal_direction = finding_normal_to_a_direction(longitudinal_direction)

    two_points_on_myo = []

    for c in [endo_coord, epi_coord]:
        minimum_dist = float('INF')
        for i, point in enumerate(c):
            dist = distance_point_to_line(point, height_of_anomaly, normal_direction)
            if dist < minimum_dist:
                candidate = point
                candidate_num = i
        two_points_on_myo.append(candidate)
    
    center_of_anomaly = (two_points_on_myo[0] + two_points_on_myo[1])/2
    # center_of_anomaly = height_of_anomaly + normal_direction * r
    radius_of_anomaly = np.linalg.norm(two_points_on_myo[0] - two_points_on_myo[1])/2

    print(two_points_on_myo)
    print(height_of_anomaly)
    print(center_of_anomaly)
    print(radius_of_anomaly)

    return anomaly_volume(center_of_anomaly, radius_of_anomaly)

    # sphere = gmsh.model.occ.addSphere(center_of_anomaly[0], center_of_anomaly[1], center_of_anomaly[2], radius_of_anomaly)

    # print(center_of_anomaly)
    # print(radius_of_anomaly)

    # ent = gmsh.model.getEntities(3)
    # print(ent)

    # anomaly_volume_list = []

    # for e in ent:
    #     vol = e[1]
    #     tag, coord, param = gmsh.model.mesh.getNodes(3, vol, True)

    #     coord = np.reshape(coord, (-1, 3))
    #     print(coord.shape)
    #     vol_mean = np.mean(coord, axis=0)

    #     if np.linalg.norm(vol_mean - center_of_anomaly) < radius_of_anomaly:
    #         anomaly_volume_list.append(vol)

    # group = gmsh.model.geo.addPhysicalGroup(3, anomaly_volume_list, 7, name="anomaly")
    # return anomaly_volume_list



#####THE SIGN OF HEIGHT IS INDETERMINATE, (in the if statement, height > h or height < h)#####
def setting_apex(coord, first_point_tag):
    longitudinal_direction = longitudinal_direction_detection(coord[:, 1::])
    center = np.mean(coord[:, 1::], axis=0)
    h = 0
    for i, p in enumerate(coord):
        height = np.dot(p[1::] - center, longitudinal_direction)
        ####ATTENTION####
        if height > h:
            h = height
            apex = p[0]   ##i
            apex_coord = p[1::]

    group = gmsh.model.addPhysicalGroup(0, [apex + first_point_tag], 4, name="apex")
    return [apex + first_point_tag], apex_coord


def longitudinal_direction_detection(coord):
    covariance = np.cov(coord.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    longitudinal_direction = eigenvectors[:, np.argmax(eigenvalues)]
    return longitudinal_direction


def distance_point_to_line(point, point_on_line, line_direction):
    #constant of surface containing point and having line_direction as normal
    surface_constant = -np.dot(line_direction, point)

    #intersection of the line in the direction of line_direction and going through point_on_line with the surface
    etta = -(np.dot(point_on_line, line_direction) + surface_constant)/np.dot(line_direction, line_direction)
    intersection = etta * line_direction + point_on_line

    #distance between point and the logitudinal line
    dist = np.linalg.norm(intersection - point)

    return dist


def setting_base(coord):
    ent = gmsh.model.getEntities(2)

    endo = gmsh.model.getEntitiesForPhysicalGroup(2, 1)

    coord = gmsh.model.mesh.getNodesForPhysicalGroup(2, 1)[1]
    coord = np.reshape(coord, (-1, 3))
    center = np.mean(coord, axis=0)

    apex = gmsh.model.mesh.getNodesForPhysicalGroup(0, 4)[1]
    endo_border = gmsh.model.mesh.getNodesForPhysicalGroup(1, 5)[1]
    endo_border = np.reshape(endo_border, (-1, 3))
    c = np.mean(endo_border, axis=0)
    longitudinal_direction = (c - apex)/np.linalg.norm(c - apex) #longitudinal_direction_detection(coord)

    mean_distance_base = 0
    base_surfs = []
    for e in ent:
        surf = e[1]
        tag, coord, param = gmsh.model.mesh.getNodes(2, surf, True)

        coord = np.reshape(coord, (-1, 3))
        surf_mean = np.mean(coord, axis=0)

        normal = gmsh.model.getNormal(surf, param)[0:3]

        height = np.dot(surf_mean - center, longitudinal_direction)

        # if abs(np.dot(normal, longitudinal_direction)) > 0.5:
        #     dist = distance_point_to_line(surf_mean, center, longitudinal_direction)
        #     base_surfs.append(surf)
        #     mean_distance_base += dist
        if height > 20:
            dist = distance_point_to_line(surf_mean, center, longitudinal_direction)
            if dist > 8:
                if abs(np.dot(normal, longitudinal_direction)) > 0.5:
                    base_surfs.append(surf)
                    mean_distance_base += dist
    mean_distance_base /= len(base_surfs)

    base_surfs = [*base_surfs, *setting_epi([*endo, *base_surfs], center, longitudinal_direction, mean_distance_base)]
    group = gmsh.model.addPhysicalGroup(2, base_surfs, 2, name="base")


def setting_epi(determined_surfs, center, longitudinal_direction, mean_distance_base):
    remaining_base_surfs = []
    epi = []
    ent = gmsh.model.getEntities(2)
    for e in ent:
        surf = e[1]
        if surf not in determined_surfs:
            tag, coord, param = gmsh.model.mesh.getNodes(2, surf, True)

            coord = np.reshape(coord, (-1, 3))
            surf_mean = np.mean(coord, axis=0)

            height = np.dot(surf_mean - center, longitudinal_direction)
            if height > 20:
                dist = distance_point_to_line(surf_mean, center, longitudinal_direction)
                if dist > mean_distance_base:
                    epi.append(surf)
                else:
                    remaining_base_surfs.append(surf)
            else:
                epi.append(surf)

    group = gmsh.model.addPhysicalGroup(2, epi, 3, name="epi")
    return remaining_base_surfs


def setting_endo_border(endo_lines, endo_surfs):
    endo_border = []
    for s in endo_surfs:
        line = gmsh.model.getAdjacencies(2, s)[1]
        for l in line:
            surf = gmsh.model.getAdjacencies(1, l)[0]
            if surf[0] not in endo_surfs or surf[1] not in endo_surfs:
                endo_border.append(l)
                continue

    group = gmsh.model.addPhysicalGroup(1, endo_border, 5, name="endo_border")


def setting_endo(startNumber, endNumber, first_point_tag):
    endocardium_lines = []
    for n in range(startNumber, endNumber):
        endocardium_lines.append(gmsh.model.getAdjacencies(0, n + first_point_tag)[0])
    endocardium_lines = np.concatenate(endocardium_lines).ravel().tolist()
    myset = set(endocardium_lines)
    endocardium_lines = list(myset)

    endocardium_surfs = []
    for n in endocardium_lines:
        endocardium_surfs.append(gmsh.model.getAdjacencies(1, n)[0])
    endocardium_surfs = np.concatenate(endocardium_surfs).ravel().tolist()
    myset = set(endocardium_surfs)
    endocardium_surfs = list(myset)
    group = gmsh.model.addPhysicalGroup(2, endocardium_surfs, 1, name="endo")

    return endocardium_lines, endocardium_surfs,\
             # center, long_direction


def first_last_tags(dim):
    ent = gmsh.model.getEntities(dim)
    starting_point_tag = ent[0][1]
    ending_point_tag = ent[-1][1]
    return starting_point_tag, ending_point_tag


def main():
    pass


if __name__ == "__main__":
    main()

