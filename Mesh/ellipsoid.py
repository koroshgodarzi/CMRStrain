import gmsh
import math
import os
import argparse

parser = argparse.ArgumentParser(description="Ratio of radial-dependent mesh resolution.")
parser.add_argument('-r', '--ratio', type=int, default=16)
parser.add_argument('-x', '--x_radius', type=float, default=0.014)
parser.add_argument('-z', '--z_radius', type=float, default=0.08)
parser.add_argument('-t', '--thickness', type=float, default=0.008)
parser.add_argument('-o', '--output_file', type=str, default='test')

args = parser.parse_args()

# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("Half_Ellipsoid_Shell_with_Cut_Surface")

# Parameters of the ellipsoid
a = args.x_radius  # semi-axis length along x
b = args.x_radius  # semi-axis length along y
c = args.z_radius  # semi-axis length along z (vertical)

# Thickness of the shell
thickness = args.thickness

# Discretization (mesh fineness)
n_points = 30  # Discretization control

# Mesh size for controlling mesh density
mesh_size_endo = 0.4
mesh_size_epi = 0.4

# Create the single top point of the ellipsoid (v = 0, phi = 0)
top_point_outer = gmsh.model.geo.addPoint(0, 0, c) #, meshSize=mesh_size_epi
top_point_inner = gmsh.model.geo.addPoint(0, 0, c - thickness) #, meshSize=mesh_size_endo

# Create points for the outer half-ellipsoid (top half)
points_outer = []
for u in range(n_points):  # Full range for theta (0 to 2*pi)
    for v in range(1, n_points // 2 + 1):  # Half range for phi (0 to pi/2)
        theta = 2 * math.pi * u / n_points
        phi = (math.pi / 2) * v / (n_points // 2)  # From 0 to pi/2 for top half
        x = a * math.sin(phi) * math.cos(theta)
        y = b * math.sin(phi) * math.sin(theta)
        z = c * math.cos(phi)
        points_outer.append(gmsh.model.geo.addPoint(x, y, z)) # , meshSize=mesh_size_epi

# Create points for the inner half-ellipsoid (shell thickness)
points_inner = []
for u in range(n_points):  # Full range for theta (0 to 2*pi)
    for v in range(1, n_points // 2 + 1):  # Half range for phi (0 to pi/2)
        theta = 2 * math.pi * u / n_points
        phi = (math.pi / 2) * v / (n_points // 2)  # From 0 to pi/2 for top half
        x = (a - thickness) * math.sin(phi) * math.cos(theta)
        y = (b - thickness) * math.sin(phi) * math.sin(theta)
        z = (c - thickness) * math.cos(phi)
        points_inner.append(gmsh.model.geo.addPoint(x, y, z)) # , meshSize=mesh_size_endo


# Create lines and surfaces for the outer and inner half-ellipsoid shell
outer_surface_loops = []
inner_surface_loops = []
cut_surface_loops = []


for i in range(n_points):
    for j in range(n_points // 2 + 1):  # Now we include the case where v = 0 only once
        # Define the cut surface (side surface) between inner and outer points
        if j == n_points // 2 :  # Only connect points on the cut surface (when phi = 0)  # - 1
            p1_outer = i * (n_points // 2 ) + (j - 1)
            p2_outer = i * (n_points // 2 ) + j
            p3_outer = ((i + 1) % n_points) * (n_points // 2 ) + (j - 1)
            p4_outer = ((i + 1) % n_points) * (n_points // 2 ) + j

            p1_inner = i * (n_points // 2 ) + (j - 1)
            p2_inner = i * (n_points // 2 ) + j
            p3_inner = ((i + 1) % n_points) * (n_points // 2 ) + (j - 1)
            p4_inner = ((i + 1) % n_points) * (n_points // 2 ) + j
            l_cut_outer = gmsh.model.geo.addLine(points_outer[p1_outer], points_outer[p3_outer])
            l_cut_inner = gmsh.model.geo.addLine(points_inner[p1_inner], points_inner[p3_inner])

            # Connect outer and inner lines along the cut edge
            l_side_1 = gmsh.model.geo.addLine(points_outer[p1_outer], points_inner[p1_inner])
            l_side_2 = gmsh.model.geo.addLine(points_outer[p3_outer], points_inner[p3_inner])

            # Create surface on the cut (vertical surface)
            cut_loop = gmsh.model.geo.addCurveLoop([l_cut_outer, l_side_2, -l_cut_inner, -l_side_1])
            cut_surface = gmsh.model.geo.addPlaneSurface([cut_loop])
            cut_surface_loops.append(cut_surface)
            continue

        if j == 0:  # Handle top point connection (phi = 0)
            p1_outer = top_point_outer
            p2_outer = i * (n_points // 2) + j
            p3_outer = top_point_outer
            p4_outer = ((i + 1) % n_points) * (n_points // 2) + j

            p1_inner = top_point_inner
            p2_inner = i * (n_points // 2) + j
            p3_inner = top_point_inner
            p4_inner = ((i + 1) % n_points) * (n_points // 2) + j

            # Create lines connecting to the top point for both outer and inner ellipsoids
            l1_outer = gmsh.model.geo.addLine(p1_outer, points_outer[p2_outer])
            l3_outer = gmsh.model.geo.addLine(points_outer[p2_outer], points_outer[p4_outer])
            l4_outer = gmsh.model.geo.addLine(points_outer[p4_outer], p3_outer)

            l1_inner = gmsh.model.geo.addLine(p1_inner, points_inner[p2_inner])
            l3_inner = gmsh.model.geo.addLine(points_inner[p2_inner], points_inner[p4_inner])
            l4_inner = gmsh.model.geo.addLine(points_inner[p4_inner], p3_inner)
            
            # Create outer surface connected to the top point
            outer_loop = gmsh.model.geo.addCurveLoop([l1_outer, l3_outer, l4_outer])
            outer_surface = gmsh.model.geo.addPlaneSurface([outer_loop])
            outer_surface_loops.append(outer_surface)

            # Create inner surface connected to the top point
            inner_loop = gmsh.model.geo.addCurveLoop([l1_inner, l3_inner, l4_inner])
            inner_surface = gmsh.model.geo.addPlaneSurface([inner_loop])
            inner_surface_loops.append(inner_surface)
            
        elif j != n_points // 2:
            p1_outer = i * (n_points // 2 ) + (j - 1)
            p2_outer = i * (n_points // 2 ) + j
            p3_outer = ((i + 1) % n_points) * (n_points // 2 ) + (j - 1)
            p4_outer = ((i + 1) % n_points) * (n_points // 2 ) + j

            p1_inner = i * (n_points // 2 ) + (j - 1)
            p2_inner = i * (n_points // 2 ) + j
            p3_inner = ((i + 1) % n_points) * (n_points // 2 ) + (j - 1)
            p4_inner = ((i + 1) % n_points) * (n_points // 2 ) + j

            # Create lines for the outer surface
            l1_outer = gmsh.model.geo.addLine(points_outer[p1_outer], points_outer[p2_outer])
            l2_outer = gmsh.model.geo.addLine(points_outer[p2_outer], points_outer[p4_outer])
            l3_outer = gmsh.model.geo.addLine(points_outer[p4_outer], points_outer[p3_outer])
            l4_outer = gmsh.model.geo.addLine(points_outer[p3_outer], points_outer[p1_outer])

            # Create outer surface
            outer_loop = gmsh.model.geo.addCurveLoop([l1_outer, l2_outer, l3_outer, l4_outer])
            outer_surface = gmsh.model.geo.addPlaneSurface([outer_loop])
            outer_surface_loops.append(outer_surface)

            # Create lines for the inner surface (shell thickness)
            l1_inner = gmsh.model.geo.addLine(points_inner[p1_inner], points_inner[p2_inner])
            l2_inner = gmsh.model.geo.addLine(points_inner[p2_inner], points_inner[p4_inner])
            l3_inner = gmsh.model.geo.addLine(points_inner[p4_inner], points_inner[p3_inner])
            l4_inner = gmsh.model.geo.addLine(points_inner[p3_inner], points_inner[p1_inner])

            # Create inner surface
            inner_loop = gmsh.model.geo.addCurveLoop([l1_inner, l2_inner, l3_inner, l4_inner])
            inner_surface = gmsh.model.geo.addPlaneSurface([inner_loop])
            inner_surface_loops.append(inner_surface)

endoNumNodes = len(points_inner)

gmsh.model.geo.removeAllDuplicates()

# Define the volume from the surfaces (outer, inner, and cut surface)
volume_surfaces = outer_surface_loops + inner_surface_loops + cut_surface_loops
volume = gmsh.model.geo.addSurfaceLoop(volume_surfaces)

v = gmsh.model.geo.addVolume([volume])

# Synchronize the geometry
gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(2, outer_surface_loops, 3, name='outer_boundary')
gmsh.model.addPhysicalGroup(2, inner_surface_loops, 1, name='inner_boundary')
gmsh.model.addPhysicalGroup(2, cut_surface_loops, 2, name='base')
gmsh.model.addPhysicalGroup(3, [v], 4, name='myo')
gmsh.model.addPhysicalGroup(0, [top_point_inner], 5, name='apex_inside')
gmsh.model.addPhysicalGroup(0, [top_point_outer], 6, name='apex_outside')

gmsh.model.geo.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

f = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(f, "F", f"((x*x/({a}*{a}) + y*y/({b}*{b}) + z*z/({c}*{c}))/{args.ratio})")  # z*z/1024 +  + x*x/({a}*{a}) + y*y/({b}*{b}) + z*z/({c}*{c})
# gmsh.model.mesh.field.setString(f, "F", f"z < 0.4 ? 1 : 2")  # z*z/1024 +  + x*x/({a}*{a}) + y*y/({b}*{b}) + z*z/({c}*{c})
# (sqrt(x*x + y*y + z*z)/1.5 - ({r2}+{r1})/2) * (sqrt(x*x + y*y + z*z)/1.5 - ({r2}+{r1})/2)  
gmsh.model.mesh.field.setAsBackgroundMesh(f)

# Mesh generation (3D)
gmsh.model.mesh.generate(3)

if not os.path.exists(os.path.join('Output', 'ellipsoid', 'geom_'+str(args.output_file))):
    os.makedirs(os.path.join('Output', 'ellipsoid', 'geom_'+str(args.output_file)))

with open(os.path.join('Output', 'ellipsoid', 'geom_'+str(args.output_file), 'info.txt'), 'w') as f:
    f.write(f'semi-axis length along x: {a} \n')
    f.write(f'semi-axis length along y: {b} \n')
    f.write(f'semi-axis length along z (vertical): {c} \n')
    f.write(f'Wall thickness: {thickness} \n')
    f.write(f'Mesh resolution ratio: {args.ratio} \n')


file = os.path.join('Output', 'ellipsoid', 'test', 'geom_'+str(args.output_file), f'ellip_{args.ratio}.msh')

# Optionally save the model
gmsh.write(file)

gmsh.model.geo.synchronize()


# Visualize in Gmsh GUI
# gmsh.fltk.run()

# Finalize Gmsh
gmsh.finalize()


