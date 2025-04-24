import gmsh
import os
import argparse

parser = argparse.ArgumentParser(description="Ratio of radial-dependent mesh resolution.")
parser.add_argument('-r', '--ratio', type=float, default=1.0)
parser.add_argument('-p', '--power', type=float, default=1.0)

args = parser.parse_args()


gmsh.initialize()
gmsh.model.add("DFG 3D")

c = [0, 0, 0]

r1 = 0.05
r2 = 0.06
sphere = gmsh.model.occ.addSphere(c[0], c[1], c[2], r1)
hollow = gmsh.model.occ.addSphere(c[0], c[1], c[2], r2)

shell = gmsh.model.occ.cut([(3, hollow)], [(3, sphere)], removeObject=True, removeTool=True)

gmsh.model.occ.synchronize()

volumes = gmsh.model.getEntities(dim=3)

gmsh.model.addPhysicalGroup(3, [volumes[0][1]], 11)
gmsh.model.setPhysicalName(3, 11, "shell")

surfaces = gmsh.model.occ.getEntities(dim=2)

gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], 111, name="outer_boundary")
gmsh.model.addPhysicalGroup(2, [surfaces[1][1]], 112, name="inner_boundary")

gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.ElementOrder", 1)

f = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(f, "F",\
 f"1/{args.ratio}")  # (sqrt(x*x + y*y + z*z)/1.5 - ({r2}+{r1})/2) * (sqrt(x*x + y*y + z*z)/1.5 - ({r2}+{r1})/2)  
gmsh.model.mesh.field.setAsBackgroundMesh(f)

gmsh.model.mesh.generate(3)
print(gmsh.model.mesh.getElements(dim=3)[1][0].shape[0])

mesh_folder = os.path.join("Output", "sphere", 'adaptive_check', f'adaptivity_pow_{args.power}') # {args.ratio}
if not os.path.exists(mesh_folder):
	os.makedirs(mesh_folder)
mesh_file = os.path.join(mesh_folder, f"mesh_res_ratio_{args.ratio}.msh")
gmsh.write(mesh_file)

gmsh.finalize()


