#!/usr/bin/env python3
"""
Standalone script to overlay (or singly render) Deltahedron geometries
with optional AlphaFold2-predicted proteins and thick ribs, using PyMOL.

New flags:
  - deltahedron_only          : draw only thin cylinders + vertex spheres
  - deltahedron_w_ribs_only  : draw thin cylinders + vertex spheres + thick ribs (no protein)
  - (default)                 : draw everything (protein + thin edges + spheres + ribs)
"""

import sys
import os
import time
from isambard.specifications import deltaprot
import numpy as np
import pandas as pd

import pymol
from pymol.cgo import CYLINDER, SPHERE, COLOR

# ---------------------------
# 1) LAUNCH PYMOL (GUI mode)
# ---------------------------

# Install license, remove "-c" so PyMOL creates a window
pymol.licensing.install_license_file(
    "/home/tadas/code/deltaproteinsBristol/pymol-edu-license.lic"
)
pymol.pymol_argv = ["pymol", "-q"] + sys.argv[1:]
pymol.finish_launching()
cmd = pymol.cmd

# ---------------------------
# 2) HELPER FUNCTIONS
# ---------------------------

import sys
from isambard.specifications import deltaprot_helper
from isambard.specifications.deltaprot import DeltaProt
from isambard.specifications.deltaprot_helper import Deltahedron
import numpy as np
import pandas as pd
import os
import time

import pymol
from pymol.cgo import CYLINDER, SPHERE, COLOR

pymol.licensing.install_license_file(
    "/home/tadas/code/deltaproteinsBristol/pymol-edu-license.lic"
)
pymol.pymol_argv = ["pymol", "-qc"] + sys.argv[1:]
cmd = pymol.cmd
import os
from PIL import Image
from fpdf import FPDF


def build_thin_edges_cgo(deltahedron, thin_radius=0.1, thin_color=(0.43, 0.41, 0.42)):
    """
    Return a CGO list of all thin‐radius cylinders for every edge in the deltahedron.
    """
    thin_cgo = []
    for i, neighbors in enumerate(deltahedron.connection_matrix):
        for j in neighbors:
            if i < j:
                v1 = deltahedron.vertices[i]
                v2 = deltahedron.vertices[j]
                thin_cgo += [
                    CYLINDER,
                    v1[0],
                    v1[1],
                    v1[2],
                    v2[0],
                    v2[1],
                    v2[2],
                    thin_radius,
                    thin_color[0],
                    thin_color[1],
                    thin_color[2],
                    thin_color[0],
                    thin_color[1],
                    thin_color[2],
                ]
    return thin_cgo


def build_vertex_spheres_cgo(
    deltahedron, thin_radius=0.1, thin_color=(0.43, 0.41, 0.42)
):
    """
    Return a CGO list of one small sphere at each deltahedron vertex.
    Each SPHERE is preceded by a COLOR primitive (required by PyMOL).
    """
    vert_cgo = []
    for v in deltahedron.vertices:
        vert_cgo += [
            COLOR,
            thin_color[0],
            thin_color[1],
            thin_color[2],  # set color for the sphere
            SPHERE,
            v[0],
            v[1],
            v[2],  # x, y, z
            thin_radius,  # radius
        ]
    return vert_cgo


def build_thick_ribs_cgo(
    deltahedron, ribs, rib_radius=0.5, rib_color=(0.80, 0.20, 0.10)
):
    """
    Return a CGO list of thick cylinders for the specified ribs.
    'ribs' is a list of (i, j) index pairs.
    """
    rib_cgo = []
    for i, j in ribs:
        v1 = deltahedron.vertices[i]
        v2 = deltahedron.vertices[j]
        rib_cgo += [
            CYLINDER,
            v1[0],
            v1[1],
            v1[2],
            v2[0],
            v2[1],
            v2[2],
            rib_radius,
            rib_color[0],
            rib_color[1],
            rib_color[2],
            rib_color[0],
            rib_color[1],
            rib_color[2],
        ]
    return rib_cgo


def apply_inverse_transform(object_names, transform_matrix):
    """
    Given a list of PyMOL object names and a 4×4 transform,
    compute the inverse (no scale) and apply it to each object via set_object_ttt().
    """
    # Extract uniform scale from the upper 3x3 block
    scale = np.linalg.norm(transform_matrix[:3, :3] @ np.array([1.0, 0.0, 0.0]))
    R = transform_matrix[:3, :3] / scale
    t = transform_matrix[:3, 3] / scale

    R_inv = R.T
    t_inv = -R_inv.dot(t)

    # Build 4×4 column-major matrix
    M_inv = np.eye(4)
    M_inv[:3, :3] = R_inv
    M_inv[:3, 3] = t_inv
    flat = M_inv.T.flatten().tolist()

    # Apply to each object
    for obj in object_names:
        cmd.set_object_ttt(obj, flat, homogenous=1)

    return R_inv, t_inv  # also return for centering step


def recenter_objects(object_names, deltahedron, R_inv, t_inv):
    """
    Compute centroid of transformed vertices and translate all object_names by -centroid.
    """
    verts = np.array(deltahedron.vertices)  # raw coordinates
    verts_trans = (R_inv @ verts.T).T + t_inv  # after inverse transform
    com = verts_trans.mean(axis=0).tolist()  # centroid coords

    # Translate each object by -com
    dx, dy, dz = com
    for obj in object_names:
        cmd.translate([-dx, -dy, -dz], object=obj)


def save_session_and_png(output_path):
    """
    Save a .pse alongside the given .png path, then render the PNG at high resolution.
    """
    pse_path = output_path.replace(".png", ".pse")
    cmd.save(pse_path)
    while not os.path.exists(pse_path):
        time.sleep(0.02)
    time.sleep(0.1)

    cmd.png(output_path, width=2000, height=2000, ray=1, quiet=1)
    while not os.path.exists(output_path):
        time.sleep(0.02)
    time.sleep(0.1)


def setup_rendering_camera(deltahedron):
    """
    Apply standard orthoscopic/white‐background settings and optionally
    rotate to a canonical orientation based on deltahedron.name.
    """
    cmd.set("orthoscopic", 1)
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("antialias", 2)
    cmd.set("ray_shadows", 0)
    cmd.set("ray_trace_mode", 1)

    if hasattr(deltahedron, "name"):
        if deltahedron.name == "octahedron":
            cmd.turn("x", -90)
            cmd.turn("y", 45)
            cmd.turn("x", 45)
        elif deltahedron.name == "snub_disphenoid":
            cmd.turn("x", -90)
            cmd.turn("y", 45)
        elif deltahedron.name == "gyro_square_bipyramid":
            cmd.turn("x", -90)
            cmd.turn("y", -105)
            cmd.turn("x", 15)
        elif deltahedron.name == "icosahedron":
            cmd.turn("x", -75)

    cmd.zoom(buffer=1, complete=1)


# ---------------------------
# 3) MAIN FUNCTION (REFRACTOR)
# ---------------------------


def pymol_visualise_deltaprot_deltahedron(
    # Always required arguments:
    deltahedron,  # Deltahedron instance (has .vertices, .connection_matrix, .name)
    output_path: str,
    transform_matrix: np.ndarray,  # 4×4 transform from raw Δ → aligned coords
    # Optional arguments for protein + ribs:
    af2_filepath: str = None,  # path to AF2 PDB (only used if not deltahedron_only / deltahedron_w_ribs_only=False)
    ribs: list = None,  # list of (i,j) for thick ribs (only used if ribs are to be drawn)
    # Flags to choose the scene:
    deltahedron_only: bool = False,
    deltahedron_w_ribs_only: bool = False,
):
    """
    Generate one of three scenes in PyMOL, then save a PSE + PNG:

    1) deltahedron_only=True:
         - Draw only thin cylinders + vertex spheres (no protein, no ribs).
         - 'af2_filepath' and 'ribs' are ignored.
         - transform_matrix still applies so that the deltahedron is centered properly.

    2) deltahedron_w_ribs_only=True:
         - Draw thin cylinders + vertex spheres + thick ribs (no protein).
         - 'ribs' must be provided; 'af2_filepath' is ignored.

    3) (default: both flags False):
         - Draw the AF2 protein (cartoon), plus thin cylinders, vertex spheres, and thick ribs.
         - 'af2_filepath' and 'ribs' both must be provided.

    Exactly one of (deltahedron_only, deltahedron_w_ribs_only) may be True; if both are False,
    we render the “full” scene including the protein.
    """

    # Validate flag combinations
    if deltahedron_only and deltahedron_w_ribs_only:
        raise ValueError(
            "Only one of deltahedron_only or deltahedron_w_ribs_only may be True."
        )

    # 1) Clear any previous objects
    cmd.delete("all")

    # 2) Optionally load the protein (full‐scene only)
    if not deltahedron_only and not deltahedron_w_ribs_only:
        if af2_filepath is None:
            raise ValueError("af2_filepath must be provided when rendering full scene.")
        cmd.load(af2_filepath, "deltaprotein", quiet=1)
        cmd.hide("everything", "deltaprotein")
        cmd.show("cartoon", "deltaprotein")
        cmd.color("slate", "deltaprotein")

    # 3) Build CGO objects
    thin_cgo = build_thin_edges_cgo(deltahedron)
    vert_cgo = build_vertex_spheres_cgo(deltahedron)
    # Load thin‐edges and vertex spheres even if deltahedron_only or deltahedron_w_ribs_only
    cmd.load_cgo(thin_cgo, "deltahedron_edges_cgo")
    cmd.load_cgo(vert_cgo, "deltahedron_vertices_cgo")

    # 4) If ribs are to be drawn (either full or ribs‐only), build & load them
    if not deltahedron_only:
        if ribs is None:
            raise ValueError("'ribs' must be provided to draw ribs.")
        rib_cgo = build_thick_ribs_cgo(deltahedron, ribs)
        cmd.load_cgo(rib_cgo, "deltahedron_ribs_cgo")

    # 5) Group thin edges + vertex spheres under a single name for toggling
    cmd.group("deltahedron_edges", "deltahedron_edges_cgo")
    cmd.group("deltahedron_edges", "deltahedron_vertices_cgo")

    # 6) Determine which objects exist and need transforms
    objects_to_transform = []
    if not deltahedron_only and not deltahedron_w_ribs_only:
        objects_to_transform.append("deltaprotein")

    # Always transform the CGO objects we loaded:
    objects_to_transform.append("deltahedron_edges_cgo")
    objects_to_transform.append("deltahedron_vertices_cgo")
    if not deltahedron_only:
        objects_to_transform.append("deltahedron_ribs_cgo")

    # 7) Apply inverse transform (no scale) to all relevant objects
    R_inv, t_inv = apply_inverse_transform(objects_to_transform, transform_matrix)

    # 8) Recenter entire scene so that the Δ‐centroid is at the origin
    recenter_objects(objects_to_transform, deltahedron, R_inv, t_inv)

    # 9) Setup rendering & camera
    setup_rendering_camera(deltahedron)

    # 10) Save session & PNG
    save_session_and_png(output_path)

    # 11) Clean up
    cmd.delete("all")
    cmd.reset()


# ---------------------------
# 4) EXAMPLE USAGE WRAPPERS
# ---------------------------


def generate_full_deltaprot_pngs():
    """
    Example loop over a DataFrame of designs, rendering the full scene (protein + deltahedron + ribs).
    """
    df = pd.read_pickle(
        "/home/tadas/code/deltaproteinsBristol/deltaprot_designs_all_data_with_results.pkl"
    )
    for _, row in df[:30].iterrows():
        print(f"Processing design: {row['dp_finder_orientation_code']}")
        orientation_code = row["dp_finder_orientation_code"]
        transform_matrix = np.array(row["dp_finder_total_transformation_matrix"])
        af2_pdb_path = os.path.join(
            "/home/tadas/code/deltaproteinsBristol/selected_deltaprots/no_disulfide",
            row["structure_prediction_file_name"],
        )

        # Build and transform Deltahedron
        rib_num = int(orientation_code[1])
        deltahedron = Deltahedron.choose_deltahedron_by_rib_number(
            rib_num=rib_num, edge_length=1
        )
        deltahedron.transform(transform_matrix)

        # Determine ribs
        ribs = deltaprot_helper.orientation_codes_sorted_ribs[
            orientation_code.replace("_", ".")
        ]

        out_dir = "/home/tadas/code/deltaproteinsBristol/deltaprot_deltahedron_fig_data"
        png_out = os.path.join(out_dir, f"{orientation_code}.png")
        print(f"Rendering full scene for {orientation_code} → {png_out}")

        pymol_visualise_deltaprot_deltahedron(
            deltahedron=deltahedron,
            af2_filepath=af2_pdb_path,
            ribs=ribs,
            output_path=png_out,
            transform_matrix=transform_matrix,
            deltahedron_only=False,
            deltahedron_w_ribs_only=False,
        )


def generate_deltahedron_only_png(deltahedron, output_path, transform_matrix):
    """
    Example of rendering only the thin‐edge + vertex spheres, no ribs, no protein.
    """
    pymol_visualise_deltaprot_deltahedron(
        deltahedron=deltahedron,
        output_path=output_path,
        transform_matrix=transform_matrix,
        ribs=None,  # ignored in this mode
        af2_filepath=None,  # ignored in this mode
        deltahedron_only=True,
        deltahedron_w_ribs_only=False,
    )


def generate_deltahedron_with_ribs_only_png(
    deltahedron, ribs, output_path, transform_matrix
):
    """
    Example of rendering the thin‐edges + vertex spheres + thick ribs, but no protein.
    """
    pymol_visualise_deltaprot_deltahedron(
        deltahedron=deltahedron,
        output_path=output_path,
        transform_matrix=transform_matrix,
        af2_filepath=None,  # ignored in this mode
        ribs=ribs,
        deltahedron_only=False,
        deltahedron_w_ribs_only=True,
    )


def generate_deltahedron_only_pngs():
    for deltahedron_name in Deltahedron.supported_deltahedrons:
        deltahedron = Deltahedron.choose_deltahedron_by_name(deltahedron_name, 11)
        transform_matrix = np.eye(4)
        output_path = f"/home/tadas/code/deltaproteinsBristol/deltaprot_deltahedron_fig_data/{deltahedron_name}_only.png"
        print(f"Generating deltahedron only PNG for {deltahedron_name} → {output_path}")
        generate_deltahedron_only_png(
            deltahedron=deltahedron,
            output_path=output_path,
            transform_matrix=transform_matrix,
        )


def generate_deltahedron_with_ribs_only_pngs():
    for orientation_code in DeltaProt.orientation_codes:
        deltahedron = Deltahedron.choose_deltahedron_by_rib_number(
            int(orientation_code[1]), edge_length=11
        )
        transform_matrix = np.eye(4)
        output_path = f"/home/tadas/code/deltaproteinsBristol/deltaprot_deltahedron_fig_data/{orientation_code}_ribs_only.png"
        print(
            f"Generating deltahedron with ribs only PNG for {orientation_code} → {output_path}"
        )
        ribs = deltaprot_helper.orientation_codes_sorted_ribs[
            orientation_code.replace("_", ".")
        ]
        generate_deltahedron_with_ribs_only_png(
            deltahedron=deltahedron,
            ribs=ribs,
            output_path=output_path,
            transform_matrix=transform_matrix,
        )


if __name__ == "__main__":
    generate_deltahedron_only_pngs()
    generate_deltahedron_with_ribs_only_pngs()
    generate_full_deltaprot_pngs()
