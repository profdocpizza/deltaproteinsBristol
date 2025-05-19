#!/usr/bin/env python3
"""
Standalone script to overlay AlphaFold2-predicted protein structures with a Deltahedron geometry using PyMOL.
"""

import sys
from isambard.specifications import deltaprot_helper
from isambard.specifications.deltaprot import DeltaProt
from isambard.specifications.deltaprot_helper import Deltahedron
import numpy as np
import pandas as pd
import os
import time

import pymol
from pymol.cgo import CYLINDER
import pymol

pymol.licensing.install_license_file(
    "/home/tadas/code/deltaproteinsBristol/pymol-edu-license.lic"
)
pymol.pymol_argv = ["pymol", "-qc"] + sys.argv[1:]
cmd = pymol.cmd
import os
from PIL import Image
from fpdf import FPDF


def check_license_in_directory(directory):
    """Search for a .lic file starting one directory up from the helpers.py directory and scanning all subdirectories."""

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".lic"):
                license_file_path = os.path.join(root, file)
                print(f"Found pymol license at {license_file_path}")
                pymol.licensing.check_license_file(license_file_path)
                # pymol.licensing.install_license_file(license_file_path)
                pymol.licensing.install_license_file(license_file_path)
                # return pymol.licensing.install_license_file(license_file_path)
    return "No .lic file found in the parent directory or its subdirectories."


def pymol_visualise_deltaprot_deltahedron(
    af2_filepath: str,
    deltahedron: Deltahedron,
    ribs,
    output_path: str,
    transform_matrix: np.ndarray,
):
    """
    Load an AF2 PDB and overlay Deltahedron edges and ribs as CGO cylinders,
    then apply the inverse rotation and translation (preserving scale) of transform_matrix,
    and save a PNG and PSE session.
    """
    # Clear scene

    cmd.delete("all")

    # Load and style protein
    cmd.load(af2_filepath, "deltaprotein", quiet=1)
    cmd.hide("everything", "deltaprotein")
    cmd.show("cartoon", "deltaprotein")
    cmd.color("slate", "deltaprotein")

    # CGO styling parameters
    thin_radius = 0.1
    rib_radius = 0.5
    thin_color = (0.43, 0.41, 0.42)
    rib_color = (0.8, 0.2, 0.1)

    # Build CGO for all edges
    cgo = []
    for i, neighbors in enumerate(deltahedron.connection_matrix):
        for j in neighbors:
            if i < j:
                v1 = deltahedron.vertices[i]
                v2 = deltahedron.vertices[j]
                cgo += [
                    CYLINDER,
                    *v1,
                    *v2,
                    thin_radius,
                    *thin_color,
                    *thin_color,
                ]
    # Highlight ribs
    for i, j in ribs:
        v1 = deltahedron.vertices[i]
        v2 = deltahedron.vertices[j]
        cgo += [
            CYLINDER,
            *v1,
            *v2,
            rib_radius,
            *rib_color,
            *rib_color,
        ]
    cmd.load_cgo(cgo, "deltahedron")

    # Extract uniform scale from T
    scale = np.linalg.norm(transform_matrix[:3, :3] @ np.array([1.0, 0, 0]))
    # Extract pure rotation (R) and original translation (t), undoing scale
    R = transform_matrix[:3, :3] / scale
    t = transform_matrix[:3, 3] / scale

    # Compute inverse rotation and translation (no scale)
    R_inv = R.T
    t_inv = -R_inv.dot(t)

    # Build inverse transform 4x4 (column-major order for PyMOL)
    M_inv = np.eye(4)
    M_inv[:3, :3] = R_inv
    M_inv[:3, 3] = t_inv
    # PyMOL expects a flattened column-major list
    flat = M_inv.T.flatten().tolist()

    # Apply inverse to objects
    cmd.set_object_ttt("deltaprotein", flat, homogenous=1)
    cmd.set_object_ttt("deltahedron", flat, homogenous=1)

    verts = np.array(deltahedron.vertices)
    verts_trans = (R_inv @ verts.T).T + t_inv  # shape (N,3)

    # 3) compute center‐of‐mass
    com = verts_trans.mean(axis=0)  # length‐3 vector

    # 4) shift both protein and deltahedron by −COM in PyMOL
    cmd.translate(list(-com), object="deltaprotein")
    cmd.translate(list(-com), object="deltahedron")
    # cmd.translate(list([0,0,50]), object = "deltaprotein")
    # cmd.translate(list([0,0,50]), object = "deltahedron")
    # set camera position at
    # prepare_for_picture()

    # Rendering settings
    cmd.set("ray_opaque_background", 0)
    cmd.bg_color("white")
    cmd.set("ray_trace_frames", 0)
    cmd.set("ray_shadows", 1)
    cmd.set("antialias", 2)
    cmd.set("orthoscopic", 1)
    cmd.set("ray_trace_mode", 1)

    # recreate M&F figure rotations
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

    # cmd.center()
    cmd.zoom(buffer=1, complete=1)
    # Save session
    pse_path = output_path.replace(".png", ".pse")
    cmd.save(pse_path)
    while not os.path.exists(pse_path):
        time.sleep(0.02)
    time.sleep(0.1)
    # Render PNG
    cmd.png(output_path, width=2000, height=2000, ray=1, quiet=1)
    while not os.path.exists(output_path):
        time.sleep(0.02)
    time.sleep(0.1)
    # Clean up
    cmd.delete("all")
    cmd.reset()


def prepare_for_picture():
    from pymol import cmd

    # Enable orthographic projection
    cmd.set("orthoscopic", 1)

    # Adjust clipping planes to encompass the entire object
    cmd.clip("slab", 0)  # Reset any slab clipping
    # cmd.set("clip_front", 0)  # Set front clipping plane to 0
    # cmd.set("clip_back", 0)   # Set back clipping plane to 0

    # Center and zoom to fit all objects
    cmd.zoom("deltahedron")

    # Optionally, set the field of view to a suitable value
    cmd.set("field_of_view", 10)


def generate_deltaprotein_deltahedron_pngs():

    # Load design metadata
    df = pd.read_pickle(
        "/home/tadas/code/deltaproteinsBristol/deltaprot_designs_all_data_with_results.pkl"
    )

    # Process first 30 designs
    for _, row in df[:30].iterrows():
        orientation_code = row["dp_finder_orientation_code"]
        # if orientation_code not in ["b4inni","b4innn"]:
        #     continue
        transform_matrix = np.array(row["dp_finder_total_transformation_matrix"])
        af2_pdb_path = os.path.join(
            "/home/tadas/code/deltaproteinsBristol/selected_deltaprots/no_disulfide",
            row["structure_prediction_file_name"],
        )

        # Build and transform the Deltahedron
        rib_num = int(orientation_code[1])
        deltahedron = Deltahedron.choose_deltahedron_by_rib_number(
            rib_num=rib_num, edge_length=1
        )
        deltahedron.transform(transform_matrix)

        # Get ribs for this orientation
        ribs = deltaprot_helper.orientation_codes_sorted_ribs[
            orientation_code.replace("_", ".")
        ]

        # Output filename
        out_dir = "/home/tadas/code/deltaproteinsBristol/deltaprot_deltahedron_fig_data"
        png_out = os.path.join(out_dir, f"{orientation_code}.png")
        print(f"Rendering {orientation_code} -> {png_out}")

        # Generate overlay image
        pymol_visualise_deltaprot_deltahedron(
            af2_filepath=af2_pdb_path,
            deltahedron=deltahedron,
            transform_matrix=transform_matrix,
            ribs=ribs,
            output_path=png_out,
        )


ROW_CODES = [
    ["b3iii", "b3nnn"],
    [
        "b4iiiix",
        "b4iiiiy",
        "b4nnnnx",
        "b4nnnny",
        "b4iiin",
        "b4innn",
        "b4inin",
        "l4iin",
        "l4inn",
        "h4i_n",
    ],
    [
        "b5iiiin",
        "b5innnn",
        "b5iinin",
        "b5ininn",
        "l5iiin",
        "l5innn",
        "l5inni",
        "l5niin",
        "h5i_i",
        "h5n_n",
    ],
    [
        "b6ininin",
        "b6iiniin",
        "b6inninn",
        "l6innni",
        "l6niiin",
        "h6i_i_i",
        "h6n_n_n",
        "s6",
    ],
]


def assemble_deltaprots_pdf(input_dir: str, save_root: str):
    # 1) Gather all PNGs
    pngs = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".png")
    ]

    # 2) Match exactly one image per code
    rows_images = []
    for row_idx, codes in enumerate(ROW_CODES, start=1):
        matched = []
        for code in codes:
            hits = [p for p in pngs if code in os.path.basename(p)]
            if len(hits) == 0:
                raise FileNotFoundError(
                    f"[Row {row_idx}] no image found for code '{code}'"
                )
            if len(hits) > 1:
                raise RuntimeError(
                    f"[Row {row_idx}] multiple images for code '{code}': {hits}"
                )
            matched.append(hits[0])
        rows_images.append(matched)

    # 3) Prepare output path
    out_dir = os.path.join(save_root, "M&F_deltaprots")
    os.makedirs(out_dir, exist_ok=True)
    output_pdf = os.path.join(out_dir, "M&F_deltaprots.pdf")

    # 4) Set up PDF
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(False)
    pdf.add_page()

    # margins
    margin_x = 5  # mm left/right
    margin_y = 5  # mm top/bottom

    total_w = pdf.w - 2 * margin_x
    total_h = pdf.h - 2 * margin_y
    nrows = len(rows_images)
    row_h = total_h / nrows * 0.7

    # 5) Compute baseline_span using rows 2–4
    lens_2to4 = [len(r) for r in rows_images[1:]]
    max_n = max(lens_2to4)
    min_n = min(lens_2to4)
    baseline_span = (min_n / max_n) * total_w

    # 6) Precompute the 4th-row cell width
    n4 = len(rows_images[3])
    cell_w_row4 = baseline_span / n4

    # 7) Place text + images
    pdf.set_font("Courier", style="B", size=8)
    text_height = 4  # mm reserved for the label

    for i, imgs in enumerate(rows_images):
        # choose cell width
        if i == 0:
            cell_w = cell_w_row4
        else:
            cell_w = baseline_span / len(imgs)

        # y-coordinate of the TOP of the image
        y_img = margin_y + i * row_h
        for j, img_path in enumerate(imgs):
            x_img = margin_x + j * cell_w

            # 7a) draw label above the image using ROW_CODES
            code = (
                ROW_CODES[i][j]
                .upper()
                .replace("_", ".")
                .replace("X", "x")
                .replace("Y", "y")
            )
            pdf.set_xy(x_img, y_img - text_height)
            pdf.cell(cell_w, text_height, code, border=0, ln=0, align="C")

            # 7b) open for aspect ratio
            with Image.open(img_path) as im:
                iw, ih = im.size
            ar = iw / ih

            # fit image into the same cell height (minus label area)
            avail_h = row_h - text_height
            w = cell_w
            h = cell_w / ar
            if h > avail_h:
                h = avail_h
                w = avail_h * ar

            # place the image
            pdf.image(img_path, x=x_img, y=y_img, w=w, h=h)

    # 8) Save PDF
    pdf.output(output_pdf)


if __name__ == "__main__":
    # generate_deltaprotein_deltahedron_pngs()
    assemble_deltaprots_pdf(
        "/home/tadas/code/deltaproteinsBristol/deltaprot_deltahedron_fig_data",
        "/home/tadas/code/deltaproteinsBristol/deltaprot_deltahedron_fig_data",
    )
