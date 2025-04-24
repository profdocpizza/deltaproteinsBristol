from itertools import combinations
import ampal
import numpy as np


gene_synthesis_cols = ["Well Position","Name","Sequence"]
sharing_cols = gene_synthesis_cols + [
    "orientation_code",
    "isoelectric_point",
    "charge",
    "mass",
    "sequence_length",
    "mean_plddt",
    "mean_pae",
    "ptm",
    "tm_rmsd100",
    "dp_finder_total_cost",
    "sequence_molar_extinction_280",
    "model_sequence"
]
metrics_cols = [
    "orientation_code",
    "atp_cost_per_aa",
    "atp_cost",
    "dna_complexity_per_aa",
    "pll",
    "pll_per_aa",
    "isoelectric_point",
    "charge",
    # "sequence_charge" redundant to charge
    "mass",
    "sequence_length",
    "mean_plddt",
    "mean_pae",
    "ptm",
    "tm_rmsd",
    "tm_score_assembly",
    "tm_score_design",
    "tm_rmsd100",
    # "n_potential_disulfide_bonds",
    "dp_finder_total_cost",
    "dp_finder_scale",

    "predicted_usability",
    "combined_score",
    "path_score_version",
    "residues_per_helix",
    "deltahedron_edge_length",
    "diffuse_termini",
    "avoid_amino_acids",
    "increase_amino_acid_likelihood",
    "energy_minimization",
    "algorithms_sequence_prediction",
    "algorithms_structure_prediction",
    "rf_file_num",
    "rib_num",
    "aa_count_per_gap",
    "model_sequence",
    "sequence_molar_extinction_280",
    # "sequence_molecular_weight", # redundant to mass
    "backbone_loop_mask_string",
    "aligned_length",
    "seq_id",
    "sequence_name",
    "dssp_assignment",
    "aggrescan3d_avg_value",
    "hydrophobic_fitness",
    "packing_density",
    "rosetta_total_per_aa",
    "name",
    "overall_distance_score_v2",
    "overall_path_distance_score_v2",
    "overall_contact_order_score",
    "overall_linker_convenience_score",
    "overall_linker_convenience_v2_score",
    "path_score_v2",
    "path_score_v3",
]

composition_cols = [f"composition_{aa}" for aa in [
    "PHE", "TYR", "TRP", "GLY", "ALA", "VAL", "LEU", "ILE", "MET", "CYS",
    "PRO", "THR", "SER", "ASN", "GLN", "ASP", "GLU", "LYS", "ARG", "HIS"
]]

useful_cols = gene_synthesis_cols + metrics_cols + composition_cols

useful_cols_non_csv = useful_cols + [
    "taylor_letter_packing_descriptors",
    "chothia_omega_angles",
] 

def count_non_overlapping_pairs(coords, ids, threshold):
    # threshold of 6 would be generous
    """
    Counts non-overlapping pairs of points from the coords array that are within
    the given threshold distance. If a coordinate is involved in multiple pairs,
    the pair with the smallest distance is chosen.
    
    Parameters:
        coords (np.ndarray): An array of shape (N, 3) containing the coordinates.
        ids (np.ndarray): An array of shape (N,) containing the corresponding residue IDs.
        threshold (float): The maximum distance for a pair to be considered.
        
    Returns:
        int: The number of non-overlapping pairs.
        list: A list of tuples, each tuple contains 
              (index1, index2, distance, id1, id2) for the pair.
    """
    potential_pairs = []
    
    # Iterate over all unique pairs (i, j)
    for i, j in combinations(range(len(coords)), 2):
        distance = np.linalg.norm(coords[i] - coords[j])
        if distance <= threshold:
            potential_pairs.append((i, j, distance, ids[i], ids[j]))
    
    # Sort pairs by distance (smallest distance first)
    potential_pairs.sort(key=lambda pair: pair[2])
    
    selected_pairs = []
    used_indices = set()
    
    # Greedily select pairs ensuring no overlapping indices
    for i, j, distance, id1, id2 in potential_pairs:
        if i not in used_indices and j not in used_indices:
            selected_pairs.append({
                "CB_distance": round(distance,2),
                "id1": id1,
                "id2": id2,
                "sequence_distance": np.abs(int(id1) - int(id2))
            })
            used_indices.add(i)
            used_indices.add(j)
    
    return selected_pairs


def check_potential_disulfide_bonds(pdb_file_path):
    assembly=ampal.load_pdb(pdb_file_path)
    if isinstance(assembly,ampal.AmpalContainer):
        assembly=assembly[0]
    
    cysteine_residues = [res for res in assembly[0] if res.mol_code == "CYS"]
    CB_cys_coordinates = np.array([res["CB"].array for res in cysteine_residues])
    CB_cys_ids = np.array([res["CB"].id for res in cysteine_residues])
    disulfides_info = count_non_overlapping_pairs(CB_cys_coordinates, CB_cys_ids, threshold=6)
    n_disulfide = len(disulfides_info)
    return n_disulfide

