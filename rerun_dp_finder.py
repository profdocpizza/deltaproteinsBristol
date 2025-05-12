
from dpFinder import find_deltaprots


find_deltaprots.find_deltaprots(pdb_dir="/home/tadas/code/deltaproteinsBristol/selected_deltaprots/no_disulfide",
                                output_dir="/home/tadas/code/deltaproteinsBristol/selected_deltaprots/no_disulfide",
                                out_filename="dp_finder_results",num_cores=5
)

find_deltaprots.find_deltaprots(pdb_dir="/home/tadas/code/deltaproteinsBristol/selected_deltaprots/variable_linkers",
                                output_dir="/home/tadas/code/deltaproteinsBristol/selected_deltaprots/variable_linkers",
                                out_filename="dp_finder_results",num_cores=5
)
