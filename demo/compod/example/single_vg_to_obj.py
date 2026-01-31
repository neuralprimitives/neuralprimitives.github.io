import os
from pycompod import VertexGroup, PolyhedralComplex


def process_file(input_file, output_file):
    regularization = 0.5  # higher value -> simpler model
    
    vg = VertexGroup(input_file, verbosity=20, debug_export=False)
    cc = PolyhedralComplex(vg, device='cpu', verbosity=20)

    cc.construct_partition()
    cc.add_bounding_box_planes()
    cc.label_partition(mode="normals", regularization={"area": regularization})
    cc.simplify_partition_tree_based()
    cc.save_surface(out_file=output_file, triangulate=False)
    
    
if __name__ == '__main__':
    import argparse

    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Convert .vg files to .obj format")
    parser.add_argument("--input_file", type=str, required=True, help="Directory to save .obj files")
    parser.add_argument("--output_file", type=str, required=True, help="Directory with .vg files")
    
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    process_file(input_file, output_file)
