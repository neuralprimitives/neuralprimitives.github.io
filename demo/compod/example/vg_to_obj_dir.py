import os

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


def process_file(file, input_directory, output_directory):
    from pycompod import VertexGroup, PolyhedralComplex
    input_file = os.path.join(input_directory, file)
    output_file = os.path.join(output_directory, file.replace(".vg", ".obj"))
    regularization = 0.5  # higher value -> simpler model
    

    vg = VertexGroup(input_file, verbosity=20, debug_export=False)
    cc = PolyhedralComplex(vg, device='cpu', verbosity=20)

    cc.construct_partition()
    cc.add_bounding_box_planes()
    cc.label_partition(mode="normals", regularization={"area": regularization})
    cc.simplify_partition_tree_based()
    cc.save_surface(out_file=output_file, triangulate=False)


def process_directory(directory, target_directory):
    files = os.listdir(directory)
    files = [file for file in files if file.endswith(".vg")]  # Ensure only .vg files are processed
    
    # Uncomment the below if you want to skip files that have already been processed
    processed_files = [file.replace(".obj", ".vg") for file in os.listdir(target_directory) if file.endswith(".obj")]
    files = [file for file in files if file not in processed_files]

    print(f"Processing {len(files)} files")
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_file, file, directory, target_directory): file for file in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            file = futures[future]
            try:
                future.result()  # This will raise an exception if the subprocess failed
            except Exception as e:
                print(f"Error processing file {file}: {e}")


if __name__ == '__main__':
    import argparse

    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Convert .vg files to .obj format")
    parser.add_argument("--obj_dir", type=str, required=True, help="Directory to save .obj files")
    parser.add_argument("--gocopp_dir", type=str, required=True, help="Directory with .vg files")

    args = parser.parse_args()

    gocopp_dir = args.gocopp_dir
    obj_dir = args.obj_dir

    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)

    process_directory(gocopp_dir, obj_dir)
