# src/fsw/fsw_embedding/build_fsw_embedding.py

import subprocess
import os
import platform
import shutil

def main(nvcc_path=None, verbose=False):
    cu_file_name = "fsw_embedding.cu"
    so_file_name = "libfsw_embedding.so"

    nvcc_at_search_path = shutil.which("nvcc")

    cuda_home = os.environ.get("CUDA_HOME")
    nvcc_at_cuda_home = os.path.join(cuda_home, "bin", "nvcc") if cuda_home is not None else None

    nvcc_at_default_path = "/usr/local/cuda/bin/nvcc"

    if nvcc_at_search_path is not None and os.path.isfile(nvcc_at_search_path):
        nvcc = nvcc_at_search_path
    elif nvcc_at_cuda_home is not None and os.path.isfile(nvcc_at_cuda_home):
        nvcc = nvcc_at_cuda_home
    elif nvcc_at_default_path is not None and os.path.isfile(nvcc_at_default_path):
        nvcc = nvcc_at_default_path
    else:
        raise RuntimeError(
            "Could not find `nvcc`. Make sure CUDA is installed and either:\n"
            "- `nvcc` is in your PATH\n"
            "- or `CUDA_HOME` is set\n"
            "- or it exists at /usr/local/cuda/bin/nvcc"
        )

    cu_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), cu_file_name))
    so_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), so_file_name))

    if not os.path.isfile(cu_file_path):
        raise FileNotFoundError(f"CUDA source file not found: {cu_file_path}")

    supported_archs = [60, 61, 70, 75, 80, 86]

    base_flags = ["-shared", "-O3", "-Wno-deprecated-gpu-targets"]

    if verbose:
        base_flags += ["-Xptxas", "-v"]

    if platform.system() == "Windows":
        platform_specific_flags = ["-Xcompiler", "/MD /O2"]
    else:
        platform_specific_flags = ["--compiler-options", "-O3 -fPIC"]


    # Try to detect the current architecture and add it to supported_archs
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            current_arch = f"{major}{minor}"

            if verbose:
                print('Current architecture: ', current_arch)

            if int(current_arch) not in supported_archs:
                supported_archs.append(int(current_arch))

    except Exception as e:
        print(f"[fsw-build] Warning: could not determine current CUDA device architecture: {e}")

    arch_flags = [f"-gencode=arch=compute_{sm},code=sm_{sm}" for sm in supported_archs]

    cmd = [nvcc] + base_flags + platform_specific_flags + ["-o", so_file_path, cu_file_path] + arch_flags

    #print(f"Building {so_file_name} ... ", end="")

    try:
        cmd = cmd
        subprocess.check_call(cmd)
    except Exception as e:
        print(f"[fsw-build] Error: failed to execute compilation command: {cmd}\n")
        raise e

    #print("Done.")

if __name__ == "__main__":
    main()
