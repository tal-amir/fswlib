# src/fswlib/build.py

import sys

import argparse
import os
import shutil
import subprocess


def find_nvcc(custom_nvcc_path: str | None = None) -> str:
    """
    Resolve the path to `nvcc`, prioritizing:
    1. Manual override, given in custom_nvcc_path
    2. Environment variable (FSW_NVCC_PATH)
    3. Search path (PATH)
    4. CUDA_HOME
    5. Default system path (/usr/local/cuda/bin/nvcc)

    If
    """
    candidates = []

    if custom_nvcc_path is None:
        env_path = os.environ.get("FSW_NVCC_PATH")
        if env_path:
            candidates.append(env_path)

        path_search = shutil.which("nvcc")
        if path_search:
            candidates.append(path_search)

        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            candidates.append(os.path.join(cuda_home, "bin", "nvcc"))

        candidates.append("/usr/local/cuda/bin/nvcc")
    else:
        candidates.append(custom_nvcc_path)

    for path in candidates:
        if path and os.path.isfile(path):
            if not is_valid_nvcc(path):
                raise RuntimeError(f"`nvcc` was found at '{path}' but is not a valid working CUDA compiler. "
                                   f"Make sure it is executable and correctly installed.")
            return path

    if len(candidates) > 1:
        error_str = "Could not find `nvcc`. Checked paths:\n" + "\n".join(candidates)
    else:
        error_str = f"Could not find `nvcc` in path {candidates[0]}"

    if custom_nvcc_path is None:
        error_str += '\nTry providing a custom nvcc path via the --nvcc <PATH> argument or the FSW_NVCC_PATH environment variable.'

    raise RuntimeError(error_str)


def is_valid_nvcc(path: str) -> bool:
    try:
        result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0 and "Cuda compilation tools" in result.stdout
    except Exception:
        return False


def build_fsw_embedding(nvcc_path: str, verbose: bool, dummy: bool, clean: bool):
    from fswlib.fsw_embedding import build_fsw_embedding

    module_name = "fsw_embedding"

    if clean:
        msg = f"[fswlib-build] Deleting {module_name} binaries"
    else:
        msg = f"[fswlib-build] Building {module_name}"

    if verbose:
        print(f"{msg}: ", flush=True)
    else:
        print(f"{msg}... ", end="", flush=True)

    build_fsw_embedding.main(nvcc_path=nvcc_path, verbose=verbose, dummy=dummy, clean=clean)

    if verbose and clean:
        print('done cleaning fsw_embedding')
    elif verbose:
        print('done building fsw_embedding')
    else:
        print('done')


def clean_entrypoint():
    import sys
    sys.argv = [sys.argv[0], "--clean"]
    main()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the FSW embedding CUDA extension.")
    parser.add_argument(
        "--nvcc",
        metavar="<PATH>",
        help=(
            "Path to the `nvcc` compiler (e.g. /usr/local/cuda/bin/nvcc). "
            "If not provided, will try $FSW_NVCC_PATH, then PATH, then default locations."
        ),
        default=None    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output", default=False)
    parser.add_argument("--dummy", action="store_true", help="Generate dummy binaries instead of compiling", default=False)
    parser.add_argument("--clean", action="store_true", help="Instead of building, delete previously compiled binaries", default=False)
    args = parser.parse_args(argv)

    nvcc_path = find_nvcc(args.nvcc)

    build_fsw_embedding(nvcc_path=nvcc_path, verbose=args.verbose, dummy=args.dummy, clean=args.clean)


if __name__ == "__main__":
    main()
