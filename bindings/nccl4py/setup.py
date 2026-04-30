import os
import shutil
import subprocess
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py


# Check CUDA_HOME is set and is a valid directory
CUDA_HOME = os.environ.get("CUDA_HOME")
if not CUDA_HOME:
    raise SystemExit("Error: CUDA_HOME is not set")

cuda_path = Path(CUDA_HOME)
if not cuda_path.exists() or not cuda_path.is_dir():
    raise SystemExit(f"Error: CUDA_HOME does not exist or is not a directory: {CUDA_HOME}")
CUDA_INC = str(cuda_path / "include")
SETUP_DIR = Path(__file__).resolve().parent
NCCL_EP_NATIVE_DIR = SETUP_DIR / "native" / "nccl_ep"
NCCL_EP_LIB_NAME = "libnccl_ep.so"
NCCL_EP_PACKAGE_LIB = SETUP_DIR / "nccl" / "ep" / "lib" / NCCL_EP_LIB_NAME
_NCCL_EP_BUILT_LIB = None


def build_nccl_ep(build_temp) -> Path:
    global _NCCL_EP_BUILT_LIB

    if _NCCL_EP_BUILT_LIB is not None and _NCCL_EP_BUILT_LIB.exists():
        return _NCCL_EP_BUILT_LIB

    if not NCCL_EP_NATIVE_DIR.exists():
        raise RuntimeError(f"NCCL EP native source tree not found: {NCCL_EP_NATIVE_DIR}")

    cmake_build_dir = Path(build_temp).resolve() / "nccl_ep"
    cmake_build_dir.mkdir(parents=True, exist_ok=True)

    cuda_architectures = (
        os.environ.get("NCCL_EP_CMAKE_CUDA_ARCHITECTURES")
        or os.environ.get("CMAKE_CUDA_ARCHITECTURES")
        or "90"
    )
    build_type = os.environ.get("NCCL_EP_CMAKE_BUILD_TYPE", "Release")

    configure_cmd = [
        "cmake",
        "-S",
        str(NCCL_EP_NATIVE_DIR),
        "-B",
        str(cmake_build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        f"-DCUDAToolkit_ROOT={cuda_path}",
    ]

    nvcc = cuda_path / "bin" / "nvcc"
    if nvcc.exists():
        configure_cmd.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")

    subprocess.check_call(configure_cmd, cwd=SETUP_DIR)

    build_cmd = [
        "cmake",
        "--build",
        str(cmake_build_dir),
        "--target",
        "nccl_ep_shared",
        "--config",
        build_type,
    ]
    if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
        build_cmd.append("--parallel")

    subprocess.check_call(build_cmd, cwd=SETUP_DIR)

    built_lib = cmake_build_dir / "lib" / NCCL_EP_LIB_NAME
    if not built_lib.exists():
        raise RuntimeError(f"Expected NCCL EP library was not built: {built_lib}")

    NCCL_EP_PACKAGE_LIB.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_lib, NCCL_EP_PACKAGE_LIB)
    _NCCL_EP_BUILT_LIB = NCCL_EP_PACKAGE_LIB
    return _NCCL_EP_BUILT_LIB


def copy_nccl_ep_to_build_lib(library_path: Path, build_lib) -> None:
    destination = Path(build_lib) / "nccl" / "ep" / "lib" / NCCL_EP_LIB_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(library_path, destination)


class build_py(_build_py):
    def run(self):
        build_cmd = self.get_finalized_command("build")
        nccl_ep_lib = build_nccl_ep(build_cmd.build_temp)
        super().run()
        copy_nccl_ep_to_build_lib(nccl_ep_lib, self.build_lib)


class build_ext(_build_ext):
    def run(self):
        nccl_ep_lib = build_nccl_ep(self.build_temp)
        super().run()
        copy_nccl_ep_to_build_lib(nccl_ep_lib, self.build_lib)


ext_modules = [
    "nccl.bindings.nccl"
]


def calculate_modules(module: str):
    module_parts = module.split(".")

    # nccl.bindings.nccl -> nccl/bindings/nccl.pyx
    lowpp_mod = module_parts.copy()
    lowpp_pyx = os.path.join(*lowpp_mod[:-1], f"{lowpp_mod[-1]}.pyx")
    lowpp_mod = ".".join(lowpp_mod)
    lowpp_ext = Extension(
        lowpp_mod,
        sources=[lowpp_pyx],
        include_dirs=[CUDA_INC],
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["dl"],
    )

    # cy variant: nccl.bindings.nccl -> nccl/bindings/cynccl.pyx
    cy_mod = module_parts.copy()
    cy_mod[-1] = f"cy{cy_mod[-1]}"
    cy_mod_pyx = os.path.join(*cy_mod[:-1], f"{cy_mod[-1]}.pyx")
    cy_mod = ".".join(cy_mod)
    cy_ext = Extension(
        cy_mod,
        sources=[cy_mod_pyx],
        include_dirs=[CUDA_INC],
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["dl"],
    )

    # internal variant: source is nccl_linux.pyx, but published module name is nccl.bindings._internal.nccl
    inter_mod = module_parts.copy()
    inter_mod.insert(-1, "_internal")
    inter_mod_pyx = os.path.join(*inter_mod[:-1], f"{inter_mod[-1]}_linux.pyx")
    inter_mod = ".".join(inter_mod)
    inter_ext = Extension(
        inter_mod,
        sources=[inter_mod_pyx],
        include_dirs=[CUDA_INC],
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["dl"],
    )

    # internal variant: insert _internal and use utils.pyx
    inter_utils_mod = module_parts.copy()
    inter_utils_mod.insert(-1, "_internal")
    inter_utils_mod[-1] = "utils"
    inter_utils_mod_pyx = os.path.join(*inter_utils_mod[:-1], f"{inter_utils_mod[-1]}.pyx")
    inter_utils_mod = ".".join(inter_utils_mod)
    inter_utils_ext = Extension(
        inter_utils_mod,
        sources=[inter_utils_mod_pyx],
        include_dirs=[CUDA_INC],
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["dl"],
    )

    return lowpp_ext, cy_ext, inter_ext, inter_utils_ext


# Note: the extension attributes are overwritten in build_extension()
ext_modules = [e for ext in ext_modules for e in calculate_modules(ext)]


compiler_directives = {"embedsignature": True, "show_performance_hints": True, "freethreading_compatible": True}


setup(
    cmdclass={"build_py": build_py, "build_ext": build_ext},
    ext_modules=cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        compiler_directives=compiler_directives,
    ),
    zip_safe=False,
    options={"build_ext": {"inplace": False}},
)
