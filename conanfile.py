import os

from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy

_EXPORT_EXCLUDES = (
    ".git",
    ".git/*",
    "build",
    "build/*",
    "build_*",
    "build_*/*",
    "install",
    "install/*",
    "cmake-build-*",
    ".cache",
    ".cache/*",
    ".vscode",
    ".idea",
    "__pycache__",
    "__pycache__/*",
    "*.pyc",
    ".DS_Store",
    "conanbuild.sh",
    "conanbuildenv-*",
    "conanrun.sh",
    "conanrunenv-*",
    "deactivate_conanbuild.sh",
    "deactivate_conanrun.sh",
    "compile_commands.json",
    "CMakeUserPresets.json",
    "output_*.pcd",
    "output_*.txt",
)


def _set_gtsam_dir(tc, conanfile):
    try:
        gtsam = conanfile.dependencies["gtsam"]
    except (KeyError, AttributeError):
        return
    gtsam_dir = os.path.join(gtsam.package_folder, "lib", "cmake", "GTSAM")
    if os.path.isdir(gtsam_dir):
        tc.variables["GTSAM_DIR"] = gtsam_dir


class GlimConan(ConanFile):
    name = "glim"
    version = "1.2.2"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "build_with_viewer": [True, False],
        "build_with_cuda": [True, False],
        "build_with_march_native": [True, False],
        "build_glim_cloud_fusion": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "build_with_viewer": False,
        "build_with_cuda": False,
        "build_with_march_native": True,
        "build_glim_cloud_fusion": True,
    }

    def export_sources(self):
        copy(self, "*", self.recipe_folder, self.export_sources_folder, excludes=_EXPORT_EXCLUDES)

    def requirements(self):
        self.requires("gtsam/4.3a1")
        self.requires("gtsam_points/1.2.2")
        self.requires("eigen/3.4.0")
        self.requires("boost/1.83.0")
        self.requires("fmt/10.2.1", override=True)
        self.requires("spdlog/1.12.0")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["BUILD_WITH_VIEWER"] = self.options.build_with_viewer
        tc.variables["BUILD_WITH_CUDA"] = self.options.build_with_cuda
        tc.variables["BUILD_WITH_MARCH_NATIVE"] = self.options.build_with_march_native
        tc.variables["BUILD_WITH_OPENMP"] = True
        tc.variables["BUILD_GLIM_CLOUD_FUSION"] = self.options.build_glim_cloud_fusion
        tc.variables["BUILD_GLIM_CLOUD_FUSION_TESTS"] = False
        _set_gtsam_dir(tc, self)
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.libdirs = ["lib"]
        self.cpp_info.libs = ["glim"]
        self.cpp_info.builddirs.append("lib/cmake/glim")
        if self.options.build_glim_cloud_fusion:
            self.cpp_info.libs.append("glim_cloud_fusion")
            self.cpp_info.builddirs.append("lib/cmake/glim_cloud_fusion")
