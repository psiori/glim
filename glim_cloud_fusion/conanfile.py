from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout


class GlimCloudFusionConan(ConanFile):
    name = "glim_cloud_fusion"
    version = "0.1.0"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": True, "fPIC": True}
    exports_sources = "glim_cloud_fusion/*", "CMakeLists.txt", "include/*", "src/*"

    def requirements(self):
        self.requires("glim/1.2.2")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["BUILD_GLIM_CLOUD_FUSION"] = True
        tc.variables["BUILD_GLIM_CLOUD_FUSION_TESTS"] = False
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "glim_cloud_fusion")
        self.cpp_info.set_property("cmake_target_name", "glim::cloud_fusion")
