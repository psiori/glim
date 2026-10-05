cmake_minimum_required(VERSION 3.22)

add_library(glim_cloud_fusion SHARED
  src/fusion.cpp
)
target_include_directories(glim_cloud_fusion PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(glim_cloud_fusion PUBLIC glim)
get_target_property(_gtsam_points_include_dirs gtsam_points::gtsam_points INTERFACE_INCLUDE_DIRECTORIES)
if(_gtsam_points_include_dirs)
  target_include_directories(glim_cloud_fusion BEFORE PUBLIC ${_gtsam_points_include_dirs})
endif()
add_library(glim::cloud_fusion ALIAS glim_cloud_fusion)

if(BUILD_GLIM_CLOUD_FUSION_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(glim_cloud_fusion_test tests/test_fusion.cpp)
  target_link_libraries(glim_cloud_fusion_test PRIVATE glim_cloud_fusion GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(glim_cloud_fusion_test DISCOVERY_MODE PRE_TEST)
endif()

install(TARGETS glim_cloud_fusion EXPORT glim_cloud_fusionTargets
  LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
