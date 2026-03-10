# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/device_precompile_project")
  file(MAKE_DIRECTORY "/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/device_precompile_project")
endif()
file(MAKE_DIRECTORY
  "/mnt/model/ccq/diffusers/aclblas_sger/build/sger_kernel_precompile-prefix/src/sger_kernel_precompile-build"
  "/mnt/model/ccq/diffusers/aclblas_sger/build/sger_kernel_precompile-prefix"
  "/mnt/model/ccq/diffusers/aclblas_sger/build/sger_kernel_precompile-prefix/tmp"
  "/mnt/model/ccq/diffusers/aclblas_sger/build/sger_kernel_precompile-prefix/src/sger_kernel_precompile-stamp"
  "/mnt/model/ccq/diffusers/aclblas_sger/build/sger_kernel_precompile-prefix/src"
  "/mnt/model/ccq/diffusers/aclblas_sger/build/sger_kernel_precompile-prefix/src/sger_kernel_precompile-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/mnt/model/ccq/diffusers/aclblas_sger/build/sger_kernel_precompile-prefix/src/sger_kernel_precompile-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/mnt/model/ccq/diffusers/aclblas_sger/build/sger_kernel_precompile-prefix/src/sger_kernel_precompile-stamp${cfgdir}") # cfgdir has leading slash
endif()
