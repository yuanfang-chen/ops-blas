cmake_minimum_required(VERSION 3.16)
project(aclblas_sger)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(ASCEND_INSTALL_PATH "/usr/local/Ascend/ascend-toolkit/latest")
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_INSTALL_PATH})
set(SOC_VERSION "ascend910b2" CACHE STRING "SoC version")

include(${ASCEND_INSTALL_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

ascendc_library(sger_kernel STATIC
    sger_kernel.cpp
)

add_executable(test_sger test_sger.cpp)
target_link_libraries(test_sger PRIVATE
    sger_kernel ascendcl runtime ${ASCENDC_RUNTIME_STATIC_TARGET} host_intf_pub
)

add_executable(test_api test_api.cpp aclblas_sger.cpp)
target_link_libraries(test_api PRIVATE
    sger_kernel ascendcl runtime ${ASCENDC_RUNTIME_STATIC_TARGET} host_intf_pub
)

add_executable(test_verbose test_verbose.cpp aclblas_sger.cpp)
target_link_libraries(test_verbose PRIVATE
    sger_kernel ascendcl runtime ${ASCENDC_RUNTIME_STATIC_TARGET} host_intf_pub
)
