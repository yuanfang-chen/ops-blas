# 生成 ascendc kernel 库
ascendc_library(kernels STATIC
    ${CMAKE_SOURCE_DIR}/src/copy/scopy_kernel.cpp
)

add_library(ops_blas STATIC
    ${CMAKE_SOURCE_DIR}/src/copy/scopy_host.cpp
)

target_include_directories(ops_blas PUBLIC
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_SOURCE_DIR}/include
)
# 测试程序
add_executable(scopy_test
    scopy_test.cpp
)

target_include_directories(scopy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(scopy_test PRIVATE
    kernels
    ops_blas
)

target_link_libraries(ops_blas PUBLIC kernels)
