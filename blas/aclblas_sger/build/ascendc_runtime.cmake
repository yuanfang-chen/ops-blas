add_library(ascendc_runtime_obj OBJECT IMPORTED)
set_target_properties(ascendc_runtime_obj PROPERTIES
    IMPORTED_OBJECTS "/mnt/model/ccq/diffusers/aclblas_sger/build/ascendc_runtime.cpp.o;/mnt/model/ccq/diffusers/aclblas_sger/build/aicpu_rt.cpp.o;/mnt/model/ccq/diffusers/aclblas_sger/build/ascendc_elf_tool.c.o"
)
