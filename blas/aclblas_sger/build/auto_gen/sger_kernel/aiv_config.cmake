set(MIX_SOURCES
)
set(AIV_SOURCES
    /mnt/model/ccq/diffusers/aclblas_sger/build/auto_gen/sger_kernel/auto_gen_sger_kernel.cpp
)
set_source_files_properties(/mnt/model/ccq/diffusers/aclblas_sger/build/auto_gen/sger_kernel/auto_gen_sger_kernel.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_sger_kernel_do_kernel=sger_kernel_do_0;ONE_CORE_DUMP_SIZE=1048576"
)
