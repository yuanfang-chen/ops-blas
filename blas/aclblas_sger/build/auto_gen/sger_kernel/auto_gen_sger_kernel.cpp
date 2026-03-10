#ifndef __SGER_KERNEL__KERNEL_FUN_H__
#define __SGER_KERNEL__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define sger_kernel_do sger_kernel_do_origin
#include "/mnt/model/ccq/diffusers/aclblas_sger/sger_kernel.cpp"

#undef sger_kernel_do
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_sger_kernel_do_kernel(
__attribute__((cce_global)) uint8_t* xGm, __attribute__((cce_global)) uint8_t* yGm, __attribute__((cce_global)) uint8_t* aGm, __attribute__((cce_global)) uint8_t* tilingBuf, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = aGm;
#else
    workspace_param = tilingBuf;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    aGm = workspace_usr;
#else
    tilingBuf = workspace_usr;
#endif
#endif
    sger_kernel_do_origin(xGm, yGm, aGm, tilingBuf);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    pipe_barrier(PIPE_ALL);
    dsb(mem_dsb_t::DSB_ALL);
    dci();
#endif
}

#endif
