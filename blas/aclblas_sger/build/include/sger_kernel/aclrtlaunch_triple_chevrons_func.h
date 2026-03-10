
#ifndef HEADER_ACLRTLAUNCH_SGER_KERNEL_DO_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_SGER_KERNEL_DO_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_sger_kernel_do(uint32_t blockDim, void* stream, void* xGm, void* yGm, void* aGm, void* tilingBuf);

inline uint32_t sger_kernel_do(uint32_t blockDim, void* hold, void* stream, void* xGm, void* yGm, void* aGm, void* tilingBuf)
{
    (void)hold;
    return aclrtlaunch_sger_kernel_do(blockDim, stream, xGm, yGm, aGm, tilingBuf);
}

#endif
