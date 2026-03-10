#ifndef HEADER_ACLRTLAUNCH_SGER_KERNEL_DO_H
#define HEADER_ACLRTLAUNCH_SGER_KERNEL_DO_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_sger_kernel_do(uint32_t blockDim, aclrtStream stream, void* xGm, void* yGm, void* aGm, void* tilingBuf);
#endif
