#include "aclblas_sger.h"
#include "sger_tiling.h"
#include <cstdlib>
#include <cstring>
#include <new>

extern "C" aclError aclrtlaunch_sger_kernel_do(
    uint32_t blockDim, aclrtStream stream,
    void* xGm, void* yGm, void* aGm, void* tilingBuf);

struct aclblasContext {
    aclrtStream stream;
    void* tilingDev;
};

aclblasStatus_t aclblasCreate(aclblasHandle_t *handle)
{
    if (!handle) return ACLBLAS_STATUS_INVALID_VALUE;
    aclblasContext* ctx = new (std::nothrow) aclblasContext();
    if (!ctx) return ACLBLAS_STATUS_ALLOC_FAILED;
    ctx->stream = nullptr;
    ctx->tilingDev = nullptr;
    aclError ret = aclrtMalloc(&ctx->tilingDev, sizeof(SgerTilingData),
                               ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        delete ctx;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    *handle = ctx;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasDestroy(aclblasHandle_t handle)
{
    if (!handle) return ACLBLAS_STATUS_INVALID_VALUE;
    if (handle->tilingDev) aclrtFree(handle->tilingDev);
    delete handle;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSetStream(aclblasHandle_t handle, aclrtStream stream)
{
    if (!handle) return ACLBLAS_STATUS_INVALID_VALUE;
    handle->stream = stream;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasGetStream(aclblasHandle_t handle, aclrtStream *stream)
{
    if (!handle || !stream) return ACLBLAS_STATUS_INVALID_VALUE;
    *stream = handle->stream;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSger(
    aclblasHandle_t handle,
    int m, int n,
    const float *alpha,
    const float *x, int incx,
    const float *y, int incy,
    float *A, int lda)
{
    if (!handle) return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || !alpha || !x || !y || !A || lda < m || lda < 1
        || incx == 0 || incy == 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;

    int blockDim = 8;
    if (m < 64) blockDim = 1;
    else if (m < 256) blockDim = 4;

    SgerTilingData tiling;
    tiling.alpha = *alpha;
    tiling.m = m;
    tiling.n = n;
    tiling.lda = lda;
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.blockDim = blockDim;

    aclError ret = aclrtMemcpy(handle->tilingDev, sizeof(SgerTilingData),
                               &tiling, sizeof(SgerTilingData),
                               ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) return ACLBLAS_STATUS_INTERNAL_ERROR;

    ret = aclrtlaunch_sger_kernel_do(
        blockDim, handle->stream,
        (void*)x, (void*)y, (void*)A, handle->tilingDev);
    if (ret != ACL_SUCCESS) return ACLBLAS_STATUS_EXECUTION_FAILED;

    return ACLBLAS_STATUS_SUCCESS;
}
