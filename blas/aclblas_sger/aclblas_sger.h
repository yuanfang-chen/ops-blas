#ifndef ACLBLAS_SGER_H
#define ACLBLAS_SGER_H

#include "acl/acl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ACLBLAS_STATUS_SUCCESS = 0,
    ACLBLAS_STATUS_NOT_INITIALIZED = 1,
    ACLBLAS_STATUS_ALLOC_FAILED = 3,
    ACLBLAS_STATUS_INVALID_VALUE = 7,
    ACLBLAS_STATUS_EXECUTION_FAILED = 13,
    ACLBLAS_STATUS_INTERNAL_ERROR = 14,
} aclblasStatus_t;

typedef struct aclblasContext *aclblasHandle_t;

aclblasStatus_t aclblasCreate(aclblasHandle_t *handle);
aclblasStatus_t aclblasDestroy(aclblasHandle_t handle);
aclblasStatus_t aclblasSetStream(aclblasHandle_t handle, aclrtStream stream);
aclblasStatus_t aclblasGetStream(aclblasHandle_t handle, aclrtStream *stream);

aclblasStatus_t aclblasSger(
    aclblasHandle_t handle,
    int m, int n,
    const float *alpha,
    const float *x, int incx,
    const float *y, int incy,
    float *A, int lda);

#ifdef __cplusplus
}
#endif

#endif
