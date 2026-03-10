#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

#include "acl/acl.h"
#include "aclblas_sger.h"

void cpuSger(int m, int n, float alpha, const float* x, int incx,
             const float* y, int incy, float* A, int lda)
{
    for (int j = 0; j < n; j++) {
        float yval = alpha * y[j * incy];
        for (int i = 0; i < m; i++) {
            A[j * lda + i] += x[i * incx] * yval;
        }
    }
}

bool verify(const float* ref, const float* out, int m, int n, int lda)
{
    int errCount = 0;
    float maxDiff = 0.0f;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            int idx = j * lda + i;
            float diff = fabsf(ref[idx] - out[idx]);
            float tol = 1e-5f + 1e-5f * fabsf(ref[idx]);
            if (diff > maxDiff) maxDiff = diff;
            if (diff > tol) errCount++;
        }
    }
    printf("  Max diff: %.6e, errors: %d/%d\n", maxDiff, errCount, m * n);
    return errCount == 0;
}

void testSger(int m, int n, float alpha, int incx, int incy, int lda,
              aclblasHandle_t handle, int warmup, int repeat)
{
    printf("=== aclblasSger m=%d n=%d alpha=%.2f incx=%d incy=%d lda=%d ===\n",
           m, n, alpha, incx, incy, lda);

    int xLen = 1 + (m - 1) * abs(incx);
    int yLen = 1 + (n - 1) * abs(incy);
    int aSize = lda * n;

    std::vector<float> hX(xLen), hY(yLen), hA(aSize), hRef(aSize);
    srand(42);
    for (int i = 0; i < xLen; i++) hX[i] = (float)(rand() % 1000 - 500) / 500.0f;
    for (int i = 0; i < yLen; i++) hY[i] = (float)(rand() % 1000 - 500) / 500.0f;
    for (int i = 0; i < aSize; i++) {
        hA[i] = (float)(rand() % 1000 - 500) / 500.0f;
        hRef[i] = hA[i];
    }
    cpuSger(m, n, alpha, hX.data(), incx, hY.data(), incy, hRef.data(), lda);

    void *dX, *dY, *dA;
    aclrtMalloc(&dX, xLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dY, yLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dA, aSize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dX, xLen * sizeof(float), hX.data(), xLen * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dY, yLen * sizeof(float), hY.data(), yLen * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMemcpy(dA, aSize * sizeof(float), hA.data(), aSize * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclblasStatus_t st = aclblasSger(handle, m, n, &alpha,
                                      (float*)dX, incx,
                                      (float*)dY, incy,
                                      (float*)dA, lda);
    aclrtStream stream;
    aclblasGetStream(handle, &stream);
    aclrtSynchronizeStream(stream);

    if (st != ACLBLAS_STATUS_SUCCESS) {
        printf("  ERROR: aclblasSger returned %d\n\n", st);
        aclrtFree(dX); aclrtFree(dY); aclrtFree(dA);
        return;
    }

    std::vector<float> hOut(aSize);
    aclrtMemcpy(hOut.data(), aSize * sizeof(float), dA, aSize * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
    bool pass = verify(hRef.data(), hOut.data(), m, n, lda);
    printf("  Correctness: %s\n", pass ? "PASS" : "FAIL");

    for (int w = 0; w < warmup; w++) {
        aclrtMemcpy(dA, aSize * sizeof(float), hA.data(), aSize * sizeof(float),
                    ACL_MEMCPY_HOST_TO_DEVICE);
        aclblasSger(handle, m, n, &alpha, (float*)dX, incx,
                    (float*)dY, incy, (float*)dA, lda);
        aclrtSynchronizeStream(stream);
    }

    aclrtEvent startEvt, endEvt;
    aclrtCreateEvent(&startEvt);
    aclrtCreateEvent(&endEvt);
    float totalMs = 0;
    for (int r = 0; r < repeat; r++) {
        aclrtMemcpy(dA, aSize * sizeof(float), hA.data(), aSize * sizeof(float),
                    ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtSynchronizeStream(stream);
        aclrtRecordEvent(startEvt, stream);
        aclblasSger(handle, m, n, &alpha, (float*)dX, incx,
                    (float*)dY, incy, (float*)dA, lda);
        aclrtRecordEvent(endEvt, stream);
        aclrtSynchronizeStream(stream);
        float ms = 0;
        aclrtEventElapsedTime(&ms, startEvt, endEvt);
        totalMs += ms;
    }
    float avgMs = totalMs / repeat;
    double gflops = 2.0 * m * n / (avgMs * 1e6);
    printf("  Kernel avg: %.3f ms (%.2f us), %.4f GFLOPS\n\n", avgMs, avgMs * 1000, gflops);

    aclrtDestroyEvent(startEvt);
    aclrtDestroyEvent(endEvt);
    aclrtFree(dX); aclrtFree(dY); aclrtFree(dA);
}

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle;
    aclblasCreate(&handle);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    testSger(64, 64, 1.0f, 1, 1, 64, handle, 5, 20);
    testSger(128, 128, 2.5f, 1, 1, 128, handle, 5, 20);
    testSger(256, 256, 0.5f, 1, 1, 256, handle, 5, 20);
    testSger(512, 512, 1.0f, 1, 1, 512, handle, 5, 20);
    testSger(1024, 1024, 1.0f, 1, 1, 1024, handle, 5, 20);
    testSger(2048, 2048, 1.0f, 1, 1, 2048, handle, 5, 20);
    testSger(4096, 4096, 1.0f, 1, 1, 4096, handle, 5, 20);
    testSger(100, 200, 1.5f, 1, 1, 100, handle, 5, 20);
    testSger(37, 53, 0.7f, 1, 1, 37, handle, 5, 20);
    testSger(1, 1, 3.0f, 1, 1, 1, handle, 5, 20);
    testSger(128, 128, 1.0f, 2, 3, 128, handle, 5, 20);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    printf("All API tests completed.\n");
    return 0;
}
