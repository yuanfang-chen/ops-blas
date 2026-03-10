#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#include "acl/acl.h"
#include "sger_tiling.h"

extern "C" aclError aclrtlaunch_sger_kernel_do(
    uint32_t blockDim, aclrtStream stream,
    void* xGm, void* yGm, void* aGm, void* tilingBuf);

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

bool checkResult(const float* ref, const float* out, int m, int n, int lda,
                 float rtol, float atol)
{
    int errCount = 0;
    float maxDiff = 0.0f;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            int idx = j * lda + i;
            float diff = fabsf(ref[idx] - out[idx]);
            float tol = atol + rtol * fabsf(ref[idx]);
            if (diff > maxDiff) maxDiff = diff;
            if (diff > tol) {
                if (errCount < 5) {
                    printf("  MISMATCH [%d,%d]: ref=%.6f got=%.6f diff=%.6f\n",
                           i, j, ref[idx], out[idx], diff);
                }
                errCount++;
            }
        }
    }
    printf("  Max abs diff: %.6e, errors: %d/%d\n", maxDiff, errCount, m * n);
    return errCount == 0;
}

void runTest(int m, int n, float alpha, int incx, int incy, int lda,
             int blockDim, int warmup, int repeat)
{
    printf("=== SGER m=%d n=%d alpha=%.2f incx=%d incy=%d lda=%d blk=%d ===\n",
           m, n, alpha, incx, incy, lda, blockDim);

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

    void *dX, *dY, *dA, *dTiling;
    aclrtMalloc(&dX, xLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dY, yLen * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dA, aSize * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dTiling, sizeof(SgerTilingData), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    SgerTilingData tiling;
    tiling.alpha = alpha;
    tiling.m = m;
    tiling.n = n;
    tiling.lda = lda;
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.blockDim = blockDim;

    aclrtMemcpy(dX, xLen * sizeof(float), hX.data(), xLen * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dY, yLen * sizeof(float), hY.data(), yLen * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dTiling, sizeof(SgerTilingData), &tiling, sizeof(SgerTilingData),
                ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMemcpy(dA, aSize * sizeof(float), hA.data(), aSize * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtlaunch_sger_kernel_do(blockDim, stream, dX, dY, dA, dTiling);
    aclrtSynchronizeStream(stream);

    std::vector<float> hOut(aSize);
    aclrtMemcpy(hOut.data(), aSize * sizeof(float), dA, aSize * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);

    bool pass = checkResult(hRef.data(), hOut.data(), m, n, lda, 1e-5f, 1e-5f);
    printf("  Correctness: %s\n", pass ? "PASS" : "FAIL");

    for (int w = 0; w < warmup; w++) {
        aclrtMemcpy(dA, aSize * sizeof(float), hA.data(), aSize * sizeof(float),
                    ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtlaunch_sger_kernel_do(blockDim, stream, dX, dY, dA, dTiling);
        aclrtSynchronizeStream(stream);
    }

    aclrtEvent startEvent, endEvent;
    aclrtCreateEvent(&startEvent);
    aclrtCreateEvent(&endEvent);

    float totalMs = 0.0f;
    for (int r = 0; r < repeat; r++) {
        aclrtMemcpy(dA, aSize * sizeof(float), hA.data(), aSize * sizeof(float),
                    ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtSynchronizeStream(stream);

        aclrtRecordEvent(startEvent, stream);
        aclrtlaunch_sger_kernel_do(blockDim, stream, dX, dY, dA, dTiling);
        aclrtRecordEvent(endEvent, stream);
        aclrtSynchronizeStream(stream);

        float ms = 0.0f;
        aclrtEventElapsedTime(&ms, startEvent, endEvent);
        totalMs += ms;
    }

    float avgMs = totalMs / repeat;
    double gflops = 2.0 * m * n / (avgMs * 1e6);
    printf("  Kernel avg: %.3f ms (%.2f us), %.4f GFLOPS\n\n", avgMs, avgMs * 1000.0f, gflops);

    aclrtDestroyEvent(startEvent);
    aclrtDestroyEvent(endEvent);
    aclrtFree(dX);
    aclrtFree(dY);
    aclrtFree(dA);
    aclrtFree(dTiling);
    aclrtDestroyStream(stream);
}

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    int blockDim = 8;

    runTest(64, 64, 1.0f, 1, 1, 64, blockDim, 5, 20);
    runTest(128, 128, 2.5f, 1, 1, 128, blockDim, 5, 20);
    runTest(256, 256, 0.5f, 1, 1, 256, blockDim, 5, 20);
    runTest(512, 512, 1.0f, 1, 1, 512, blockDim, 5, 20);
    runTest(1024, 1024, 1.0f, 1, 1, 1024, blockDim, 5, 20);
    runTest(100, 200, 1.5f, 1, 1, 100, blockDim, 5, 20);
    runTest(37, 53, 0.7f, 1, 1, 37, blockDim, 5, 20);
    runTest(1, 1, 3.0f, 1, 1, 1, 1, 5, 20);
    runTest(128, 128, 1.0f, 2, 3, 128, blockDim, 5, 20);
    runTest(2048, 2048, 1.0f, 1, 1, 2048, blockDim, 5, 20);
    runTest(4096, 4096, 1.0f, 1, 1, 4096, blockDim, 5, 20);

    aclrtResetDevice(0);
    aclFinalize();
    printf("All tests completed.\n");
    return 0;
}
