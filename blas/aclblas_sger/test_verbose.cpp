#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include "acl/acl.h"
#include "aclblas_sger.h"

void printMatrix(const char* name, const float* A, int m, int n, int lda)
{
    printf("  %s (%dx%d, lda=%d):\n", name, m, n, lda);
    for (int i = 0; i < m; i++) {
        printf("    [");
        for (int j = 0; j < n; j++) {
            printf("%9.4f", A[j * lda + i]);
            if (j < n - 1) printf(",");
        }
        printf(" ]\n");
    }
}

void printVector(const char* name, const float* v, int len, int inc)
{
    printf("  %s (len=%d, inc=%d): [", name, len, inc);
    for (int i = 0; i < len; i++) {
        printf("%8.4f", v[i * inc]);
        if (i < len - 1) printf(",");
    }
    printf(" ]\n");
}

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

void runVerboseTest(int m, int n, float alpha, int incx, int incy, int lda,
                    aclblasHandle_t handle)
{
    printf("\n");
    printf("================================================================\n");
    printf("  SGER: A = alpha * x * y^T + A   (column-major)\n");
    printf("  m=%d, n=%d, alpha=%.4f, incx=%d, incy=%d, lda=%d\n",
           m, n, alpha, incx, incy, lda);
    printf("================================================================\n");

    int xLen = 1 + (m - 1) * abs(incx);
    int yLen = 1 + (n - 1) * abs(incy);
    int aSize = lda * n;

    std::vector<float> hX(xLen), hY(yLen), hA(aSize);
    std::vector<float> hCpuOut(aSize), hOuter(m * n);

    srand(123);
    for (int i = 0; i < xLen; i++) hX[i] = (float)(rand() % 100 - 50) / 10.0f;
    for (int i = 0; i < yLen; i++) hY[i] = (float)(rand() % 100 - 50) / 10.0f;
    for (int i = 0; i < aSize; i++) hA[i] = (float)(rand() % 100 - 50) / 10.0f;
    for (int i = 0; i < aSize; i++) hCpuOut[i] = hA[i];

    printf("\n[1] Input\n");
    printf("  alpha = %.4f\n", alpha);
    printVector("x", hX.data(), m, incx);
    printVector("y", hY.data(), n, incy);
    printMatrix("A_input", hA.data(), m, n, lda);

    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            hOuter[j * m + i] = alpha * hX[i * incx] * hY[j * incy];

    printf("\n[2] Intermediate: alpha * x * y^T\n");
    printMatrix("alpha*x*yT", hOuter.data(), m, n, m);

    printf("\n[3] CPU Output   (A_cpu = alpha*x*yT + A_input)\n");
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpuSger(m, n, alpha, hX.data(), incx, hY.data(), incy, hCpuOut.data(), lda);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuUs = std::chrono::duration<double, std::micro>(cpuEnd - cpuStart).count();
    printMatrix("A_cpu", hCpuOut.data(), m, n, lda);
    printf("  CPU time: %.2f us\n", cpuUs);

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

    aclrtStream stream;
    aclblasGetStream(handle, &stream);

    aclrtEvent startEvt, endEvt;
    aclrtCreateEvent(&startEvt);
    aclrtCreateEvent(&endEvt);

    aclrtRecordEvent(startEvt, stream);
    aclblasSger(handle, m, n, &alpha,
                (float*)dX, incx, (float*)dY, incy, (float*)dA, lda);
    aclrtRecordEvent(endEvt, stream);
    aclrtSynchronizeStream(stream);

    float npuMs = 0;
    aclrtEventElapsedTime(&npuMs, startEvt, endEvt);

    std::vector<float> hNpuOut(aSize);
    aclrtMemcpy(hNpuOut.data(), aSize * sizeof(float), dA, aSize * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);

    printf("\n[4] NPU Output   (A_npu = alpha*x*yT + A_input)\n");
    printMatrix("A_npu", hNpuOut.data(), m, n, lda);
    printf("  NPU kernel time: %.2f us\n", npuMs * 1000.0);

    printf("\n[5] Element-wise Comparison (CPU vs NPU)\n");
    printf("  %6s  %12s  %12s  %12s  %6s\n",
           "[i,j]", "CPU", "NPU", "abs_diff", "Match");
    printf("  %6s  %12s  %12s  %12s  %6s\n",
           "------", "------------", "------------", "------------", "------");
    float maxDiff = 0.0f;
    int errCount = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            int idx = j * lda + i;
            float diff = fabsf(hCpuOut[idx] - hNpuOut[idx]);
            if (diff > maxDiff) maxDiff = diff;
            float tol = 1e-5f + 1e-5f * fabsf(hCpuOut[idx]);
            bool ok = diff <= tol;
            if (!ok) errCount++;
            printf("  [%d,%d]  %12.6f  %12.6f  %12.2e  %s\n",
                   i, j, hCpuOut[idx], hNpuOut[idx], diff,
                   ok ? "  OK" : "FAIL");
        }
    }
    printf("\n  Summary: max_abs_diff = %.2e, errors = %d/%d\n",
           maxDiff, errCount, m * n);
    printf("  Result:  %s\n",
           errCount == 0 ? "PASS (CPU == NPU within fp32 precision)"
                         : "FAIL");

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

    runVerboseTest(4, 3, 1.5f, 1, 1, 4, handle);
    runVerboseTest(3, 4, 2.0f, 1, 1, 3, handle);
    runVerboseTest(5, 3, 0.5f, 2, 1, 5, handle);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
