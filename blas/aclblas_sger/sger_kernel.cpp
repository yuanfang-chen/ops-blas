#include "kernel_operator.h"
#include "sger_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ALIGN_FP32 = 8;

class KernelSger {
public:
    __aicore__ inline KernelSger() {}

    __aicore__ inline void Init(
        GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm,
        float alpha, int32_t m, int32_t n, int32_t lda,
        int32_t incx, int32_t incy)
    {
        this->alpha = alpha;
        this->m = m;
        this->n = n;
        this->lda = lda;
        this->incx = incx;
        this->incy = incy;

        int32_t blockIdx = GetBlockIdx();
        int32_t blockNum = GetBlockNum();

        int32_t rowsPerBlock = (m + blockNum - 1) / blockNum;
        this->rowStart = blockIdx * rowsPerBlock;
        this->rowEnd = (rowStart + rowsPerBlock < m) ? (rowStart + rowsPerBlock) : m;
        if (rowStart >= m) {
            this->rowStart = m;
            this->rowEnd = m;
        }
        this->tileRows = rowEnd - rowStart;
        this->alignedTileRows = ((tileRows + ALIGN_FP32 - 1) / ALIGN_FP32) * ALIGN_FP32;
        if (alignedTileRows == 0) alignedTileRows = ALIGN_FP32;
        this->alignedN = ((n + ALIGN_FP32 - 1) / ALIGN_FP32) * ALIGN_FP32;
        if (alignedN == 0) alignedN = ALIGN_FP32;
        this->isAligned = (tileRows % ALIGN_FP32 == 0);

        xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xGm));
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yGm));
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm));

        pipe.InitBuffer(inQueueX, 1, alignedTileRows * sizeof(float));
        pipe.InitBuffer(inQueueY, 1, alignedN * sizeof(float));
        pipe.InitBuffer(inQueueA, BUFFER_NUM, alignedTileRows * sizeof(float));
        pipe.InitBuffer(outQueueA, BUFFER_NUM, alignedTileRows * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (tileRows <= 0) return;
        CopyXToUB();
        CopyYToUB();
        ComputeAllColumns();
    }

private:
    __aicore__ inline void CopyXToUB()
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        if (incx == 1) {
            if (isAligned) {
                DataCopy(xLocal, xGlobal[rowStart], tileRows);
            } else {
                DataCopyExtParams cpX;
                cpX.blockCount = 1;
                cpX.blockLen = tileRows * sizeof(float);
                cpX.srcStride = 0;
                cpX.dstStride = 0;
                DataCopyPadExtParams<float> ppX{true, 0,
                    (uint8_t)(alignedTileRows - tileRows), 0.0f};
                DataCopyPad(xLocal, xGlobal[rowStart], cpX, ppX);
            }
        } else {
            Duplicate(xLocal, 0.0f, alignedTileRows);
            pipe_barrier(PIPE_V);
            for (int32_t i = 0; i < tileRows; i++) {
                float val = xGlobal.GetValue((rowStart + i) * incx);
                xLocal.SetValue(i, val);
            }
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyYToUB()
    {
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        if (incy == 1) {
            if (n % ALIGN_FP32 == 0) {
                DataCopy(yLocal, yGlobal, n);
            } else {
                DataCopyExtParams cpY;
                cpY.blockCount = 1;
                cpY.blockLen = n * sizeof(float);
                cpY.srcStride = 0;
                cpY.dstStride = 0;
                DataCopyPadExtParams<float> ppY{true, 0,
                    (uint8_t)(alignedN - n), 0.0f};
                DataCopyPad(yLocal, yGlobal, cpY, ppY);
            }
        } else {
            Duplicate(yLocal, 0.0f, alignedN);
            pipe_barrier(PIPE_V);
            for (int32_t j = 0; j < n; j++) {
                float val = yGlobal.GetValue(j * incy);
                yLocal.SetValue(j, val);
            }
        }
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyColumnIn(LocalTensor<float>& aIn, int32_t colOffset)
    {
        if (isAligned) {
            DataCopy(aIn, aGlobal[colOffset], tileRows);
        } else {
            DataCopyExtParams cpA;
            cpA.blockCount = 1;
            cpA.blockLen = tileRows * sizeof(float);
            cpA.srcStride = 0;
            cpA.dstStride = 0;
            DataCopyPadExtParams<float> ppA{true, 0,
                (uint8_t)(alignedTileRows - tileRows), 0.0f};
            DataCopyPad(aIn, aGlobal[colOffset], cpA, ppA);
        }
    }

    __aicore__ inline void CopyColumnOut(LocalTensor<float>& aRes, int32_t colOffset)
    {
        if (isAligned) {
            DataCopy(aGlobal[colOffset], aRes, tileRows);
        } else {
            DataCopyExtParams cpOut;
            cpOut.blockCount = 1;
            cpOut.blockLen = tileRows * sizeof(float);
            cpOut.srcStride = 0;
            cpOut.dstStride = 0;
            DataCopyPad(aGlobal[colOffset], aRes, cpOut);
        }
    }

    __aicore__ inline void ComputeAllColumns()
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();

        if (alpha != 1.0f) {
            Muls(xLocal, xLocal, alpha, alignedTileRows);
            pipe_barrier(PIPE_V);
        }

        for (int32_t j = 0; j < n; j++) {
            float yVal = yLocal.GetValue(j);

            LocalTensor<float> aIn = inQueueA.AllocTensor<float>();
            int32_t colOffset = j * lda + rowStart;
            CopyColumnIn(aIn, colOffset);
            inQueueA.EnQue(aIn);

            LocalTensor<float> aWork = inQueueA.DeQue<float>();
            LocalTensor<float> aOut = outQueueA.AllocTensor<float>();

            Muls(aOut, xLocal, yVal, alignedTileRows);
            pipe_barrier(PIPE_V);
            Add(aOut, aOut, aWork, alignedTileRows);

            outQueueA.EnQue(aOut);
            inQueueA.FreeTensor(aWork);

            LocalTensor<float> aRes = outQueueA.DeQue<float>();
            CopyColumnOut(aRes, colOffset);
            outQueueA.FreeTensor(aRes);
        }

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueY;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueA;

    GlobalTensor<float> xGlobal;
    GlobalTensor<float> yGlobal;
    GlobalTensor<float> aGlobal;

    float alpha;
    int32_t m, n, lda;
    int32_t incx, incy;
    int32_t rowStart, rowEnd, tileRows, alignedTileRows, alignedN;
    bool isAligned;
};

extern "C" __global__ __aicore__ void sger_kernel_do(
    GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm,
    GM_ADDR tilingBuf)
{
    __gm__ SgerTilingData* tilingPtr =
        reinterpret_cast<__gm__ SgerTilingData*>(tilingBuf);
    SgerTilingData td;
    auto* src = reinterpret_cast<__gm__ uint8_t*>(tilingPtr);
    auto* dst = reinterpret_cast<uint8_t*>(&td);
    for (uint32_t i = 0; i < sizeof(SgerTilingData); i++) {
        dst[i] = src[i];
    }

    KernelSger op;
    op.Init(xGm, yGm, aGm, td.alpha, td.m, td.n, td.lda, td.incx, td.incy);
    op.Process();
}
