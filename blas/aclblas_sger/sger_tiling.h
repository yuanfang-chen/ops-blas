#ifndef SGER_TILING_H
#define SGER_TILING_H

#include <stdint.h>

struct SgerTilingData {
    float alpha;
    int32_t m;
    int32_t n;
    int32_t lda;
    int32_t incx;
    int32_t incy;
    int32_t blockDim;
};

#endif
