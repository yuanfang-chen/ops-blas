/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include "cann_ops_blasLt.h"

 #include <acl/acl.h>
 #include <algorithm>
 #include <cmath>
 #include <cstdlib>
 #include <cstring>
 #include <list>
 #include <mutex>
 #include <new>
 #include <unordered_map>
 #include <vector>
 
 #define GM_ADDR uint8_t*
 
 namespace {
 
 constexpr int ACLBLASLT_VERSION_MAJOR = 1;
 constexpr int ACLBLASLT_VERSION_MINOR = 0;
 constexpr int ACLBLASLT_VERSION_PATCH = 0;
 
 constexpr uint32_t ACLBLASLT_HANDLE_MAGIC = 0xACBA1234;
 constexpr uint32_t ACLBLASLT_LAYOUT_MAGIC = 0xACBB1234;
 constexpr uint32_t ACLBLASLT_DESC_MAGIC = 0xACBC1234;
 constexpr uint32_t ACLBLASLT_ALGO_MAGIC = 0xACBD1234;
 
 constexpr size_t DEFAULT_WORKSPACE_SIZE = 32 * 1024 * 1024;
 constexpr size_t DEFAULT_L1_BUFFER_SIZE = 512 * 1024;
 constexpr size_t DEFAULT_L0_CUBE_SIZE = 256;
 constexpr uint32_t DEFAULT_AI_CORES = 8;
 constexpr double DEFAULT_PEAK_TFLOPS = 140.0;
 constexpr double DEFAULT_PEAK_GBPS = 900.0;
 
 enum DispatchPolicyType : uint8_t {
   DISPATCH_POLICY_MMAD_SYNC = 0,
   DISPATCH_POLICY_MMAD_PINGPONG = 1,
   DISPATCH_POLICY_MMAD_MULTI_STAGE = 2,
 };
 
 struct AlgoKey {
   uint64_t m = 0;
   uint64_t n = 0;
   uint64_t k = 0;
   aclDataType aType = ACL_FLOAT;
   aclDataType bType = ACL_DT_UNDEFINED;
   aclDataType cType = ACL_DT_UNDEFINED;
   aclDataType dType = ACL_DT_UNDEFINED;
   aclblasComputeType_t computeType = ACLBLAS_COMPUTE_32F;
   aclblasLtEpilogue_t epilogue = ACLBLASLT_EPILOGUE_DEFAULT;
   bool transA = false;
   bool transB = false;
 
   bool operator==(const AlgoKey& other) const
   {
     return m == other.m && n == other.n && k == other.k && aType == other.aType && bType == other.bType &&
            cType == other.cType && dType == other.dType && computeType == other.computeType && epilogue == other.epilogue &&
            transA == other.transA && transB == other.transB;
   }
 };
 
 struct AlgoKeyHasher {
   size_t operator()(const AlgoKey& x) const
   {
     size_t h = 1469598103934665603ull;
     auto mix = [&](uint64_t v) {
       h ^= static_cast<size_t>(v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
     };
     mix(x.m);
     mix(x.n);
     mix(x.k);
     mix(static_cast<uint64_t>(x.aType));
     mix(static_cast<uint64_t>(x.bType));
     mix(static_cast<uint64_t>(x.cType));
     mix(static_cast<uint64_t>(x.dType));
     mix(static_cast<uint64_t>(x.computeType));
     mix(static_cast<uint64_t>(x.epilogue));
     mix(static_cast<uint64_t>(x.transA));
     mix(static_cast<uint64_t>(x.transB));
     return h;
   }
 };
 
 struct CacheEntry {
   aclblasLtMatmulAlgo_t algo;
   std::list<AlgoKey>::iterator lruIter;
 };
 
 struct aclblasLtHandle {
   uint32_t magic = ACLBLASLT_HANDLE_MAGIC;
   bool initialized = false;
   // version info
   int versionMajor = ACLBLASLT_VERSION_MAJOR;
   int versionMinor = ACLBLASLT_VERSION_MINOR;
   // AscendCL runtime
   aclrtContext context = nullptr;
   aclrtStream defaultStream = nullptr;
   int32_t deviceId = 0;
   // workspace
   void* internalWorkspace = nullptr;
   size_t workspaceSize = 0;
   // thread safety
   std::mutex* mutex = nullptr;
   // soc spec
   int npuArch = 0;
   size_t maxSharedMemory = 0;
   // algo cache
   std::unordered_map<AlgoKey, CacheEntry, AlgoKeyHasher>* algoCache = nullptr;
   size_t algoCacheMaxSize = 128;
   std::list<AlgoKey>* lruList = nullptr;
 };
 
 struct aclblasLtMatrixLayoutImpl {
   uint32_t magic;
   aclDataType type;
   uint64_t rows;
   uint64_t cols;
   int64_t ld;
   aclblasLtOrder_t order = ACLBLASLT_ORDER_COL;
   int32_t batchCount = 1;
   int64_t stridedBatchOffset = 0;
 };
 static_assert(sizeof(aclblasLtMatrixLayoutImpl) <= sizeof(aclblasLtMatrixLayoutOpaque_t),
               "Impl of aclblasLtMatrixLayout must fit in capsule!");
 
 struct aclblasLtMatmulDescImpl {
   uint32_t magic;
   aclblasComputeType_t computeType;
   aclDataType scaleType;
   aclblasOperation_t transA = ACLBLAS_OP_N;
   aclblasOperation_t transB = ACLBLAS_OP_N;
   aclblasLtEpilogue_t epilogue = ACLBLASLT_EPILOGUE_DEFAULT;
   const void* bias = nullptr;
   aclDataType biasDataType = ACL_DT_UNDEFINED;
 };
 static_assert(sizeof(aclblasLtMatmulDescImpl) <= sizeof(aclblasLtMatmulDescOpaque_t),
               "Impl of aclblasLtMatmulDesc must fit in capsule!");
 
 struct aclblasLtMatmulPreferenceImpl {
   uint32_t magic;
   uint32_t searchMode = 0;
   size_t maxWorkspaceBytes = DEFAULT_WORKSPACE_SIZE;
   int32_t maxResults = 3;
   bool allowMixedPrecision = true;
   bool allowSplitK = true;
   // tiling
   uint32_t preferredL0M = 0;
   uint32_t preferredL0N = 0;
   uint32_t preferredL0K = 0;
   // Scheduling
   bool preferPingpong = false;
   bool preferDoubleBuffer = false;
   float minEfficiency = 0.5f;
 };
 static_assert(sizeof(aclblasLtMatmulPreferenceImpl) <= sizeof(aclblasLtMatmulPreferenceOpaque_t),
               "Impl of aclblasLtMatmulPreference must fit in capsule!");
 
 struct AscendHardwareCaps {
   uint32_t numAICores = DEFAULT_AI_CORES;
   uint32_t l0CubeSize = DEFAULT_L0_CUBE_SIZE;
   size_t l1BufferSize = DEFAULT_L1_BUFFER_SIZE;
   double memoryBandwidthGBps = DEFAULT_PEAK_GBPS;
   double peakTFlops = DEFAULT_PEAK_TFLOPS;
   double bandwidthBoundThreshold = 32.0;
 };
 
 struct AlgoCandidate {
   uint32_t algoId = 0;
   uint32_t l1TileM = 128;
   uint32_t l1TileN = 128;
   uint32_t l1TileK = 128;
   uint32_t l0TileM = 64;
   uint32_t l0TileN = 64;
   uint32_t l0TileK = 64;
   DispatchPolicyType policy = DISPATCH_POLICY_MMAD_SYNC;
   uint32_t numBuffers = 1;
   uint32_t splitKFactor = 1;
   size_t workspaceSize = 0;
   double peakPerformance = DEFAULT_PEAK_TFLOPS;
 };
 
 
 struct ScoredResult {
   AlgoCandidate cand;
   double estimatedTimeMs = 0.0;
   double totalScore = 0.0;
   bool isEfficient = true;
 };
 
 struct PackedAlgo {
   uint32_t magic;
   uint32_t algoId;
   uint16_t l1mDiv16;
   uint16_t l1nDiv16;
   uint8_t policy;
   uint8_t numBuffers;
   uint8_t splitK;
   uint8_t flags;
 };
 static_assert(sizeof(PackedAlgo) == 16, "PackedAlgo must fit algo.data");
 
 template <typename T>
 static aclblasStatus_t AllocHandle(T** out)
 {
   if (out == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
   *out = nullptr;
   T* p = new (std::nothrow) T();
   if (p == nullptr) {
     return ACLBLAS_STATUS_ALLOC_FAILED;
   }
   *out = p;
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 template <typename T>
 static aclblasStatus_t FreeHandle(T* p)
 {
   if (p == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
   delete p;
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 static size_t GetTypeSize(aclDataType dt)
 {
   switch (dt) {
     case ACL_FLOAT16:
       return 2;
     case ACL_FLOAT:
     case ACL_INT32:
       return 4;
     case ACL_INT8:
       return 1;
     default:
       return 0;
   }
 }
 
 static bool IsDataTypeSupported(aclDataType dt)
 {
   return GetTypeSize(dt) != 0;
 }
 
 static bool CheckComputeTypeCompatibility(aclblasComputeType_t ct, aclDataType typeA, aclDataType typeB)
 {
   if (typeA != typeB) {
     return false;
   }
   if (typeA == ACL_FLOAT16) {
     return ct == ACLBLAS_COMPUTE_16F || ct == ACLBLAS_COMPUTE_16F_PEDANTIC || ct == ACLBLAS_COMPUTE_32F ||
            ct == ACLBLAS_COMPUTE_32F_PEDANTIC || ct == ACLBLAS_COMPUTE_32F_FAST_16F;
   }
   if (typeA == ACL_FLOAT) {
     return ct == ACLBLAS_COMPUTE_32F || ct == ACLBLAS_COMPUTE_32F_PEDANTIC || ct == ACLBLAS_COMPUTE_32F_FAST_TF32;
   }
   if (typeA == ACL_INT8) {
     return ct == ACLBLAS_COMPUTE_32I || ct == ACLBLAS_COMPUTE_32I_PEDANTIC;
   }
   return false;
 }
 
 static uint32_t CeilDivU64(uint64_t a, uint64_t b)
 {
   return static_cast<uint32_t>((a + b - 1) / b);
 }
 
 static uint32_t GenerateAlgoId(DispatchPolicyType policy,
                                uint32_t l1m,
                                uint32_t l1n,
                                uint32_t l1k,
                                uint32_t splitKFactor)
 {
   return (static_cast<uint32_t>(policy) << 28) ^ (l1m << 16) ^ (l1n << 8) ^ (l1k << 2) ^ splitKFactor;
 }
 
 static aclblasLtMatmulAlgo_t BuildAlgoFromCandidate(const AlgoCandidate& cand)
 {
   aclblasLtMatmulAlgo_t out{};
   PackedAlgo packed{};
   packed.magic = ACLBLASLT_ALGO_MAGIC;
   packed.algoId = cand.algoId;
   packed.l1mDiv16 = static_cast<uint16_t>(cand.l1TileM / 16);
   packed.l1nDiv16 = static_cast<uint16_t>(cand.l1TileN / 16);
   packed.policy = static_cast<uint8_t>(cand.policy);
   packed.numBuffers = static_cast<uint8_t>(cand.numBuffers);
   packed.splitK = static_cast<uint8_t>(cand.splitKFactor);
   std::memcpy(out.data, &packed, sizeof(packed));
   out.max_workspace_bytes = cand.workspaceSize;
   return out;
 }
 
 static bool DecodeAlgo(const aclblasLtMatmulAlgo_t& algo, PackedAlgo* packed)
 {
   if (packed == nullptr) {
     return false;
   }
   std::memcpy(packed, algo.data, sizeof(PackedAlgo));
   return packed->magic == ACLBLASLT_ALGO_MAGIC;
 }
 
 static void GetAscendHardwareCaps(int32_t, AscendHardwareCaps* caps)
 {
   if (caps == nullptr) {
     return;
   }
   // 当前仓库保持轻量默认能力值，后续可对接真实设备查询。
   caps->numAICores = DEFAULT_AI_CORES;
   caps->l0CubeSize = DEFAULT_L0_CUBE_SIZE;
   caps->l1BufferSize = DEFAULT_L1_BUFFER_SIZE;
   caps->memoryBandwidthGBps = DEFAULT_PEAK_GBPS;
   caps->peakTFlops = DEFAULT_PEAK_TFLOPS;
   caps->bandwidthBoundThreshold = 32.0;
 }
 
 static void SelectL1TileShape(uint64_t m,
                               uint64_t n,
                               uint64_t,
                               uint32_t numAICores,
                               uint32_t prefL0M,
                               uint32_t prefL0N,
                               uint32_t prefL0K,
                               uint32_t* l1M,
                               uint32_t* l1N,
                               uint32_t* l1K)
 {
   const uint32_t candidates[][3] = {{128, 256, 256}, {256, 128, 256}, {256, 256, 128}, {128, 128, 128}, {256, 256, 64}};
 
   float bestScore = -1.0f;
   const uint32_t* best = candidates[0];
 
   for (const auto& cand : candidates) {
     uint32_t cm = cand[0];
     uint32_t cn = cand[1];
     uint32_t ck = cand[2];
 
     uint32_t tilesM = CeilDivU64(m, cm);
     uint32_t tilesN = CeilDivU64(n, cn);
     uint32_t totalTiles = tilesM * tilesN;
 
     float balanceScore = 1.0f - static_cast<float>(totalTiles % std::max(1u, numAICores)) / std::max(1u, numAICores);
     size_t l1Usage = static_cast<size_t>(cm) * ck * sizeof(float) + static_cast<size_t>(cn) * ck * sizeof(float);
     float l1Util = static_cast<float>(l1Usage) / static_cast<float>(DEFAULT_L1_BUFFER_SIZE);
 
     float l0Match = 0.0f;
     if (prefL0M > 0 && cm % prefL0M == 0) {
       l0Match += 0.3f;
     }
     if (prefL0N > 0 && cn % prefL0N == 0) {
       l0Match += 0.3f;
     }
     if (prefL0K > 0 && ck % prefL0K == 0) {
       l0Match += 0.4f;
     }
 
     float score = balanceScore * 0.4f + std::min(l1Util, 1.0f) * 0.3f + l0Match * 0.3f;
     if (score > bestScore) {
       bestScore = score;
       best = cand;
     }
   }
 
   *l1M = best[0];
   *l1N = best[1];
   *l1K = best[2];
 }
 
 static void SelectL0TileShape(uint32_t l1M,
                               uint32_t l1N,
                               uint32_t l1K,
                               size_t,
                               size_t,
                               aclDataType,
                               aclDataType,
                               uint32_t* l0M,
                               uint32_t* l0N,
                               uint32_t* l0K)
 {
   *l0K = std::min(64u, l1K);
   *l0M = std::min(128u, l1M);
   *l0N = std::min(256u, l1N);
 
   while (*l0M > 16 && (l1M % *l0M != 0)) {
     --(*l0M);
   }
   while (*l0N > 16 && (l1N % *l0N != 0)) {
     --(*l0N);
   }
 }
 
 static uint32_t SelectSplitKForAscend(uint32_t l1LoopsK, uint32_t numAICores)
 {
   if (numAICores == 0) {
     return 1;
   }
   uint32_t candidate = std::min(l1LoopsK, numAICores);
   return std::max(1u, candidate);
 }
 
 static size_t CalculateWorkspaceForAscend(uint64_t m,
                                           uint64_t n,
                                           uint32_t splitKFactor,
                                           aclblasLtEpilogue_t epilogue)
 {
   size_t workspace = 0;
   if (splitKFactor > 1) {
     workspace += static_cast<size_t>(splitKFactor) * static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);
   }
 
   switch (epilogue) {
     case ACLBLASLT_EPILOGUE_BIAS:
     case ACLBLASLT_EPILOGUE_RELU_BIAS:
     case ACLBLASLT_EPILOGUE_GELU_BIAS:
       workspace += static_cast<size_t>(m) * sizeof(float);
       break;
     case ACLBLASLT_EPILOGUE_GELU:
     case ACLBLASLT_EPILOGUE_RELU:
       workspace += 64 * 1024;
       break;
     default:
       break;
   }
   return workspace;
 }
 
 static bool CheckHandleValid(const aclblasLtHandle* h)
 {
   return h != nullptr && h->magic == ACLBLASLT_HANDLE_MAGIC && h->initialized && h->algoCache != nullptr && h->lruList != nullptr;
 }
 
 static bool BuildGemmShape(const aclblasLtMatmulDescImpl* desc,
                            const aclblasLtMatrixLayoutImpl* A,
                            const aclblasLtMatrixLayoutImpl* B,
                            const aclblasLtMatrixLayoutImpl* D,
                            uint64_t* m,
                            uint64_t* n,
                            uint64_t* k)
 {
   if (desc == nullptr || A == nullptr || B == nullptr || D == nullptr || m == nullptr || n == nullptr || k == nullptr) {
     return false;
   }
   const bool transA = (desc->transA != ACLBLAS_OP_N);
   const bool transB = (desc->transB != ACLBLAS_OP_N);
 
   const uint64_t mA = transA ? A->cols : A->rows;
   const uint64_t kA = transA ? A->rows : A->cols;
   const uint64_t kB = transB ? B->cols : B->rows;
   const uint64_t nB = transB ? B->rows : B->cols;
 
   if (mA != D->rows || kA != kB || nB != D->cols) {
     return false;
   }
 
   *m = mA;
   *n = nB;
   *k = kA;
   return true;
 }
 
 
 } // namespace
 
 extern void matmul_kernel_do(GM_ADDR a,
   GM_ADDR b,
   GM_ADDR c,
   GM_ADDR d,
   uint32_t m,
   uint32_t k,
   uint32_t n,
   uint32_t numBlocks,
   void *stream);
 
 extern "C" {
 
 aclblasStatus_t aclblasLtGetVersion(size_t* version)
 {
   if (version == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   *version = (static_cast<size_t>(ACLBLASLT_VERSION_MAJOR) << 24) |
              (static_cast<size_t>(ACLBLASLT_VERSION_MINOR) << 16) |
              static_cast<size_t>(ACLBLASLT_VERSION_PATCH);
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtGetProperty(aclblasLtPropertyType_t type, int* value)
 {
   if (value == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   switch (type) {
     case ACLBLASLT_PROPERTY_MAJOR_VERSION:
       *value = ACLBLASLT_VERSION_MAJOR;
       return ACLBLAS_STATUS_SUCCESS;
     case ACLBLASLT_PROPERTY_MINOR_VERSION:
       *value = ACLBLASLT_VERSION_MINOR;
       return ACLBLAS_STATUS_SUCCESS;
     case ACLBLASLT_PROPERTY_PATCH_LEVEL:
       *value = ACLBLASLT_VERSION_PATCH;
       return ACLBLAS_STATUS_SUCCESS;
     default:
       return ACLBLAS_STATUS_INVALID_VALUE;
   }
 }
 
 aclblasStatus_t aclblasLtCreate(aclblasLtHandle_t* lightHandle)
 {
   if (lightHandle == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   aclblasLtHandle* h = nullptr;
   auto st = AllocHandle(&h);
   if (st != ACLBLAS_STATUS_SUCCESS) {
     return st;
   }
 
   int32_t deviceId = 0;
   aclError aclRet = aclrtGetDevice(&deviceId);
   if (aclRet != ACL_SUCCESS) {
     delete h;
     return ACLBLAS_STATUS_NOT_INITIALIZED;
   }
 
   aclrtContext currentCtx = nullptr;
   aclRet = aclrtGetCurrentContext(&currentCtx);
   if (aclRet != ACL_SUCCESS || currentCtx == nullptr) {
     delete h;
     return ACLBLAS_STATUS_NOT_INITIALIZED;
   }
 
   h->deviceId = deviceId;
   h->context = currentCtx;
   h->defaultStream = nullptr;
   h->workspaceSize = DEFAULT_WORKSPACE_SIZE;
   h->internalWorkspace = std::malloc(h->workspaceSize);
   if (h->internalWorkspace == nullptr) {
     delete h;
     return ACLBLAS_STATUS_ALLOC_FAILED;
   }
 
   h->mutex = new (std::nothrow) std::mutex();
   h->algoCache = new (std::nothrow) std::unordered_map<AlgoKey, CacheEntry, AlgoKeyHasher>();
   h->lruList = new (std::nothrow) std::list<AlgoKey>();
   if (h->mutex == nullptr || h->algoCache == nullptr || h->lruList == nullptr) {
     delete h->mutex;
     delete h->algoCache;
     delete h->lruList;
     std::free(h->internalWorkspace);
     delete h;
     return ACLBLAS_STATUS_ALLOC_FAILED;
   }
 
   h->npuArch = 2;
   h->maxSharedMemory = DEFAULT_L1_BUFFER_SIZE;
   h->initialized = true;
   *lightHandle = reinterpret_cast<aclblasLtHandle_t>(h);
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtDestroy(const aclblasLtHandle_t lightHandle)
 {
   if (lightHandle == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   auto* h = reinterpret_cast<aclblasLtHandle*>(lightHandle);
   if (h->magic != ACLBLASLT_HANDLE_MAGIC) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   h->initialized = false;
   delete h->mutex;
   delete h->algoCache;
   delete h->lruList;
   std::free(h->internalWorkspace);
   h->mutex = nullptr;
   h->algoCache = nullptr;
   h->lruList = nullptr;
   h->internalWorkspace = nullptr;
   h->workspaceSize = 0;
   return FreeHandle(h);
 }
 
 aclblasStatus_t aclblasLtMatrixLayoutCreate(aclblasLtMatrixLayout_t* layout,
                                             aclDataType type,
                                             uint64_t rows,
                                             uint64_t cols,
                                             int64_t ld)
 {
   // 1. 参数校验
   if (layout == nullptr || rows == 0 || cols == 0 || ld < 0) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
   *layout = nullptr;
   // 2. 堆上分配胶囊
   auto* capsule = new (std::nothrow) aclblasLtMatrixLayoutOpaque_t();
   if (capsule == nullptr) {
       return ACLBLAS_STATUS_ALLOC_FAILED;
   }
   // 3. 栈上创建Impl，初始化后拷贝进胶囊
   aclblasLtMatrixLayoutImpl impl;
   impl.magic = ACLBLASLT_LAYOUT_MAGIC;
   impl.type = type;
   impl.rows = rows;
   impl.cols = cols;
   impl.ld = (ld == 0) ? static_cast<int64_t>(rows) : ld;
   // 4. Impl → 胶囊
   static_assert(sizeof(impl) <= sizeof(*capsule), "aclblasLtMatrixLayoutImpl too large, not fit in capsule!");
   memcpy(capsule, &impl, sizeof(impl));
   if (sizeof(*capsule) > sizeof(impl)) {
       memset(reinterpret_cast<char*>(capsule) + sizeof(impl),
               0,
               sizeof(*capsule) - sizeof(impl));
   }
   // 5. 返回胶囊指针
   *layout = capsule;
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtMatrixLayoutDestroy(const aclblasLtMatrixLayout_t layout)
 {
   if (layout == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   auto* capsule = reinterpret_cast<aclblasLtMatrixLayoutOpaque_t*>(layout);
   delete capsule;
 
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtMatrixLayoutSetAttribute(aclblasLtMatrixLayout_t layout,
                                                   aclblasLtMatrixLayoutAttribute_t attr,
                                                   const void* buf,
                                                   size_t sizeInBytes)
 {
   if (layout == nullptr || buf == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   // 解包到栈上
   aclblasLtMatrixLayoutImpl impl;
   memcpy(&impl, layout, sizeof(impl));
 
   switch (attr) {
     case ACLBLASLT_MATRIX_LAYOUT_TYPE:
       if (sizeInBytes != sizeof(impl.type)) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       impl.type = *reinterpret_cast<const aclDataType*>(buf);
       break;
 
     case ACLBLASLT_MATRIX_LAYOUT_ROWS:
       if (sizeInBytes != sizeof(impl.rows)) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       impl.rows = *reinterpret_cast<const uint64_t*>(buf);
       break;
 
     case ACLBLASLT_MATRIX_LAYOUT_COLS:
       if (sizeInBytes != sizeof(impl.cols)) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       impl.cols = *reinterpret_cast<const uint64_t*>(buf);
       break;
 
     case ACLBLASLT_MATRIX_LAYOUT_LD:
       if (sizeInBytes != sizeof(impl.ld)) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       impl.ld = *reinterpret_cast<const int64_t*>(buf);
       break;
 
     case ACLBLASLT_MATRIX_LAYOUT_ORDER:
       if (sizeInBytes != sizeof(impl.order)) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       impl.order = *reinterpret_cast<const aclblasLtOrder_t*>(buf);
       break;
 
     case ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
       if (sizeInBytes != sizeof(impl.batchCount)) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       impl.batchCount = *reinterpret_cast<const int32_t*>(buf);
       break;
 
     case ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
       if (sizeInBytes != sizeof(impl.stridedBatchOffset)) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       impl.stridedBatchOffset = *reinterpret_cast<const int64_t*>(buf);
       break;
 
     default:
         return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   // 压缩回堆上
   memcpy(layout, &impl, sizeof(impl));
   if (sizeof(*layout) > sizeof(impl)) {
     memset(reinterpret_cast<char*>(layout) + sizeof(impl),
             0,
             sizeof(*layout) - sizeof(impl));
   }
 
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtMatrixLayoutGetAttribute(const aclblasLtMatrixLayout_t layout,
                                                    aclblasLtMatrixLayoutAttribute_t attr,
                                                    void* buf,
                                                    size_t sizeInBytes,
                                                    size_t* sizeWritten)
 {
   if (layout == nullptr || buf == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   aclblasLtMatrixLayoutImpl impl;
   static_assert(sizeof(impl) <= sizeof(*layout), "aclblasLtMatrixLayoutImpl too large for capsule");
   memcpy(&impl, layout, sizeof(impl));
 
   size_t actualSize = 0;
 
   switch (attr) {
     case ACLBLASLT_MATRIX_LAYOUT_TYPE:
         actualSize = sizeof(impl.type);
         if (sizeInBytes < actualSize) {
             return ACLBLAS_STATUS_INVALID_VALUE;
         }
         *reinterpret_cast<aclDataType*>(buf) = impl.type;
         break;
 
     case ACLBLASLT_MATRIX_LAYOUT_ROWS:
         actualSize = sizeof(impl.rows);
         if (sizeInBytes < actualSize) {
             return ACLBLAS_STATUS_INVALID_VALUE;
         }
         *reinterpret_cast<uint64_t*>(buf) = impl.rows;
         break;
 
     case ACLBLASLT_MATRIX_LAYOUT_COLS:
         actualSize = sizeof(impl.cols);
         if (sizeInBytes < actualSize) {
             return ACLBLAS_STATUS_INVALID_VALUE;
         }
         *reinterpret_cast<uint64_t*>(buf) = impl.cols;
         break;
 
     case ACLBLASLT_MATRIX_LAYOUT_LD:
         actualSize = sizeof(impl.ld);
         if (sizeInBytes < actualSize) {
             return ACLBLAS_STATUS_INVALID_VALUE;
         }
         *reinterpret_cast<int64_t*>(buf) = impl.ld;
         break;
 
     case ACLBLASLT_MATRIX_LAYOUT_ORDER:
         actualSize = sizeof(impl.order);
         if (sizeInBytes < actualSize) {
             return ACLBLAS_STATUS_INVALID_VALUE;
         }
         *reinterpret_cast<aclblasLtOrder_t*>(buf) = impl.order;
         break;
 
     case ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
         actualSize = sizeof(impl.batchCount);
         if (sizeInBytes < actualSize) {
             return ACLBLAS_STATUS_INVALID_VALUE;
         }
         *reinterpret_cast<int32_t*>(buf) = impl.batchCount;
         break;
 
     case ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
         actualSize = sizeof(impl.stridedBatchOffset);
         if (sizeInBytes < actualSize) {
             return ACLBLAS_STATUS_INVALID_VALUE;
         }
         *reinterpret_cast<int64_t*>(buf) = impl.stridedBatchOffset;
         break;
 
     default:
         return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   if (sizeWritten != nullptr) {
       *sizeWritten = actualSize;
   }
 
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtMatmulDescCreate(aclblasLtMatmulDesc_t* desc,
                                           aclblasComputeType_t computeType,
                                           aclDataType scaleType)
 {
   if (desc == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
   *desc = nullptr;
 
   auto* capsule = new (std::nothrow) aclblasLtMatmulDescOpaque_t();
   if (capsule == nullptr) {
     return ACLBLAS_STATUS_ALLOC_FAILED;
   }
 
   aclblasLtMatmulDescImpl impl;
   impl.magic = ACLBLASLT_DESC_MAGIC;
   impl.computeType = computeType;
   impl.scaleType = scaleType;
 
   static_assert(sizeof(impl) <= sizeof(*capsule), "aclblasLtMatmulDescImpl too large, not fit in capsule!");
   std::memcpy(capsule, &impl, sizeof(impl));
   if (sizeof(*capsule) > sizeof(impl)) {
     std::memset(reinterpret_cast<char*>(capsule) + sizeof(impl), 0, sizeof(*capsule) - sizeof(impl));
   }
 
   *desc = capsule;
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtMatmulDescDestroy(const aclblasLtMatmulDesc_t desc)
 {
   if (desc == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   auto* capsule = reinterpret_cast<aclblasLtMatmulDescOpaque_t*>(desc);
   delete capsule;
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtMatmulDescSetAttribute(aclblasLtMatmulDesc_t desc,
                                                 aclblasLtMatmulDescAttribute_t attr,
                                                 const void* buf,
                                                 size_t sizeInBytes)
 {
   if (desc == nullptr || buf == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   aclblasLtMatmulDescImpl impl;
   std::memcpy(&impl, desc, sizeof(impl));
 
   switch (attr) {
     case ACLBLASLT_MATMUL_DESC_EPILOGUE: {
       if (sizeInBytes != sizeof(aclblasLtEpilogue_t) && sizeInBytes != sizeof(uint32_t)) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       uint32_t v = 0;
       std::memcpy(&v, buf, sizeof(uint32_t));
       impl.epilogue = static_cast<aclblasLtEpilogue_t>(v);
       break;
     }
     case ACLBLASLT_MATMUL_DESC_BIAS_POINTER:
       if (sizeInBytes != sizeof(void*)) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       std::memcpy(&impl.bias, buf, sizeof(void*));
       break;
     case ACLBLASLT_MATMUL_DESC_TRANSA: {
       if (sizeInBytes != sizeof(int32_t)) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       int32_t v = 0;
       std::memcpy(&v, buf, sizeof(int32_t));
       impl.transA = static_cast<aclblasOperation_t>(v);
       break;
     }
     case ACLBLASLT_MATMUL_DESC_TRANSB: {
       if (sizeInBytes != sizeof(int32_t)) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       int32_t v = 0;
       std::memcpy(&v, buf, sizeof(int32_t));
       impl.transB = static_cast<aclblasOperation_t>(v);
       break;
     }
     case ACLBLASLT_MATMUL_DESC_BIAS_DATA_TYPE: {
       if (sizeInBytes != sizeof(int32_t)) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       int32_t v = 0;
       std::memcpy(&v, buf, sizeof(int32_t));
       impl.biasDataType = static_cast<aclDataType>(v);
       break;
     }
     default:
       return ACLBLAS_STATUS_NOT_SUPPORTED;
   }
 
   std::memcpy(desc, &impl, sizeof(impl));
   if (sizeof(*desc) > sizeof(impl)) {
     std::memset(reinterpret_cast<char*>(desc) + sizeof(impl), 0, sizeof(*desc) - sizeof(impl));
   }
 
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtMatmulDescGetAttribute(aclblasLtMatmulDesc_t desc,
                                                 aclblasLtMatmulDescAttribute_t attr,
                                                 void* buf,
                                                 size_t sizeInBytes,
                                                 size_t* sizeWritten)
 {
     if (desc == nullptr || buf == nullptr) {
         return ACLBLAS_STATUS_INVALID_VALUE;
     }
 
     aclblasLtMatmulDescImpl impl;
     std::memcpy(&impl, desc, sizeof(impl));
 
     // if (!impl.valid()) {
     //     return ACLBLAS_STATUS_INVALID_VALUE;
     // }
 
     size_t requiredSize = 0;
     const void* srcPtr = nullptr;
 
     switch (attr) {
       case ACLBLASLT_MATMUL_DESC_EPILOGUE:
         requiredSize = sizeof(impl.epilogue);
         srcPtr = &impl.epilogue;
         break;
 
       case ACLBLASLT_MATMUL_DESC_BIAS_POINTER:
         requiredSize = sizeof(impl.bias);
         srcPtr = &impl.bias;
         break;
 
       case ACLBLASLT_MATMUL_DESC_TRANSA:
         requiredSize = sizeof(impl.transA);
         srcPtr = &impl.transA;
         break;
 
       case ACLBLASLT_MATMUL_DESC_TRANSB:
         requiredSize = sizeof(impl.transB);
         srcPtr = &impl.transB;
         break;
 
       case ACLBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
         requiredSize = sizeof(impl.biasDataType);
         srcPtr = &impl.biasDataType;
         break;
 
       default:
         return ACLBLAS_STATUS_NOT_SUPPORTED;
     }
 
     // 检查用户缓冲区大小
     if (sizeInBytes < requiredSize) {
         if (sizeWritten != nullptr) {
             *sizeWritten = requiredSize;
         }
         return ACLBLAS_STATUS_INVALID_VALUE;
     }
 
     std::memcpy(buf, srcPtr, requiredSize);
 
     if (sizeWritten != nullptr) {
         *sizeWritten = requiredSize;
     }
 
     return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtMatmulPreferenceCreate(aclblasLtMatmulPreference_t* pref)
 {
   if (pref == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   *pref = nullptr;
   auto* capsule = new (std::nothrow) aclblasLtMatmulPreferenceOpaque_t();
   if (capsule == nullptr) {
     return ACLBLAS_STATUS_ALLOC_FAILED;
   }
   std::memset(capsule, 0, sizeof(*capsule));
 
   aclblasLtMatmulPreferenceImpl impl;
   impl.magic = ACLBLASLT_PREF_MAGIC;
   std::memcpy(capsule, &impl, sizeof(impl));
   if (sizeof(*capsule) > sizeof(impl)) {
     std::memset(reinterpret_cast<char*>(capsule) + sizeof(impl), 0, sizeof(*capsule) - sizeof(impl));
   }
 
   *pref = capsule;
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtMatmulPreferenceDestroy(const aclblasLtMatmulPreference_t pref)
 {
   if (pref == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   auto* capsule = reinterpret_cast<aclblasLtMatmulPreferenceOpaque_t*>(pref);
   delete capsule;
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtMatmulPreferenceSetAttribute(aclblasLtMatmulPreference_t pref,
                                                       aclblasLtMatmulPreferenceAttribute_t attr,
                                                       const void* buf,
                                                       size_t sizeInBytes)
 {
   if (pref == nullptr || buf == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   aclblasLtMatmulPreferenceImpl impl;
   std::memcpy(&impl, pref, sizeof(impl));
 
   switch (attr) {
     case ACLBLASLT_MATMUL_PREF_SEARCH_MODE: {
       if (sizeInBytes != sizeof(uint32_t)) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       uint32_t v = 0;
       std::memcpy(&v, buf, sizeof(v));
       // 0=heuristic, 1=exhaustive, 2=fast
       if (v > 2) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       impl.searchMode = v;
       break;
     }
 
     case ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: {
       if (sizeInBytes != sizeof(size_t) && sizeInBytes != sizeof(uint64_t)) {
           return ACLBLAS_STATUS_INVALID_VALUE;
       }
       size_t v = 0;
       std::memcpy(&v, buf, std::min(sizeInBytes, sizeof(v)));
       if (v > INT64_MAX) {
           return ACLBLAS_STATUS_INVALID_VALUE;
       }
       impl.maxWorkspaceBytes = v;
       break;
     }
 
     // case ACLBLASLT_MATMUL_PREF_MAX_RESULTS: {
     //   if (sizeInBytes != sizeof(int32_t)) {
     //       return ACLBLAS_STATUS_INVALID_VALUE;
     //   }
     //   int32_t v = 0;
     //   std::memcpy(&v, buf, sizeof(v));
     //   if (v <= 0 || v > 10) {
     //       return ACLBLAS_STATUS_INVALID_VALUE;
     //   }
     //   impl.maxResults = v;
     //   break;
     // }
 
     // case ACLBLASLT_MATMUL_PREF_ALLOW_MIXED_PRECISION: {
     //   if (sizeInBytes != sizeof(bool) && sizeInBytes != sizeof(int32_t)) {
     //     return ACLBLAS_STATUS_INVALID_VALUE;
     //   }
     //   bool v = false;
     //   std::memcpy(&v, buf, sizeof(bool));
     //   impl.allowMixedPrecision = v;
     //   break;
     // }
 
     // case ACLBLASLT_MATMUL_PREF_ALLOW_SPLIT_K: {
     //   if (sizeInBytes != sizeof(bool) && sizeInBytes != sizeof(int32_t)) {
     //     return ACLBLAS_STATUS_INVALID_VALUE;
     //   }
     //   bool v = false;
     //   std::memcpy(&v, buf, sizeof(bool));
     //   impl.allowSplitK = v;
     //   break;
     // }
 
     // case ACLBLASLT_MATMUL_PREF_L0_TILE_M: {
     //   if (sizeInBytes != sizeof(uint32_t)) {
     //     return ACLBLAS_STATUS_INVALID_VALUE;
     //   }
     //   std::memcpy(&impl.preferredL0M, buf, sizeof(uint32_t));
     //   break;
     // }
 
     // case ACLBLASLT_MATMUL_PREF_L0_TILE_N: {
     //   if (sizeInBytes != sizeof(uint32_t)) {
     //     return ACLBLAS_STATUS_INVALID_VALUE;
     //   }
     //   std::memcpy(&impl.preferredL0N, buf, sizeof(uint32_t));
     //   break;
     // }
 
     // case ACLBLASLT_MATMUL_PREF_L0_TILE_K: {
     //   if (sizeInBytes != sizeof(uint32_t)) {
     //     return ACLBLAS_STATUS_INVALID_VALUE;
     //   }
     //   std::memcpy(&impl.preferredL0K, buf, sizeof(uint32_t));
     //   break;
     // }
 
     // case ACLBLASLT_MATMUL_PREF_PREFER_PINGPONG: {
     //   if (sizeInBytes != sizeof(bool)) {
     //     return ACLBLAS_STATUS_INVALID_VALUE;
     //   }
     //   std::memcpy(&impl.preferPingpong, buf, sizeof(bool));
     //   break;
     // }
 
     // case ACLBLASLT_MATMUL_PREF_PREFER_DOUBLE_BUFFER: {
     //   if (sizeInBytes != sizeof(bool)) {
     //     return ACLBLAS_STATUS_INVALID_VALUE;
     //   }
     //   std::memcpy(&impl.preferDoubleBuffer, buf, sizeof(bool));
     //   break;
     // }
 
     // case ACLBLASLT_MATMUL_PREF_MIN_EFFICIENCY: {
     //   if (sizeInBytes != sizeof(float)) {
     //     return ACLBLAS_STATUS_INVALID_VALUE;
     //   }
     //   float v = 0.0f;
     //   std::memcpy(&v, buf, sizeof(v));
     //   if (v < 0.0f || v > 1.0f) {
     //     return ACLBLAS_STATUS_INVALID_VALUE;
     //   }
     //   impl.minEfficiency = v;
     //   break;
     // }
 
     default:
       return ACLBLAS_STATUS_NOT_SUPPORTED;
   }
 
   std::memcpy(pref, &impl, sizeof(impl));
 
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtMatmulPreferenceGetAttribute(aclblasLtMatmulPreference_t pref,
                                                       aclblasLtMatmulPreferenceAttribute_t attr,
                                                       void* buf,
                                                       size_t sizeInBytes,
                                                       size_t* sizeWritten)
 {
     if (pref == nullptr || buf == nullptr) {
         return ACLBLAS_STATUS_INVALID_VALUE;
     }
 
     aclblasLtMatmulPreferenceImpl impl;
     std::memcpy(&impl, pref, sizeof(impl));
 
     size_t requiredSize = 0;
     const void* srcPtr = nullptr;
 
     switch (attr) {
       case ACLBLASLT_MATMUL_PREF_SEARCH_MODE:
           requiredSize = sizeof(impl.searchMode);
           srcPtr = &impl.searchMode;
           break;
 
       case ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
           requiredSize = sizeof(impl.maxWorkspaceBytes);
           srcPtr = &impl.maxWorkspaceBytes;
           break;
 
       // case ACLBLASLT_MATMUL_PREF_MAX_RESULTS:
       //     requiredSize = sizeof(impl.maxResults);
       //     srcPtr = &impl.maxResults;
       //     break;
 
       // case ACLBLASLT_MATMUL_PREF_ALLOW_MIXED_PRECISION:
       //     requiredSize = sizeof(impl.allowMixedPrecision);
       //     srcPtr = &impl.allowMixedPrecision;
       //     break;
 
       // case ACLBLASLT_MATMUL_PREF_ALLOW_SPLIT_K:
       //     requiredSize = sizeof(impl.allowSplitK);
       //     srcPtr = &impl.allowSplitK;
       //     break;
 
       // case ACLBLASLT_MATMUL_PREF_L0_TILE_M:
       //     requiredSize = sizeof(impl.preferredL0M);
       //     srcPtr = &impl.preferredL0M;
       //     break;
 
       // case ACLBLASLT_MATMUL_PREF_L0_TILE_N:
       //     requiredSize = sizeof(impl.preferredL0N);
       //     srcPtr = &impl.preferredL0N;
       //     break;
 
       // case ACLBLASLT_MATMUL_PREF_L0_TILE_K:
       //     requiredSize = sizeof(impl.preferredL0K);
       //     srcPtr = &impl.preferredL0K;
       //     break;
 
       // case ACLBLASLT_MATMUL_PREF_PREFER_PINGPONG:
       //     requiredSize = sizeof(impl.preferPingpong);
       //     srcPtr = &impl.preferPingpong;
       //     break;
 
       // case ACLBLASLT_MATMUL_PREF_PREFER_DOUBLE_BUFFER:
       //     requiredSize = sizeof(impl.preferDoubleBuffer);
       //     srcPtr = &impl.preferDoubleBuffer;
       //     break;
 
       // case ACLBLASLT_MATMUL_PREF_MIN_EFFICIENCY:
       //     requiredSize = sizeof(impl.minEfficiency);
       //     srcPtr = &impl.minEfficiency;
       //     break;
 
       default:
           return ACLBLAS_STATUS_NOT_SUPPORTED;
     }
 
     if (sizeInBytes < requiredSize) {
         if (sizeWritten != nullptr) {
             *sizeWritten = requiredSize;
         }
         return ACLBLAS_STATUS_INVALID_VALUE;
     }
 
     std::memcpy(buf, srcPtr, requiredSize);
 
     if (sizeWritten != nullptr) {
         *sizeWritten = requiredSize;
     }
 
     return ACLBLAS_STATUS_SUCCESS;
 }
 
 aclblasStatus_t aclblasLtMatmulAlgoGetHeuristic(aclblasLtHandle_t lightHandle,
                                                 aclblasLtMatmulDesc_t computeDesc,
                                                 aclblasLtMatrixLayout_t Adesc,
                                                 aclblasLtMatrixLayout_t Bdesc,
                                                 aclblasLtMatrixLayout_t Cdesc,
                                                 aclblasLtMatrixLayout_t Ddesc,
                                                 aclblasLtMatmulPreference_t preference,
                                                 int requestedAlgoCount,
                                                 aclblasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                                 int* returnAlgoCount)
 {
   if (returnAlgoCount == nullptr || heuristicResultsArray == nullptr || requestedAlgoCount <= 0) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
   *returnAlgoCount = 0;
 
   auto* h = reinterpret_cast<aclblasLtHandle*>(lightHandle);
   if (!CheckHandleValid(h) || computeDesc == nullptr || Adesc == nullptr || Bdesc == nullptr || Ddesc == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   aclrtContext currentCtx = nullptr;
   if (aclrtGetCurrentContext(&currentCtx) != ACL_SUCCESS || currentCtx != h->context) {
     return ACLBLAS_STATUS_NOT_INITIALIZED;
   }
   int32_t currentDevice = -1;
   if (aclrtGetDevice(&currentDevice) != ACL_SUCCESS || currentDevice != h->deviceId) {
     return ACLBLAS_STATUS_ARCH_MISMATCH;
   }
 
   auto* desc = reinterpret_cast<aclblasLtMatmulDescImpl*>(computeDesc);
   auto* A = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Adesc);
   auto* B = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Bdesc);
   auto* D = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Ddesc);
 
   if (desc->magic != ACLBLASLT_DESC_MAGIC) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   if (!IsDataTypeSupported(A->type) || !IsDataTypeSupported(B->type) || !IsDataTypeSupported(D->type) ||
       !CheckComputeTypeCompatibility(desc->computeType, A->type, B->type)) {
     return ACLBLAS_STATUS_NOT_SUPPORTED;
   }
 
   uint64_t m = 0;
   uint64_t n = 0;
   uint64_t k = 0;
   if (!BuildGemmShape(desc, A, B, D, &m, &n, &k)) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   aclDataType typeC = D->type;
   if (Cdesc != nullptr) {
     typeC = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Cdesc)->type;
   }
 
   aclblasLtMatmulPreferenceImpl pref;
   // BuildDefaultPreference(&pref);
   if (preference != nullptr) {
     std::memcpy(&pref, preference, sizeof(pref));
   }
 
   AscendHardwareCaps hwCaps;
   GetAscendHardwareCaps(h->deviceId, &hwCaps);
 
   const uint64_t problemSize = m * n * k;
   const bool isSmallGemm = (problemSize < 1024 * 1024ull);
   const bool isLargeGemm = (problemSize > 256ull * 1024ull * 1024ull);
   const bool isSkinny = (m < 16 || n < 16 || k < 16);
 
   std::vector<AlgoCandidate> candidates;
   candidates.reserve(3);
 
   uint32_t l1TileM = 128;
   uint32_t l1TileN = 128;
   uint32_t l1TileK = 128;
   SelectL1TileShape(m,
                     n,
                     k,
                     hwCaps.numAICores,
                     pref.preferredL0M,
                     pref.preferredL0N,
                     pref.preferredL0K,
                     &l1TileM,
                     &l1TileN,
                     &l1TileK);
 
   uint32_t l0TileM = 64;
   uint32_t l0TileN = 64;
   uint32_t l0TileK = 64;
   SelectL0TileShape(l1TileM,
                     l1TileN,
                     l1TileK,
                     hwCaps.l0CubeSize,
                     hwCaps.l1BufferSize,
                     A->type,
                     B->type,
                     &l0TileM,
                     &l0TileN,
                     &l0TileK);
 
   const uint32_t l1LoopsK = CeilDivU64(k, l1TileK);
 
   std::vector<DispatchPolicyType> policyCandidates = {
     DISPATCH_POLICY_MMAD_SYNC,
     DISPATCH_POLICY_MMAD_PINGPONG,
     DISPATCH_POLICY_MMAD_MULTI_STAGE,
   };
 
   for (auto policy : policyCandidates) {
     AlgoCandidate cand;
     cand.policy = policy;
     cand.l1TileM = l1TileM;
     cand.l1TileN = l1TileN;
     cand.l1TileK = l1TileK;
     cand.l0TileM = l0TileM;
     cand.l0TileN = l0TileN;
     cand.l0TileK = l0TileK;
 
     if (policy == DISPATCH_POLICY_MMAD_PINGPONG || pref.preferPingpong || pref.preferDoubleBuffer) {
       cand.numBuffers = 2;
       cand.policy = DISPATCH_POLICY_MMAD_PINGPONG;
     } else if (policy == DISPATCH_POLICY_MMAD_MULTI_STAGE && k > 256 && l1LoopsK > 2) {
       cand.numBuffers = 3;
       cand.policy = DISPATCH_POLICY_MMAD_MULTI_STAGE;
     } else {
       cand.numBuffers = 1;
       cand.policy = DISPATCH_POLICY_MMAD_SYNC;
     }
 
     if (pref.allowSplitK && l1LoopsK >= hwCaps.numAICores * 2) {
       cand.splitKFactor = SelectSplitKForAscend(l1LoopsK, hwCaps.numAICores);
     } else {
       cand.splitKFactor = 1;
     }
 
     cand.workspaceSize = CalculateWorkspaceForAscend(m, n, cand.splitKFactor, desc->epilogue);
     cand.peakPerformance = hwCaps.peakTFlops;
     cand.algoId = GenerateAlgoId(cand.policy, cand.l1TileM, cand.l1TileN, cand.l1TileK, cand.splitKFactor);
     candidates.push_back(cand);
   }
 
   std::vector<ScoredResult> scored;
   scored.reserve(candidates.size());
 
   for (const auto& cand : candidates) {
     ScoredResult sr;
     sr.cand = cand;
 
     const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
     const double bytesA = static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(GetTypeSize(A->type));
     const double bytesB = static_cast<double>(k) * static_cast<double>(n) * static_cast<double>(GetTypeSize(B->type));
     const double bytesCD = static_cast<double>(m) * static_cast<double>(n) *
                            static_cast<double>(GetTypeSize(typeC) + GetTypeSize(D->type));
     const double totalBytes = std::max(1.0, bytesA + bytesB + bytesCD);
     const double arithmeticIntensity = flops / totalBytes;
 
     const double computeTimeUs = ((flops / 1e12) / std::max(1.0, cand.peakPerformance)) * 1e6;
     const double memoryTimeUs = ((totalBytes / 1e9) / std::max(1.0, hwCaps.memoryBandwidthGBps)) * 1e6;
 
     double estimatedTimeUs = 0.0;
     if (arithmeticIntensity > hwCaps.bandwidthBoundThreshold) {
       estimatedTimeUs = computeTimeUs / 0.85;
     } else {
       estimatedTimeUs = memoryTimeUs / 0.70;
     }
 
     double adjust = 1.0;
     if (cand.splitKFactor > 1) {
       adjust *= 1.05;
     }
     if (isSmallGemm) {
       adjust *= 1.2;
     }
     if (isSkinny) {
       adjust *= 1.15;
     }
     if (!isLargeGemm && cand.policy == DISPATCH_POLICY_MMAD_MULTI_STAGE) {
       adjust *= 1.1;
     }
 
     sr.estimatedTimeMs = (estimatedTimeUs * adjust) / 1000.0;
 
     const double timeScore = 1.0 / (1.0 + sr.estimatedTimeMs);
     const double wsDenom = std::max<size_t>(1, pref.maxWorkspaceBytes);
     const double wsPenalty = static_cast<double>(cand.workspaceSize) / static_cast<double>(wsDenom);
     const double reliabilityBonus = (cand.policy == DISPATCH_POLICY_MMAD_SYNC) ? 0.2 : 0.1;
 
     sr.totalScore = timeScore * 0.6 + (1.0 - std::min(1.0, wsPenalty)) * 0.2 + reliabilityBonus * 0.2;
     sr.isEfficient = (timeScore > pref.minEfficiency);
     scored.push_back(sr);
   }
 
   std::sort(scored.begin(), scored.end(), [](const ScoredResult& a, const ScoredResult& b) {
     return a.totalScore > b.totalScore;
   });
 
   int outCount = 0;
   const int cap = std::min(requestedAlgoCount, static_cast<int>(scored.size()));
   for (int i = 0; i < cap; ++i) {
     if (!scored[i].isEfficient && outCount > 0) {
       continue;
     }
     auto algoOut = BuildAlgoFromCandidate(scored[i].cand);
     heuristicResultsArray[outCount].algo = algoOut;
     heuristicResultsArray[outCount].workspaceSize = scored[i].cand.workspaceSize;
     heuristicResultsArray[outCount].state = ACLBLAS_STATUS_SUCCESS;
     heuristicResultsArray[outCount].wavesCount = static_cast<float>(std::max(0.1, scored[i].totalScore));
     std::memset(heuristicResultsArray[outCount].reserved, 0, sizeof(heuristicResultsArray[outCount].reserved));
     ++outCount;
   }
 
   *returnAlgoCount = outCount;
   return (outCount > 0) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_NOT_SUPPORTED;
 }
 
 aclblasStatus_t aclblasLtMatmul(aclblasLtHandle_t lightHandle,
                                 aclblasLtMatmulDesc_t computeDesc,
                                 const void* alpha,
                                 const void* A,
                                 aclblasLtMatrixLayout_t Adesc,
                                 const void* B,
                                 aclblasLtMatrixLayout_t Bdesc,
                                 const void* beta,
                                 const void* C,
                                 aclblasLtMatrixLayout_t Cdesc,
                                 void* D,
                                 aclblasLtMatrixLayout_t Ddesc,
                                 const aclblasLtMatmulAlgo_t* algo,
                                 void* workspace,
                                 size_t workspaceSizeInBytes,
                                 aclrtStream stream)
 {
   auto* ctx = reinterpret_cast<aclblasLtHandle*>(lightHandle);
   if (!CheckHandleValid(ctx)) {
     return ACLBLAS_STATUS_NOT_INITIALIZED;
   }
 
   aclrtContext currentCtx = nullptr;
   if (aclrtGetCurrentContext(&currentCtx) != ACL_SUCCESS || currentCtx != ctx->context) {
     return ACLBLAS_STATUS_NOT_INITIALIZED;
   }
   int32_t currentDevice = -1;
   if (aclrtGetDevice(&currentDevice) != ACL_SUCCESS || currentDevice != ctx->deviceId) {
     return ACLBLAS_STATUS_ARCH_MISMATCH;
   }
 
   if (computeDesc == nullptr || Adesc == nullptr || Bdesc == nullptr || Ddesc == nullptr || alpha == nullptr || beta == nullptr ||
       A == nullptr || B == nullptr || D == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   auto* desc = reinterpret_cast<aclblasLtMatmulDescImpl*>(computeDesc);
   auto* ALayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Adesc);
   auto* BLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Bdesc);
   auto* DLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Ddesc);
 
   if (desc->magic != ACLBLASLT_DESC_MAGIC) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   if (C == nullptr) {
     if (Cdesc != nullptr) {
       return ACLBLAS_STATUS_INVALID_VALUE;
     }
     if (desc->epilogue == ACLBLASLT_EPILOGUE_BIAS || desc->epilogue == ACLBLASLT_EPILOGUE_RELU_BIAS ||
         desc->epilogue == ACLBLASLT_EPILOGUE_GELU_BIAS) {
       return ACLBLAS_STATUS_INVALID_VALUE;
     }
   } else if (Cdesc == nullptr) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   uint64_t m = 0;
   uint64_t n = 0;
   uint64_t k = 0;
   if (!BuildGemmShape(desc, ALayout, BLayout, DLayout, &m, &n, &k)) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   aclDataType typeC = DLayout->type;
   if (Cdesc != nullptr) {
     typeC = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Cdesc)->type;
   }
 
   if (!IsDataTypeSupported(ALayout->type) || !IsDataTypeSupported(BLayout->type) || !IsDataTypeSupported(DLayout->type) ||
       !CheckComputeTypeCompatibility(desc->computeType, ALayout->type, BLayout->type)) {
     return ACLBLAS_STATUS_NOT_SUPPORTED;
   }
 
   if (workspace != nullptr && ((reinterpret_cast<uintptr_t>(workspace) & 0xF) != 0 || workspaceSizeInBytes == 0)) {
     return ACLBLAS_STATUS_INVALID_VALUE;
   }
 
   aclblasLtMatmulAlgo_t selectedAlgo{};
 
   if (algo != nullptr) {
     selectedAlgo = *algo;
     PackedAlgo packed{};
     if (!DecodeAlgo(selectedAlgo, &packed)) {
       return ACLBLAS_STATUS_INVALID_VALUE;
     }
   } else {
     std::lock_guard<std::mutex> guard(*ctx->mutex);
 
     AlgoKey key;
     key.m = m;
     key.n = n;
     key.k = k;
     key.aType = ALayout->type;
     key.bType = BLayout->type;
     key.cType = typeC;
     key.dType = DLayout->type;
     key.computeType = desc->computeType;
     key.epilogue = desc->epilogue;
     key.transA = (desc->transA != ACLBLAS_OP_N);
     key.transB = (desc->transB != ACLBLAS_OP_N);
 
     auto it = ctx->algoCache->find(key);
     if (it != ctx->algoCache->end()) {
       selectedAlgo = it->second.algo;
       ctx->lruList->splice(ctx->lruList->begin(), *ctx->lruList, it->second.lruIter);
       it->second.lruIter = ctx->lruList->begin();
     } else {
       aclblasLtMatmulHeuristicResult_t heuristic[4]{};
       int returned = 0;
       aclblasStatus_t st = aclblasLtMatmulAlgoGetHeuristic(lightHandle,
                                                            computeDesc,
                                                            Adesc,
                                                            Bdesc,
                                                            Cdesc,
                                                            Ddesc,
                                                            nullptr,
                                                            4,
                                                            heuristic,
                                                            &returned);
       if (st != ACLBLAS_STATUS_SUCCESS || returned <= 0 || heuristic[0].state != ACLBLAS_STATUS_SUCCESS) {
         return ACLBLAS_STATUS_NOT_SUPPORTED;
       }
 
       selectedAlgo = heuristic[0].algo;
 
       if (ctx->algoCache->size() >= ctx->algoCacheMaxSize && !ctx->lruList->empty()) {
         const AlgoKey& oldKey = ctx->lruList->back();
         ctx->algoCache->erase(oldKey);
         ctx->lruList->pop_back();
       }
 
       ctx->lruList->push_front(key);
       CacheEntry entry;
       entry.algo = selectedAlgo;
       entry.lruIter = ctx->lruList->begin();
       (*ctx->algoCache)[key] = entry;
     }
   }
 
   if (selectedAlgo.max_workspace_bytes > 0) {
     if (workspace == nullptr) {
       if (ctx->internalWorkspace == nullptr || ctx->workspaceSize < selectedAlgo.max_workspace_bytes) {
         return ACLBLAS_STATUS_INVALID_VALUE;
       }
       workspace = ctx->internalWorkspace;
       workspaceSizeInBytes = ctx->workspaceSize;
     } else if (workspaceSizeInBytes < selectedAlgo.max_workspace_bytes) {
       return ACLBLAS_STATUS_INVALID_VALUE;
     }
   }
 
   (void)workspace;
   (void)workspaceSizeInBytes;
   (void)alpha;
   (void)beta;
   (void)stream;
 
   // 当前仓库 matmul kernel 接口保持统一入口，dispatch policy 影响后续 kernel 细分实现。
   const uint32_t numBlocks = 24;
   matmul_kernel_do(static_cast<GM_ADDR>(const_cast<void*>(A)),
                    static_cast<GM_ADDR>(const_cast<void*>(B)),
                    static_cast<GM_ADDR>(const_cast<void*>(C)),
                    static_cast<GM_ADDR>(D),
                    static_cast<uint32_t>(m),
                    static_cast<uint32_t>(k),
                    static_cast<uint32_t>(n),
                    numBlocks,
                    stream != nullptr ? stream : ctx->defaultStream);
 
   return ACLBLAS_STATUS_SUCCESS;
 }
 
 } // extern "C"
 