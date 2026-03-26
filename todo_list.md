# MatrixLayout
aclblasLtMatrixLayout结构体在内部的实现，未手动设定初始值的成员是create函数的入参。
```cpp
struct aclblasLtMatrixLayoutImpl {
  uint32_t magic;
  aclDataType type;
  uint64_t rows;
  uint64_t cols;
  int64_t ld;
  aclblasLtOrder_t order = ACLBLASLT_ORDER_COL;
  int32_t batchCount = 1;
  int64_t stridedBatchOffset = 0;

  static_assert(sizeof(aclblasLtMatrixLayoutImpl) <= sizeof(aclblasLtMatrixLayoutOpaque_t),
                "Impl of aclblasLtMatrixLayout must fit in capsule!");
};
```
```c
aclblasStatus_t aclblasLtMatrixLayoutCreate(aclblasLtMatrixLayout_t* layout,
                                            aclDataType type,
                                            uint64_t rows,
                                            uint64_t cols,
                                            int64_t ld);

aclblasStatus_t aclblasLtMatrixLayoutSetAttribute(aclblasLtMatrixLayout_t layout,    // 矩阵布局描述符
                                                  aclblasLtMatrixLayoutAttribute_t attr,  // 要设置的属性类型
                                                  const void* buf,                   // 属性值的缓冲区指针
                                                  size_t sizeInBytes                 // 缓冲区大小用于校验
);
```
aclblasLtMatrixLayoutAttribute_t可选属性值，已完全支持，和aclblasLtMatrixLayoutImpl成员变量一一对应：
| 枚举值                                            | 数值 | 说明            | 数据类型                          | 默认值                   |
| ---------------------------------------------- | -- | ------------- | ----------------------------- | --------------------- |
| `ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT`          | 0  | Batch 数量（批次数） | `int32_t`                     | 1                     |
| `ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET` | 1  | 跨批次步长（元素个数）   | `int64_t`                     | 0                     |
| `ACLBLASLT_MATRIX_LAYOUT_TYPE`                 | 2  | 矩阵数据类型        | `uint32_t` (aclDataType)      | Create 参数             |
| `ACLBLASLT_MATRIX_LAYOUT_ORDER`                | 3  | 内存布局顺序        | `int32_t` (aclblasLtOrder\_t) | `ACLBLASLT_ORDER_COL` |
| `ACLBLASLT_MATRIX_LAYOUT_ROWS`                 | 4  | 矩阵行数          | `uint64_t`                    | Create 参数             |
| `ACLBLASLT_MATRIX_LAYOUT_COLS`                 | 5  | 矩阵列数          | `uint64_t`                    | Create 参数             |
| `ACLBLASLT_MATRIX_LAYOUT_LD`                   | 6  | 主维度/前导维度      | `int64_t`                     | Create 参数             |


# MatmulDesc

暂时支持部分基础属性，成员变量待扩展：
```cpp
struct aclblasLtMatmulDescImpl {
  uint32_t magic;
  aclblasComputeType_t computeType;
  aclDataType scaleType;
  aclblasOperation_t transA = ACLBLAS_OP_N;
  aclblasOperation_t transB = ACLBLAS_OP_N;
  aclblasLtEpilogue_t epilogue = ACLBLASLT_EPILOGUE_DEFAULT;
  const void* bias = nullptr;
  aclDataType biasDataType = ACL_DT_UNDEFINED;
  static_assert(sizeof(aclblasLtMatmulDescImpl) <= sizeof(aclblasLtMatmulDescOpaque_t),
                "Impl of aclblasLtMatmulDesc must fit in capsule!");
};
```

```c
aclblasStatus_t aclblasLtMatmulDescCreate(aclblasLtMatmulDesc_t* desc,
                                          aclblasComputeType_t computeType,
                                          aclDataType scaleType);

aclblasStatus_t aclblasLtMatmulDescSetAttribute(aclblasLtMatmulDesc_t desc,
                                                aclblasLtMatmulDescAttribute_t attr,
                                                const void* buf,
                                                size_t sizeInBytes);
```

aclblasLtMatmulDescAttribute_t基础属性，0~4已支持，其余属性待扩展：
| 枚举值                                                | 数值 | 说明               | 数据类型                                | 默认值                           |
| -------------------------------------------------- | -- | ---------------- | ----------------------------------- | ----------------------------- |
| `ACLBLASLT_MATMUL_DESC_TRANSA`                     | 0  | A 矩阵转置操作         | `int32_t` (aclblasOperation\_t)     | `ACLBLAS_OP_N`                |
| `ACLBLASLT_MATMUL_DESC_TRANSB`                     | 1  | B 矩阵转置操作         | `int32_t` (aclblasOperation\_t)     | `ACLBLAS_OP_N`                |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE`                   | 2  | Epilogue 后处理函数   | `uint32_t` (aclblasLtEpilogue\_t)   | `ACLBLASLT_EPILOGUE_DEFAULT`  |
| `ACLBLASLT_MATMUL_DESC_BIAS_POINTER`               | 3  | Bias 向量设备指针      | `void*` / `const void*`             | NULL                          |
| `ACLBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`             | 4  | Bias 数据类型        | `int32_t` (aclDataType)             | 同 D 矩阵                        |
| `ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER`            | 5  | A 矩阵缩放因子设备指针     | `void*` / `const void*`             | NULL                          |
| `ACLBLASLT_MATMUL_DESC_B_SCALE_POINTER`            | 6  | B 矩阵缩放因子设备指针     | `void*` / `const void*`             | NULL                          |
| `ACLBLASLT_MATMUL_DESC_C_SCALE_POINTER`            | 7  | C 矩阵缩放因子设备指针     | `void*` / `const void*`             | NULL                          |
| `ACLBLASLT_MATMUL_DESC_D_SCALE_POINTER`            | 8  | D 矩阵缩放因子设备指针     | `void*` / `const void*`             | NULL                          |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER` | 9  | AUX 缓冲区缩放因子指针    | `void*` / `const void*`             | NULL                          |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`       | 10 | Epilogue 辅助缓冲区指针 | `void*` / `const void*`             | NULL                          |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD`            | 11 | AUX 缓冲区主维度       | `int64_t`                           | -                             |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE`  | 12 | AUX 缓冲区批次步长      | `int64_t`                           | -                             |
| `ACLBLASLT_MATMUL_DESC_POINTER_MODE`               | 13 | Alpha/Beta 指针模式  | `int32_t` (aclblasLtPointerMode\_t) | `ACLBLASLT_POINTER_MODE_HOST` |

高级属性：
| 枚举值                                            | 数值 | 说明              | 数据类型                    |
| ---------------------------------------------- | -- | --------------- | ----------------------- |
| `ACLBLASLT_MATMUL_DESC_AMAX_D_POINTER`         | 14 | D 矩阵绝对值最大值的设备指针 | `void*` / `const void*` |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE` | 22 | AUX 数据类型        | `int32_t` (aclDataType) |

扩展属性：
| 枚举值                                  | 数值 | 说明       | 数据类型                           |
| ------------------------------------ | -- | -------- | ------------------------------ |
| `ACLBLASLT_MATMUL_DESC_A_SCALE_MODE` | 31 | A 矩阵缩放模式 | `aclblasLtMatmulMatrixScale_t` |
| `ACLBLASLT_MATMUL_DESC_B_SCALE_MODE` | 32 | B 矩阵缩放模式 | `aclblasLtMatmulMatrixScale_t` |

自定义属性：
| 枚举值                                              | 数值  | 说明         | 数据类型    |
| ------------------------------------------------ | --- | ---------- | ------- |
| `ACLBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT` | 100 | A 矩阵计算输入类型 | -       |
| `ACLBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT` | 101 | B 矩阵计算输入类型 | -       |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT`    | 102 | 激活函数第一个参数  | `float` |
| `ACLBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT`    | 103 | 激活函数第二个参数  | `float` |

# Preference
maxWorkspaceBytes、searchMode已支持，其余成员变量暂时占位，后续待和heuristic接口适配：
```cpp
struct aclblasLtMatmulPreferenceImpl {
  uint32_t magic;
  uint32_t searchMode = 0;
  size_t maxWorkspaceBytes = DEFAULT_WORKSPACE_SIZE;
  // 下列设计的pref属性还未支持，待后续扩展
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

  static_assert(sizeof(aclblasLtMatmulPreferenceImpl) <= sizeof(aclblasLtMatmulPreferenceOpaque_t),
                "Impl of aclblasLtMatmulPreference must fit in capsule!");
};
```

```c
aclblasStatus_t aclblasLtMatmulPreferenceCreate(aclblasLtMatmulPreference_t* pref);

aclblasStatus_t aclblasLtMatmulPreferenceSetAttribute(aclblasLtMatmulPreference_t pref,
                                                      aclblasLtMatmulPreferenceAttribute_t attr,
                                                      const void* buf,
                                                      size_t sizeInBytes);
```

aclblasLtMatmulPreferenceAttribute_t基础属性，已支持：
| 枚举值                                         | 数值 | 说明        | 数据类型       | 默认值       |
| ------------------------------------------- | -- | --------- | ---------- | --------- |
| `ACLBLASLT_MATMUL_PREF_SEARCH_MODE`         | 0  | 搜索模式      | `uint32_t`（aclblasLtMatmulPreferenceAttribute_t） | 0 (启发式搜索) |
| `ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` | 1  | 最大工作空间字节数 | `uint64_t` | 0         |
| `ACLBLASLT_MATMUL_PREF_MAX`                 | 2  | 占位符/最大值   | -          | -         |

和cublasLtMatmulPreferenceAttribute_t支持的可调属性竞分，从“A矩阵对齐”开始的属性待后续进一步支持：
| 属性                                           | 数值 | 说明       | 数据类型       |
| -------------------------------------------- | -- | -------- | ---------- |
| `CUBLASLT_MATMUL_PREF_SEARCH_MODE`           | 0  | 搜索模式     | `uint32_t` |
| `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`   | 1  | 最大工作空间   | `uint64_t` |
| `CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK` | 2  | 归约方案掩码   | `uint32_t` |
| `CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES` | 5  | A矩阵对齐    | `uint32_t` |
| `CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES` | 6  | B矩阵对齐    | `uint32_t` |
| `CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES` | 7  | C矩阵对齐    | `uint32_t` |
| `CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES` | 8  | D矩阵对齐    | `uint32_t` |
| `CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT`       | 9  | 最大 waves | `float`    |
| `CUBLASLT_MATMUL_PREF_IMPL_MASK`             | 12 | 实现掩码     | `uint64_t` |



# lLHandle
``` cpp
constexpr uint32_t ACLBLASLT_HANDLE_MAGIC = 0xACLB1234;
struct aclblasLtHandle {
    // 魔数校验（防止野指针/版本不匹配）
    uint32_t magic = ACLBLASLT_HANDLE_MAGIC;

    // 版本信息
    int versionMajor;
    int versionMinor;

    // 核心 AscendCL 资源
    aclrtContext context;      // AscendCL运行时上下文
    aclrtStream defaultStream;       // 默认流
    int32_t  deviceId;           // 关联设备

    // aclblasLt 特定状态
    void* internalWorkspace;    // 内部 workspace 缓存
    size_t workspaceSize;

    // 线程安全
    std::mutex* mutex;          // 若 handle 非线程安全

    // 设备能力缓存
    int npuArch;              // npu架构对应编号
    size_t maxSharedMemory;

    // 算法缓存（Heuristic 结果缓存）
    std::unordered_map<AlgoKey, aclblasLtMatmulAlgo_t>* algoCache;
    size_t algoCacheMaxSize = 128;  // 最大缓存条目数
    // LRU 链表（用于淘汰策略）
    std::list<AlgoKey>* lruList = nullptr;
};
```

# 完善aclblasLtMatmulAlgoGetHeuristic接口的实现
当前设计的流程图：
```plain
aclblasLtMatmulAlgoGetHeuristic(...)
        │
        ▼
┌─────────────────────────────────────────┐
│  Phase 1: 入口校验                      │
│  ├── Handle 有效性（魔数、上下文、设备）   │
│  ├── 指针非空检查                        │
│  └── 参数范围检查（requestedAlgoCount）    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Phase 2: 提取问题特征                    │
│  ├── 从matrixLayout A/B/Ddesc提取 M,N,K │
│  ├── 提取（A/B/C/D）数据类型              │
│  ├── 从中operationDesc提取转置模式（transA/B）  │
│  ├── 提取计算类型（computeType）           │
│  ├── 提取后处理（epilogue）               │
│  └── 计算特征：small/large/skinny（TODO） │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Phase 3: Ascend 特化算法生成             │
│                                         │
│  3.1 获取硬件能力                         │
│      └── numAICores, l0BufferSize,        │
│          l1BufferSize                   │
│                                         │
│  3.2 确定 L1 Tiling ⭐                 │
│      ├── 候选: (128,256,256), (256,128,256)│
│      ├── 评分: 负载均衡 + L2命中率 + L1数据搬运量（待扩展）    │
│      └── 输出: l1TileM, l1TileN, l1TileK （TODO 命名） │
│                                         │
│  3.3 确定 L0 Tiling ⭐                  │
│      ├── 约束: L0 <= L1切分, 且能整除L1   │
│      ├── 约束: L0切分能放入L0 Cache      │
│      └── 输出: l0TileM, l0TileN, l0TileK （TODO 命名） │
│                                         │
│  3.4 确定 Dispatch Policy ⭐（TODO）       │
│      ├── preferPingpong? → PINGPONG      │
│      ├── k 较长? → MULTI_STAGE           │
│      └── 默认 → SYNC                     │
│                                         │
│  3.5 Split-K 决策 （？）                    │
│      └── 基于 l1LoopsK 和 numAICores      │
│                                         │
│  3.6 Workspace 计算                        │
│      └── Split-K 部分和 + Epilogue 缓冲    │
│                                         │
│  输出：candidates[]（含 L0/L1 配置）       │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Phase 4: 性能建模与评分（Roofline）(TODO 和samples仓对齐)        │
│                                         │
│  对每个 candidate：                       │
│      │                                  │
│      ├── 计算理论指标                     │
│      │   ├── FLOPs = 2*M*N*K             │
│      │   ├── Bytes = A+B+C+D             │
│      │   └── ArithmeticIntensity          │
│      │                                  │
│      ├── 估算执行时间                     │
│      │   ├── 计算时间 = FLOPs/peakTFLOPS │
│      │   ├── 内存时间 = Bytes/peakGBps    │
│      │   └── 取瓶颈（Roofline）           │
│      │                                  │
│      ├── 应用调整因子                     │
│      │   ├── Split-K 开销 +5%             │
│      │   ├── small矩阵启动 +20%              │
│      │   ├── skinny矩阵 +15%                │
│      │                                  │
│      └── 综合评分                         │
│          ├── 性能 60%                     │
│          ├── 内存效率 20%                 │
│          └── 可靠性 20%                   │
│                                         │
│  输出：scoredResults[]                     │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Phase 5: 排序与输出                      │
│  ├── 按 totalScore 降序排序              │
│  ├── 过滤 efficiency < threshold（保留至少1个）│
│  ├── 填充 results[0..requestedAlgoCount] │
│  └── 设置 *returnAlgoCount                │
│                                         │
│  返回：SUCCESS / NOT_SUPPORTED            │
└─────────────────────────────────────────┘
```
对应的接口中aclblasLtMatmulAlgo_t定义：
```cpp
// 算法结构（加入 L0/L1 Tiling 参数）
typedef struct {
    uint32_t magic;                       // 魔数
    uint32_t algoId;                      // 算法 ID
    AlgoType algoType;                    // 算法类型

    // L1 Tiling
    uint32_t l1TileM;                     // L1 Tile M（如 128, 256）
    uint32_t l1TileN;                     // L1 Tile N（如 128, 256）
    uint32_t l1TileK;                     // L1 Tile K（如 64, 128, 256）

    // L0 Tiling
    uint32_t l0TileM;                     // L0 Tile M（如 128）
    uint32_t l0TileN;                     // L0 Tile N（如 256）
    uint32_t l0TileK;                     // L0 Tile K（如 64）

    // 调度策略
    DispatchPolicyType dispatchPolicy;    // MMAD 调度策略
    uint32_t numBuffers;                  // 缓冲数量（单/双/多缓冲）
    uint32_t splitKFactor;                // K 维度拆分

    // 执行参数
    size_t workspaceSize;                 // 所需 workspace
    float peakPerformance;                // 峰值算力（TFlops）
    KernelFuncType kernelFunc;            // 核函数入口
} aclblasLtMatmulAlgo_t;
```

# 完善aclblasLtMatmul接口的实现
当前接口整体的核心流程图设计为：
```plain
aclblasLtMatmul(handle, computeDesc, A, B, C, D, ...)
        │
        ▼
┌─────────────────────────────────────┐
│  Phase 1: Handle 入口校验            │
│  ├── 空指针 + 魔数检查               │
│  ├── Ascend 上下文匹配               │
│  └── 设备一致性                      │
└─────────────────────────────────────┘
        │ 失败返回 INVALID_HANDLE/CONTEXT/DEVICE
        ▼
┌─────────────────────────────────────┐
│  Phase 2: 核心参数校验               │
│                                     │
│  Step 2.1: computeDesc 校验          │
│  ├── 非空检查                        │
│  ├── 魔数/有效性                     │
│  ├── 计算类型合法性                  │
│  └── 后处理操作合法性                │
│           │                         │
│           ▼ 失败                    │
│    INVALID_VALUE / NOT_SUPPORTED    │
│           │                         │
│  Step 2.2: 矩阵指针校验              │
│  ├── A != nullptr ✓                 │
│  ├── B != nullptr ✓                 │
│  ├── D != nullptr ✓                 │
│  └── C 可为空（仅当 beta=0）         │
│           │                         │
│           ▼ 失败                    │
│      INVALID_VALUE                  │
│           │                         │
│  Step 2.3: 布局描述符校验            │
│  ├── Adesc/Bdesc/Ddesc 非空         │
│  └── 提取维度/类型信息               │
│           │                         │
│           ▼ 失败                    │
│      INVALID_VALUE                  │
│           │                         │
│  Step 2.4: 维度匹配校验       │
│  ├── M: A.m == D.m ?                │
│  ├── K: A.k == B.k ?                │
│  └── N: B.n == D.n ?                │
│           │                         │
│           ▼ 失败                    │
│      INVALID_VALUE                  │
│      ("M/K/N mismatch")             │
│           │                         │
│  Step 2.5: 数据类型兼容性            │
│  ├── A/B/C/D 类型支持？              │
│  └── computeType 兼容？              │
│           │                         │
│           ▼ 失败                    │
│    NOT_SUPPORTED                    │
│           │                         │
│  Step 2.6: 内存合法性                │
│  ├── 设备内存检查（可选）             │
│  └── 内存重叠检查                    │
│           │                         │
│           ▼ 失败                    │
│      INVALID_VALUE                  │
│           │                         │
│  Step 2.7: Workspace 校验            │
│  └── 大小足够？                      │
│           │                         │
│           ▼ 失败                    │
│      INVALID_VALUE                  │
│                                     │
└─────────────────────────────────────┘
        │ 所有校验通过
        ▼
┌─────────────────────────────────────┐
│  Phase 3: 算法选择                   │
│                                     │
│  ┌─ 用户指定 algo? ──┐              │
│  │   是 → 校验合法性  │              │
│  │   └── 直接使用    │              │
│  │                   │              │
│  └── 否 → 查 algoCache              │
│           ├── Hit: 取出 + 更新 LRU  │
│           │                        │
│           └── Miss: 启发式搜索      │
│               ├── 生成候选列表       │
│               ├── 过滤不兼容         │
│               ├── 性能估算排序       │
│               ├── 选最优算法         │
│               └── 存入缓存(LRU管理)  │
│                                     │
│  最终确认 Workspace 足够             │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Phase 4: 执行 Ascend GEMM           │
│                                     │
│  4.1 准备 KernelParams              │
│      └── m,n,k,指针,stride,workspace│
│                                     │
│  4.2 根据 algoType 分发:             │
│      ├── BUILT_IN: 预编译 TBE 算子   │
│      ├── CUSTOM_TILE: 动态 Tile 配置 │
│      ├── SPLITK: K维拆分并行规约     │
│      └── ACL_FALLBACK: 基础实现兜底  │
│                                     │
│  4.3 启动核函数                      │
│      └── aclrtLaunchKernel           │
│                                     │
│  4.4 错误处理                        │
│      └── 失败时清除无效缓存条目       │
│                                     │
│  4.5 调试同步（可选）                 │
└─────────────────────────────────────┘
        │
        ▼
    SUCCESS
```