# aclblasLt 接口文档

## 0. 头文件与目录结构

- **aclblasLt API 头文件**
  - 接口声明、句柄与描述符结构、Lt 专用枚举：
    - `ops-blas/include/cann_ops_blasLt.h`
- **通用 BLAS 状态码与基础枚举**
  - 返回码 `aclblasStatus_t` / `aclblasLtStatus`、计算类型 `aclblasComputeType_t`、矩阵转置枚举 `aclblasOperation_t`：
    - `ops-blas/include/cann_ops_blas_common.h`
- **ACL 基础类型**
  - `aclDataType`、`aclrtStream` 等类型来自：
    - `acl/acl.h`（通过 `cann_ops_blasLt.h` 间接包含）

> **查找接口/枚举定义**：  
> 如需查看某个接口或枚举的完整注释与声明，可在上述目录下打开对应头文件（`cann_ops_blasLt.h` / `cann_ops_blas_common.h`）检索名称。

---

## 1. 模块简介

`aclblasLt` 是面向矩阵乘（GEMM）场景的轻量级高级接口，提供以下能力：

- 句柄生命周期管理（Create/Destroy）
- 矩阵布局描述（MatrixLayout）及属性配置/查询
- Matmul 操作描述（MatmulDesc）及属性配置
- 算法搜索偏好设置（Preference）
- 启发式算法查询（Heuristic）
- 执行矩阵乘计算（Matmul）
- 版本与属性查询（GetVersion/GetProperty）


---

## 2. 框架层基础结构与类型关系

### 2.1 基础句柄与描述符类型（定义于 `cann_ops_blasLt.h`）

- **库上下文句柄**
  - `aclblasLtHandle_t`：`typedef void* aclblasLtHandle_t;`
- **矩阵布局描述符**
  - `aclblasLtMatrixLayoutOpaque_t`：内部保存布局信息的 opaque 结构
  - `aclblasLtMatrixLayout_t`：`typedef aclblasLtMatrixLayoutOpaque_t* aclblasLtMatrixLayout_t;`
- **矩阵乘操作描述符**
  - `aclblasLtMatmulDescOpaque_t`
  - `aclblasLtMatmulDesc_t`：`typedef aclblasLtMatmulDescOpaque_t* aclblasLtMatmulDesc_t;`
- **算法偏好描述符**
  - `aclblasLtMatmulPreferenceOpaque_t`
  - `aclblasLtMatmulPreference_t`：`typedef aclblasLtMatmulPreferenceOpaque_t* aclblasLtMatmulPreference_t;`
- **算法与启发式结构**
  - `aclblasLtMatmulAlgo_t`：算法描述（包含内部编码和 `max_workspace_bytes`）
  - `aclblasLtMatmulHeuristicResult_t`：启发式结果（包含 `algo`、`workspaceSize`、`state`、`wavesCount` 等）

### 2.2 通用状态码与基础枚举（定义于 `cann_ops_blas_common.h`）

- **返回状态枚举**
  - `aclblasStatus_t`：主返回码类型
  - `aclblasLtStatus`：`typedef aclblasStatus_t aclblasLtStatus;`（Lt 接口复用相同状态码集合）
- **矩阵操作枚举**
  - `aclblasOperation_t`：`ACLBLAS_OP_N` / `ACLBLAS_OP_T` / `ACLBLAS_OP_C`
- **计算类型枚举**
  - `aclblasComputeType_t`：`ACLBLAS_COMPUTE_16F` / `ACLBLAS_COMPUTE_32F` 等

### 2.3 Lt 专用枚举与属性（定义于 `cann_ops_blasLt.h`）

- **矩阵存储顺序**
  - `aclblasLtOrder_t`
- **库属性类型**
  - `aclblasLtPropertyType_t`
- **Epilogue 类型**
  - `aclblasLtEpilogue_t`
- **矩阵布局属性**
  - `aclblasLtMatrixLayoutAttribute_t`
- **Matmul 描述符属性**
  - `aclblasLtMatmulDescAttribute_t`
- **Preference 属性**
  - `aclblasLtMatmulPreferenceAttribute_t`

### 2.4 类型关系简图

> 下图仅表达“包含/别名关系”，非 C++ 继承。

```text
aclblasStatus_t  <------------------------------+
       ^                                        |
       | typedef                                |
aclblasLtStatus --------------------------------+

aclblasLtMatmulHeuristicResult_t
  ├── aclblasLtMatmulAlgo_t algo
  ├── size_t workspaceSize
  └── aclblasStatus_t state

aclblasLtMatrixLayout_t
  └── typedef aclblasLtMatrixLayoutOpaque_t*

aclblasLtMatmulDesc_t
  └── typedef aclblasLtMatmulDescOpaque_t*

aclblasLtMatmulPreference_t
  └── typedef aclblasLtMatmulPreferenceOpaque_t*
```

---

## 3. 主要枚举与属性

> **头文件位置**：以下枚举均定义在 `ops-blas/include/cann_ops_blasLt.h` 或 `ops-blas/include/cann_ops_blas_common.h` 中，对应关系见每小节说明。

### 3.1 矩阵存储顺序 `aclblasLtOrder_t`（`cann_ops_blasLt.h`）

- `ACLBLASLT_ORDER_COL`：列主序
- `ACLBLASLT_ORDER_ROW`：行主序

### 3.2 Epilogue 类型 `aclblasLtEpilogue_t`（`cann_ops_blasLt.h`）

- 包含默认、ReLU、GELU、Bias、Sigmoid、Swish、Clamp 及其 Aux/Bias 组合等多种后处理选项：
  - 例如：`ACLBLASLT_EPILOGUE_DEFAULT`、`ACLBLASLT_EPILOGUE_RELU`、`ACLBLASLT_EPILOGUE_BIAS`、
    `ACLBLASLT_EPILOGUE_RELU_BIAS`、`ACLBLASLT_EPILOGUE_GELU` 等。
- 具体枚举值与行为说明可查阅 `cann_ops_blasLt.h` 中 `aclblasLtEpilogue` 的注释。

### 3.3 MatrixLayout 属性 `aclblasLtMatrixLayoutAttribute_t`（`cann_ops_blasLt.h`）

常用属性包括（括号中为类型/含义）：

- `ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT`（`int32_t`，批次数，默认 1）
- `ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`（`int64_t`，跨 batch 偏移元素数）
- `ACLBLASLT_MATRIX_LAYOUT_TYPE`（`uint32_t`，矩阵数据类型，对应 `aclDataType`）
- `ACLBLASLT_MATRIX_LAYOUT_ORDER`（`int32_t`，存储顺序，对应 `aclblasLtOrder_t`）
- `ACLBLASLT_MATRIX_LAYOUT_ROWS`（`uint64_t`，行数）
- `ACLBLASLT_MATRIX_LAYOUT_COLS`（`uint64_t`，列数）
- `ACLBLASLT_MATRIX_LAYOUT_LD`（`int64_t`，leading dimension）

### 3.4 MatmulDesc 属性 `aclblasLtMatmulDescAttribute_t`（`cann_ops_blasLt.h`）

包括但不限于：

- 转置控制：
  - `ACLBLASLT_MATMUL_DESC_TRANSA` / `ACLBLASLT_MATMUL_DESC_TRANSB`（`int32_t`，对应 `aclblasOperation_t`）
- 后处理与 Bias：
  - `ACLBLASLT_MATMUL_DESC_EPILOGUE`（`uint32_t`，对应 `aclblasLtEpilogue_t`）
  - `ACLBLASLT_MATMUL_DESC_BIAS_POINTER`、`ACLBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`
- Scale 相关：
  - `ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER` / `B/C/D_SCALE_POINTER`
- Aux 相关：
  - `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER` /
    `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD` /
    `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE` /
    `ACLBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE`
- Pointer 模式与扩展：
  - `ACLBLASLT_MATMUL_DESC_POINTER_MODE`
  - `ACLBLASLT_MATMUL_DESC_A_SCALE_MODE` / `B_SCALE_MODE`
  - `ACLBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT` / `B_EXT`
  - 激活扩展参数：`ACLBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT` / `ACT_ARG1_EXT`

### 3.5 Preference 属性 `aclblasLtMatmulPreferenceAttribute_t`（`cann_ops_blasLt.h`）

- `ACLBLASLT_MATMUL_PREF_SEARCH_MODE`（`uint32_t`，搜索模式）
- `ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`（`uint64_t`，最大可用 workspace 大小）

---

## 4. API 参考

> **说明**：以下所有接口的返回类型为 `aclblasStatus_t` / `aclblasLtStatus`，返回码含义详见第 7 章。  
> **头文件路径统一为**：`ops-blas/include/cann_ops_blasLt.h`。

### 4.1 版本与属性查询

#### 4.1.1 `aclblasLtGetVersion`（`cann_ops_blasLt.h`）

```c
aclblasStatus_t aclblasLtGetVersion(size_t* version);
```

- **功能**
  - 查询 aclblasLt 打包版本号。
- **参数**
  - `version`（输出）：版本值地址，不能为空。
- **返回**
  - `ACLBLAS_STATUS_SUCCESS`：成功。
  - `ACLBLAS_STATUS_INVALID_VALUE`：`version == NULL` 等非法参数。

#### 4.1.2 `aclblasLtGetProperty`（`cann_ops_blasLt.h`）

```c
aclblasStatus_t aclblasLtGetProperty(aclblasLtPropertyType_t type, int* value);
```

- **功能**
  - 查询库属性（主版本、次版本、补丁号等）。
- **参数**
  - `type`（输入）：属性类型。
  - `value`（输出）：属性值输出地址。
- **返回**
  - `ACLBLAS_STATUS_SUCCESS`：成功。
  - `ACLBLAS_STATUS_INVALID_VALUE`：参数非法或属性类型不支持。

---

### 4.2 库句柄管理

#### 4.2.1 `aclblasLtCreate`（`cann_ops_blasLt.h`）

```c
aclblasStatus_t aclblasLtCreate(aclblasLtHandle_t* handle);
```

- **功能**
  - 创建 aclblasLt 上下文句柄，初始化库并绑定当前设备。
- **参数**
  - `handle`（输出）：返回创建的句柄。
- **返回**
  - `ACLBLAS_STATUS_SUCCESS`
  - `ACLBLAS_STATUS_INVALID_VALUE`：`handle == NULL`
  - `ACLBLAS_STATUS_ALLOC_FAILED`：内部资源分配失败

#### 4.2.2 `aclblasLtDestroy`（`cann_ops_blasLt.h`）

```c
aclblasStatus_t aclblasLtDestroy(const aclblasLtHandle_t handle);
```

- **功能**
  - 销毁句柄并释放资源，可触发设备同步。
- **参数**
  - `handle`（输入）：待销毁句柄。
- **返回**
  - `ACLBLAS_STATUS_SUCCESS`
  - `ACLBLAS_STATUS_NOT_INITIALIZED`：库未初始化
  - `ACLBLAS_STATUS_INVALID_VALUE`：`handle == NULL`

---

### 4.3 MatrixLayout 描述符（`cann_ops_blasLt.h`）

#### 4.3.1 `aclblasLtMatrixLayoutCreate`

```c
aclblasStatus_t aclblasLtMatrixLayoutCreate(aclblasLtMatrixLayout_t* matLayout,
                                            aclDataType type,
                                            uint64_t rows,
                                            uint64_t cols,
                                            int64_t ld);
```

- **功能**
  - 创建矩阵布局描述符，配置基础形状与数据类型。
- **参数说明**
  - `type`：数据类型（`aclDataType`）。
  - `rows/cols`：矩阵行列。
  - `ld`：leading dimension。
- **返回**
  - `ACLBLAS_STATUS_SUCCESS`
  - `ACLBLAS_STATUS_ALLOC_FAILED`

#### 4.3.2 `aclblasLtMatrixLayoutDestroy`

```c
aclblasStatus_t aclblasLtMatrixLayoutDestroy(const aclblasLtMatrixLayout_t matLayout);
```

- **功能**
  - 销毁矩阵布局描述符。
- **返回**
  - `ACLBLAS_STATUS_SUCCESS`

#### 4.3.3 `aclblasLtMatrixLayoutSetAttribute`

```c
aclblasStatus_t aclblasLtMatrixLayoutSetAttribute(aclblasLtMatrixLayout_t matLayout,
                                                  aclblasLtMatrixLayoutAttribute_t attr,
                                                  const void* buf,
                                                  size_t sizeInBytes);
```

- **功能**
  - 设置 MatrixLayout 属性值。
- **返回**
  - `ACLBLAS_STATUS_SUCCESS`
  - `ACLBLAS_STATUS_INVALID_VALUE`：`buf == NULL` 或 `sizeInBytes` 不匹配

#### 4.3.4 `aclblasLtMatrixLayoutGetAttribute`

```c
aclblasStatus_t aclblasLtMatrixLayoutGetAttribute(const aclblasLtMatrixLayout_t matLayout,
                                                  aclblasLtMatrixLayoutAttribute_t attr,
                                                  void* buf,
                                                  size_t sizeInBytes,
                                                  size_t* sizeWritten);
```

- **功能**
  - 查询 MatrixLayout 属性值。
- **典型失败返回**
  - `ACLBLAS_STATUS_INVALID_VALUE`
  - `ACLBLAS_STATUS_NOT_SUPPORTED`

---

### 4.4 MatmulDesc 描述符（`cann_ops_blasLt.h`）

#### 4.4.1 `aclblasLtMatmulDescCreate`

```c
aclblasStatus_t aclblasLtMatmulDescCreate(aclblasLtMatmulDesc_t* matmulDesc,
                                          aclblasComputeType_t computeType,
                                          aclDataType scaleType);
```

- **功能**
  - 创建矩阵乘操作描述符，配置计算类型与 scale 类型。
- **返回**
  - `ACLBLAS_STATUS_SUCCESS`
  - `ACLBLAS_STATUS_ALLOC_FAILED`

#### 4.4.2 `aclblasLtMatmulDescDestroy`

```c
aclblasStatus_t aclblasLtMatmulDescDestroy(const aclblasLtMatmulDesc_t matmulDesc);
```

#### 4.4.3 `aclblasLtMatmulDescSetAttribute`

```c
aclblasStatus_t aclblasLtMatmulDescSetAttribute(aclblasLtMatmulDesc_t matmulDesc,
                                                aclblasLtMatmulDescAttribute_t attr,
                                                const void* buf,
                                                size_t sizeInBytes);
```

- **典型失败返回**
  - `ACLBLAS_STATUS_INVALID_VALUE`：`buf == NULL` 或大小不匹配

#### 4.4.4 `aclblasLtMatmulDescGetAttribute`（`cann_ops_blasLt.h`）

```c
aclblasStatus_t aclblasLtMatmulDescGetAttribute(aclblasLtMatmulDesc_t desc,
                                                aclblasLtMatmulDescAttribute_t attr,
                                                void* buf,
                                                size_t sizeInBytes,
                                                size_t* sizeWritten);
```

---

### 4.5 MatmulPreference 描述符（`cann_ops_blasLt.h`）

#### 4.5.1 `aclblasLtMatmulPreferenceCreate`

```c
aclblasStatus_t aclblasLtMatmulPreferenceCreate(aclblasLtMatmulPreference_t* pref);
```

#### 4.5.2 `aclblasLtMatmulPreferenceDestroy`

```c
aclblasStatus_t aclblasLtMatmulPreferenceDestroy(const aclblasLtMatmulPreference_t pref);
```

#### 4.5.3 `aclblasLtMatmulPreferenceSetAttribute`

```c
aclblasStatus_t aclblasLtMatmulPreferenceSetAttribute(aclblasLtMatmulPreference_t pref,
                                                      aclblasLtMatmulPreferenceAttribute_t attr,
                                                      const void* buf,
                                                      size_t sizeInBytes);
```

- **典型失败返回**
  - `ACLBLAS_STATUS_INVALID_VALUE`

#### 4.5.4 `aclblasLtMatmulPreferenceGetAttribute`

```c
aclblasStatus_t aclblasLtMatmulPreferenceGetAttribute(aclblasLtMatmulPreference_t pref,
                                                      aclblasLtMatmulPreferenceAttribute_t attr,
                                                      void* buf,
                                                      size_t sizeInBytes,
                                                      size_t* sizeWritten);
```

---

### 4.6 启发式算法查询（`cann_ops_blasLt.h`）

#### 4.6.1 `aclblasLtMatmulAlgoGetHeuristic`

```c
aclblasStatus_t aclblasLtMatmulAlgoGetHeuristic(aclblasLtHandle_t handle,
                                                aclblasLtMatmulDesc_t matmulDesc,
                                                aclblasLtMatrixLayout_t Adesc,
                                                aclblasLtMatrixLayout_t Bdesc,
                                                aclblasLtMatrixLayout_t Cdesc,
                                                aclblasLtMatrixLayout_t Ddesc,
                                                aclblasLtMatmulPreference_t pref,
                                                int requestedAlgoCount,
                                                aclblasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                                int* returnAlgoCount);
```

- **功能**
  - 基于输入布局与偏好，返回若干候选算法及其估计性能。
- **典型失败返回**
  - `ACLBLAS_STATUS_NOT_SUPPORTED`：当前配置无可用启发式
  - `ACLBLAS_STATUS_INVALID_VALUE`：`requestedAlgoCount <= 0` 等非法参数

---

### 4.7 矩阵乘执行（`cann_ops_blasLt.h`）

#### 4.7.1 `aclblasLtMatmul`

```c
aclblasStatus_t aclblasLtMatmul(aclblasLtHandle_t handle,
                                aclblasLtMatmulDesc_t matmulDesc,
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
                                aclrtStream stream);
```

- **功能**
  - 执行矩阵乘及线性组合，支持 `C == D` 原位更新。
- **说明**
  - `workspace` 建议满足 16B 对齐。
  - 若 `algo == NULL`，内部可进行一次隐式启发式查询选择算法。
- **典型失败返回**
  - `ACLBLAS_STATUS_EXECUTION_FAILED`：设备执行失败
  - `ACLBLAS_STATUS_ARCH_MISMATCH`：配置与设备架构不匹配
  - `ACLBLAS_STATUS_NOT_SUPPORTED`：当前设备不支持该配置
  - `ACLBLAS_STATUS_INVALID_VALUE`：参数为 NULL 或配置冲突
  - `ACLBLAS_STATUS_NOT_INITIALIZED`：库未初始化

---

## 5. 推荐调用流程

1. 调用 `aclblasLtCreate` 创建句柄。
2. 使用 `aclblasLtMatrixLayoutCreate` 创建 A/B/C/D 的 `MatrixLayout`，并通过 `SetAttribute` 配置属性。
3. 使用 `aclblasLtMatmulDescCreate` 创建 `MatmulDesc`，设置转置、epilogue 等属性。
4. 使用 `aclblasLtMatmulPreferenceCreate` 创建 `MatmulPreference`，设置最大 workspace 等偏好。
5. 调用 `aclblasLtMatmulAlgoGetHeuristic` 获取可用算法。
6. 调用 `aclblasLtMatmul` 执行计算。
7. 依次销毁 Preference/Desc/Layout/Handle。

---

## 6. 最小示例（伪代码）

```c
aclblasLtHandle_t handle;
aclblasLtCreate(&handle);

aclblasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
aclblasLtMatrixLayoutCreate(&Adesc, ACL_FLOAT16, m, k, lda);
aclblasLtMatrixLayoutCreate(&Bdesc, ACL_FLOAT16, k, n, ldb);
aclblasLtMatrixLayoutCreate(&Cdesc, ACL_FLOAT16, m, n, ldc);
aclblasLtMatrixLayoutCreate(&Ddesc, ACL_FLOAT16, m, n, ldd);

aclblasLtMatmulDesc_t opDesc;
aclblasLtMatmulDescCreate(&opDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT);

aclblasLtMatmulPreference_t pref;
aclblasLtMatmulPreferenceCreate(&pref);
size_t workspaceCap = 32 * 1024 * 1024;
aclblasLtMatmulPreferenceSetAttribute(pref,
  ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
  &workspaceCap,
  sizeof(workspaceCap));

aclblasLtMatmulHeuristicResult_t heur[8];
int algoCount = 0;
aclblasLtMatmulAlgoGetHeuristic(handle, opDesc, Adesc, Bdesc, Cdesc, Ddesc,
                                pref, 8, heur, &algoCount);

aclblasLtMatmul(handle, opDesc,
                &alpha, A, Adesc,
                B, Bdesc,
                &beta, C, Cdesc,
                D, Ddesc,
                &heur[0].algo,
                workspace, workspaceBytes,
                stream);

aclblasLtMatmulPreferenceDestroy(pref);
aclblasLtMatmulDescDestroy(opDesc);
aclblasLtMatrixLayoutDestroy(Adesc);
aclblasLtMatrixLayoutDestroy(Bdesc);
aclblasLtMatrixLayoutDestroy(Cdesc);
aclblasLtMatrixLayoutDestroy(Ddesc);
aclblasLtDestroy(handle);
```

---

## 7. 返回码对照表（含定义位置）

> **定义头文件**：`ops-blas/include/cann_ops_blas_common.h`（`aclblasStatus_t`，`typedef aclblasStatus_t aclblasLtStatus;`）

| 枚举名                               | 数值 | 含义                                   | 典型触发场景示例                                                                 | 头文件路径                                      |
|--------------------------------------|------|----------------------------------------|----------------------------------------------------------------------------------|-------------------------------------------------|
| `ACLBLAS_STATUS_SUCCESS`            | 0    | 调用成功                               | 接口执行成功，无错误。                                                           | `ops-blas/include/cann_ops_blas_common.h`       |
| `ACLBLAS_STATUS_NOT_INITIALIZED`    | 1    | 库未初始化                             | 使用未创建或已销毁的 `aclblasLtHandle_t` 调用接口；环境未正确初始化。           | 同上                                            |
| `ACLBLAS_STATUS_ALLOC_FAILED`       | 2    | 资源/内存分配失败                      | 创建描述符、句柄或内部 buffer 时内存不足；设备侧资源分配失败。                   | 同上                                            |
| `ACLBLAS_STATUS_INVALID_VALUE`      | 3    | 非法参数                               | 传入 NULL 指针、size 不匹配、不支持的属性类型/组合、无效 `requestedAlgoCount`。 | 同上                                            |
| `ACLBLAS_STATUS_MAPPING_ERROR`      | 4    | 内存映射/访问失败                      | 访问非法设备内存地址；设备侧 DMA/映射失败。                                     | 同上                                            |
| `ACLBLAS_STATUS_EXECUTION_FAILED`   | 5    | 程序执行失败                           | `aclblasLtMatmul` 在设备上执行过程中出现运行时错误。                             | 同上                                            |
| `ACLBLAS_STATUS_INTERNAL_ERROR`     | 6    | 内部错误                               | 内部逻辑异常或未预期状态，通常为实现内部错误。                                  | 同上                                            |
| `ACLBLAS_STATUS_NOT_SUPPORTED`      | 7    | 功能/配置不支持                        | 当前设备/实现不支持的矩阵尺寸、数据类型、Epilogue 组合或启发式不可用。          | 同上                                            |
| `ACLBLAS_STATUS_ARCH_MISMATCH`      | 8    | 架构不匹配                             | 算法或配置与当前 NPU 架构不兼容。                                                | 同上                                            |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`  | 9    | 句柄为 nullptr                         | 调用接口时传入的 `aclblasLtHandle_t` 或 BLAS 句柄为 NULL。                      | 同上                                            |
| `ACLBLAS_STATUS_INVALID_ENUM`       | 10   | 不支持的枚举值                         | 传入未定义的 `aclblasOperation_t`/`aclblasComputeType_t` 等枚举。                | 同上                                            |
| `ACLBLAS_STATUS_UNKNOWN`            | 11   | 后端返回未知状态码                     | 后端库返回了当前版本未识别的错误码。                                            | 同上                                            |

> **说明**：  
> Lt 接口（如 `aclblasLtMatmul`、`aclblasLtMatmulAlgoGetHeuristic` 等）返回的 `aclblasStatus_t` 与上述枚举一一对应，错误语义以此枚举为准。

---

## 8. 备注

- 接口能力与属性支持范围以当前实现版本为准，详细限制请参考头文件内注释。
- 若文档描述与头文件声明不一致，请**以头文件声明与实际实现行为为准**。
- 开发调试时，建议结合返回码对照表和具体接口注释快速定位问题。