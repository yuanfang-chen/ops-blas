# ops-blas

## 🔥Latest News
- [2026/03] ops-blas项目上线，提供BLAS计算的API以及现代灵活接口aclBLASLt。

## 🚀概述
ops-blas是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供高性能线性代数计算以及轻量化GEMM调用的算子库。

## ⚡️快速入门
若您希望**从零到一快速体验**项目能力，请访问下述简易教程。

1. [环境部署](docs/zh/install/quick_install.md)：介绍基础环境搭建，包括软件包和三方依赖的获取和安装、源码下载等。

   >  **说明**：本步骤是QuickStart和各类教程的操作前提，请先完成基础环境搭建。
2. [QuickStart](docs/QUICKSTART.md)：提供快速上手本项目能力的指南，包括编译部署、算子调用/开发/调试等核心能力。

## 📖学习教程

若您已学习**环境部署和QuickStart**，对本项目有一定认知，并希望**深入了解和体验项目**，请访问下述详细教程。

1. [接口列表](docs/zh/api_list.md)：提供全量API信息，方便您查看aclblas和aclblasLt接口的分类和功能。


## 🔍目录结构
ops-blas仓关键目录结构如下。
```
ops-blas
├── docs                # 项目文档介绍
├── examples            # 端到端算子开发和调用示例
├── include             # 对外头文件
├── scripts             # 脚本目录
├── blasLt              # blasLt主体源代码目录
├── blas                # blas主体源代码目录
│   ├── utils           # 公共函数
│   ├── dot             # 向量点积算子实现
│   ├── gemv            # 一般矩阵向量乘法算子实现
│   ├──  ...            # 其他算子实现
│   └── CMakeLists.txt  # 算子编译配置文件
├── tests               # 测试代码
```

## 💬相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-linear-algebra)

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/ops-blas/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/ops-blas/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/ops-blas/wiki)获取技术文章，如系列化教程、优秀实践等。