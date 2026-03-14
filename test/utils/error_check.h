#ifndef OPS_BLAS_TEST_UTILS_ERROR_CHECK_H
#define OPS_BLAS_TEST_UTILS_ERROR_CHECK_H

#include <cstdlib>
#include <iostream>

#include "acl/acl.h"
#include "cann_ops_blasLt.h"

#define CHECK_ACLRT(func)                                                               \
  {                                                                                     \
    aclError status = (func);                                                           \
    if (status != ACL_SUCCESS) {                                                        \
      std::cerr << "ACL Runtime Error at " << __FILE__ << ":" << __LINE__ << " (error code: " << status << ")" << std::endl; \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
  }

#define CHECK_ACLBLAS(func)                                                             \
  {                                                                                     \
    aclblasStatus_t status = (func);                                                    \
    if (status != ACLBLAS_STATUS_SUCCESS) {                                             \
      std::cerr << "BLASLT Error at " << __FILE__ << ":" << __LINE__ << " (error code: " << status << ")" << std::endl; \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
  }

#endif // OPS_BLAS_TEST_UTILS_ERROR_CHECK_H
