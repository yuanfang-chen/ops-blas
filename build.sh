#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -e

echo "Starting build process..."

# Create build directory if not exists
if [ ! -d "build" ]; then
    mkdir build
fi

# Change to build directory
cd build

# Run CMake and build
echo "Running CMake configuration..."
cmake ..

echo "Building project..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Binary is available at: build/examples/scopy/scopy"

cd ..