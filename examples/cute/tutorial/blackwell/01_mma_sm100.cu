/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//                             CuTe Tutorial for SM100 Programming
// This tutorial series demonstrates CuTe Blackwell capabilities that are frequently used
// throughout CUTLASS. The goal is to familiarize developers with CuTe SM100 interfaces.
//
// The tutorial series is split into five stages:
// * 01_mma_sm100.cu: Simple Blackwell SM100 GEMM using a tcgen05.mma instruction.
// * 02_mma_tma_sm100.cu: Simple Blackwell SM100 GEMM using tcgen05.mma and TMA instructions.
// * 03_mma_tma_multicast_sm100.cu: Blackwell SM100 GEMM using tcgen05.mma and Multicast TMA.
// * 04_mma_tma_2sm_sm100.cu: Blackwell SM100 GEMM with 2SM tcgen05.mma and 2SM Multicast TMA.
// * 05_mma_tma_epi_sm100.cu: Blackwell SM100 GEMM with 2SM tcgen05.mma, 2SM TMA mainloop, and TMA epilogue.
//
///////////////////////////////////////////////////////////////////////////////////////////////////
1/0