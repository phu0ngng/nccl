/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests for ncclEpCreateHandle output layout and combine round-trip:
 *   NCCL_EP_OUTPUT_LAYOUT_RANK_MAJOR    — tokens grouped by source GPU rank
 *   NCCL_EP_OUTPUT_LAYOUT_EXPERT_MAJOR — tokens grouped by local expert
 *
 * Tests (OutputLayoutTest fixture — top-k=1, each token targets one expert):
 *   RankMajorLayout           — dispatch output slot order (rank-major)
 *   ExpertMajorLayout        — dispatch output slot order (expert-major)
 *   ExpertMajorWithAlignment — dispatch output with per-expert zone alignment
 *   CombineRankMajor          — dispatch + identity expert + combine recovers original values (rank-major)
 *   CombineExpertMajor       — dispatch + identity expert + combine recovers original values (expert-major)
 *   DispatchMeta             — expert_token_counts_padded and expert_token_offsets
 *
 * Tests (TopK2MixedRoutingTest fixture — top-k=2, mixed same-rank and cross-rank routing):
 *   RankMajorLayout              — correct recv counts and no duplication for same-rank pairs
 *   ExpertMajorNoAlign          — E1 duplicated from E0 for same-rank pairs; both zones distinct for cross-rank
 *   ExpertMajorAlignZeroPadding — E1 duplicated+padded for same-rank pairs; E1 filled for cross-rank
 *   ExpertMajorDupTokens        — slot-by-slot equality of E0/E1 zones for same-rank pairs (LCP warp)
 *
 * Setup: 4 ranks (LSA team size must be a multiple of 4), 8 experts total
 *        (experts_per_rank = 2).
 *   Rank 0 hosts E0 (local 0), E1 (local 1)
 *   Rank 1 hosts E2 (local 0), E3 (local 1)
 *   Rank 2 hosts E4 (local 0), E5 (local 1)
 *   Rank 3 hosts E6 (local 0), E7 (local 1)
 *
 * Token values: rank r, local token i → value = r*kNumTokens + i + 1
 *
 * OutputLayoutTest routing (top-k=1): expert_for_token(i) = (g_rank * kNumTokens + i) % kNumExperts
 *   → experts 0-3 receive tokens from ranks 0,2; experts 4-7 from ranks 1,3.
 *   Each local expert receives exactly 2 tokens (from 2 different source ranks).
 *
 *   Rank 0: T0=1, T1=2, T2=3, T3=4    routing: T0→E0, T1→E1, T2→E2, T3→E3
 *   Rank 1: T0=5, T1=6, T2=7, T3=8    routing: T0→E4, T1→E5, T2→E6, T3→E7
 *   Rank 2: T0=9, T1=10, T2=11, T3=12 routing: T0→E0, T1→E1, T2→E2, T3→E3
 *   Rank 3: T0=13,T1=14, T2=15, T3=16 routing: T0→E4, T1→E5, T2→E6, T3→E7
 *
 *   Rank 0 dispatch output (hosts E0,E1 ← ranks 0,2):
 *     rank-major:     slots [0,1]={1,2}   slots [2,3]={9,10}
 *     Expert-major:  slots [0,1]={1,9}   slots [2,3]={2,10}
 *     +alignment=4:  slots [0..3]={1,9,pad,pad}  slots [4..7]={2,10,pad,pad}
 *
 * TopK2MixedRoutingTest routing (top-k=2, fixed for all source ranks):
 *   T0 → E0 (rank 0) AND E1 (rank 0)  ← same-rank pair: both local experts of rank 0
 *   T1 → E2 (rank 1) AND E3 (rank 1)  ← same-rank pair: both local experts of rank 1
 *   T2 → E4 (rank 2) AND E6 (rank 3)  ← cross-rank pair: one expert on each of ranks 2,3
 *   T3 → E5 (rank 2) AND E7 (rank 3)  ← cross-rank pair: one expert on each of ranks 2,3
 *
 *   All 4 source ranks send T_i identically.
 *   Ranks 0,1 receive 4 tokens (T_{g_rank} from each source, rank-major only).
 *   Ranks 2,3 receive 8 tokens (T2 and T3 from each source, one token per local expert).
 *
 *   Same-rank pair behavior (ranks 0,1) — rank 0 shown:
 *     rank-major:             slots [0..3] = {1,5,9,13}  (no slot duplication; 4 recv)
 *     Expert-major no-align: E0 zone [0..3] = {1,5,9,13}, E1 zone [4..7] = {1,5,9,13} (LCP; 8 recv)
 *     Expert-major align=4:  E0 zone [0..3] = {1,5,9,13}, E1 zone [4..7] = {1,5,9,13} (LCP; 8 recv)
 *
 *   Cross-rank pair behavior (ranks 2,3) — rank 2 shown (E4=local-E0, E5=local-E1):
 *     rank-major:             slots [0..7] = {3,4,7,8,11,12,15,16} (T2 and T3, grouped by source)
 *     Expert-major no-align: E0 zone = {3,7,11,15} (T2→E4), E1 zone = {4,8,12,16} (T3→E5)
 *     Expert-major align=4:  same zones, each padded to 4 (already exactly 4 tokens)
 *
 * Build:  make -C contrib/nccl_ep/tests [BUILDDIR=...]
 * Run:    bash contrib/nccl_ep/tests/run_tests.sh 4
 */

#include "test_common.h"
#include <set>

static float bf16_val(nv_bfloat16 v) { return __bfloat162float(v); }

// ── Test fixture ──────────────────────────────────────────────────────────────

class OutputLayoutTest : public EpTestBase {
protected:
    // All ranks run dispatch then combine (identity expert: dispatch output → combine input).
    // With topk=1 weight=1.0 and identity expert, combine output[i] == original token value.
    // Returns per-token first-hidden-element for the kNumTokens combined output tokens.
    std::vector<float> run_dispatch_combine(ncclEpHandle_t handle, int num_recv) {
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh)
                h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }

        nv_bfloat16 *d_tok, *d_recv, *d_out;
        float*       d_weights;
        EXPECT_EQ(cudaMalloc(&d_tok,     kNumTokens * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv,    num_recv   * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_out,     kNumTokens * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_weights, kNumTokens * kTopK   * sizeof(float)),       cudaSuccess);
        EXPECT_EQ(cudaMemset(d_recv, 0,  num_recv   * kHidden * sizeof(nv_bfloat16)), cudaSuccess);

        std::vector<float> h_w(kNumTokens * kTopK, 1.0f);
        EXPECT_EQ(cudaMemcpy(d_tok,     h_tok.data(), kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d_weights, h_w.data(),   kNumTokens*kTopK*sizeof(float),         cudaMemcpyHostToDevice), cudaSuccess);

        ncclNDTensor_t t_tok, t_recv, t_out, t_w;
        EXPECT_EQ(ncclEpTensorCreate(g_ep_group, &t_tok,  2, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS,       d_tok,     kNumTokens, kHidden), ncclSuccess);
        EXPECT_EQ(ncclEpTensorCreate(g_ep_group, &t_recv, 2, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS,       d_recv,    num_recv,   kHidden), ncclSuccess);
        EXPECT_EQ(ncclEpTensorCreate(g_ep_group, &t_out,  2, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS,       d_out,     kNumTokens, kHidden), ncclSuccess);
        EXPECT_EQ(ncclEpTensorCreate(g_ep_group, &t_w,    2, ncclFloat32,  NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, d_weights, kNumTokens, kTopK),   ncclSuccess);

        const ncclNDTensor_t d_inputs[]  = {t_tok, t_w, topk_idx_};
        const ncclNDTensor_t d_outputs[] = {t_recv};
        ncclEpDispatchConfig_t dcfg{};
        EXPECT_EQ(ncclEpDispatch(handle, d_inputs, 3, d_outputs, 1, nullptr, 0, 0, &dcfg, g_stream), ncclSuccess);
        EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        const ncclNDTensor_t c_inputs[]  = {t_recv};
        const ncclNDTensor_t c_outputs[] = {t_out};
        EXPECT_EQ(ncclEpCombine(handle, c_inputs, 1, c_outputs, 1, nullptr, 0, 0, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        std::vector<nv_bfloat16> h_out(kNumTokens * kHidden);
        EXPECT_EQ(cudaMemcpy(h_out.data(), d_out, kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

        ncclEpTensorDestroy(g_ep_group, t_tok);
        ncclEpTensorDestroy(g_ep_group, t_recv);
        ncclEpTensorDestroy(g_ep_group, t_out);
        ncclEpTensorDestroy(g_ep_group, t_w);
        cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_out); cudaFree(d_weights);

        std::vector<float> vals(kNumTokens);
        for (int i = 0; i < kNumTokens; ++i)
            vals[i] = bf16_val(h_out[i * kHidden]);
        return vals;
    }

    // All ranks run dispatch; returns per-slot first-hidden-element value for this rank.
    std::vector<float> run_dispatch(ncclEpHandle_t handle, int num_recv) {
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh)
                h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }

        nv_bfloat16 *d_tok, *d_recv;
        float*       d_weights;
        EXPECT_EQ(cudaMalloc(&d_tok,     kNumTokens * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv,    num_recv   * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMemset(d_recv, 0,  num_recv   * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_weights, kNumTokens * kTopK   * sizeof(float)),       cudaSuccess);

        std::vector<float> h_w(kNumTokens * kTopK, 1.0f);
        EXPECT_EQ(cudaMemcpy(d_tok,     h_tok.data(), kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d_weights, h_w.data(),   kNumTokens*kTopK  *sizeof(float),       cudaMemcpyHostToDevice), cudaSuccess);

        ncclNDTensor_t t_tok, t_recv, t_w;
        EXPECT_EQ(ncclEpTensorCreate(g_ep_group, &t_tok,  2, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS,       d_tok,     kNumTokens, kHidden), ncclSuccess);
        EXPECT_EQ(ncclEpTensorCreate(g_ep_group, &t_recv, 2, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS,       d_recv,    num_recv,   kHidden), ncclSuccess);
        EXPECT_EQ(ncclEpTensorCreate(g_ep_group, &t_w,    2, ncclFloat32,  NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, d_weights, kNumTokens, kTopK),   ncclSuccess);

        const ncclNDTensor_t inputs[]  = {t_tok, t_w, topk_idx_};
        const ncclNDTensor_t outputs[] = {t_recv};
        ncclEpDispatchConfig_t dcfg{};
        EXPECT_EQ(ncclEpDispatch(handle, inputs, 3, outputs, 1, nullptr, 0, 0, &dcfg, g_stream), ncclSuccess);
        EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        std::vector<nv_bfloat16> h_recv(num_recv * kHidden);
        EXPECT_EQ(cudaMemcpy(h_recv.data(), d_recv, num_recv*kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

        ncclEpTensorDestroy(g_ep_group, t_tok);
        ncclEpTensorDestroy(g_ep_group, t_recv);
        ncclEpTensorDestroy(g_ep_group, t_w);
        cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_weights);

        std::vector<float> vals(num_recv);
        for (int s = 0; s < num_recv; ++s)
            vals[s] = bf16_val(h_recv[s * kHidden]);
        return vals;
    }
};

// ── Test: rank-major layout ────────────────────────────────────────────────────

TEST_F(OutputLayoutTest, RankMajorLayout) {
    ncclEpHandle_t h = make_handle(nullptr);
    ASSERT_NE(h, nullptr);

    // Each rank sends 2 tokens to this rank's experts (1 per expert × 2 experts).
    // Total received = 2 ranks × 2 tokens = 4.
    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandleGetNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 4u);

    auto slots = run_dispatch(h, static_cast<int>(num_recv));

    // rank-major: tokens from the lower-numbered contributing rank in [0,1],
    //            tokens from the higher-numbered rank in [2,3].
    // Ranks 0,2 contribute to experts 0-3; ranks 1,3 contribute to experts 4-7.
    std::set<float> first_half(slots.begin(), slots.begin() + 2);
    std::set<float> second_half(slots.begin() + 2, slots.end());
    if (g_rank == 0) {        // E0,E1 ← ranks 0,2
        EXPECT_EQ(first_half,  (std::set<float>{1.f,  2.f}));
        EXPECT_EQ(second_half, (std::set<float>{9.f, 10.f}));
    } else if (g_rank == 1) { // E2,E3 ← ranks 0,2
        EXPECT_EQ(first_half,  (std::set<float>{ 3.f,  4.f}));
        EXPECT_EQ(second_half, (std::set<float>{11.f, 12.f}));
    } else if (g_rank == 2) { // E4,E5 ← ranks 1,3
        EXPECT_EQ(first_half,  (std::set<float>{ 5.f,  6.f}));
        EXPECT_EQ(second_half, (std::set<float>{13.f, 14.f}));
    } else {                  // E6,E7 ← ranks 1,3
        EXPECT_EQ(first_half,  (std::set<float>{ 7.f,  8.f}));
        EXPECT_EQ(second_half, (std::set<float>{15.f, 16.f}));
    }
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: expert-major layout (no alignment) ──────────────────────────────────

TEST_F(OutputLayoutTest, ExpertMajorLayout) {
    ncclEpHandleConfig cfg{};
    cfg.dispatch_output_layout               = NCCL_EP_OUTPUT_LAYOUT_EXPERT_MAJOR;
    cfg.dispatch_output_per_expert_alignment = 0;

    ncclEpHandle_t h = make_handle(&cfg);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandleGetNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 4u);

    auto slots = run_dispatch(h, static_cast<int>(num_recv));

    // Expert-major: slots [0,1] = local expert 0 tokens, slots [2,3] = local expert 1 tokens.
    std::set<float> e_local0(slots.begin(), slots.begin() + 2);
    std::set<float> e_local1(slots.begin() + 2, slots.end());
    if (g_rank == 0) {        // E0={1,9},  E1={2,10}
        EXPECT_EQ(e_local0, (std::set<float>{ 1.f,  9.f}));
        EXPECT_EQ(e_local1, (std::set<float>{ 2.f, 10.f}));
    } else if (g_rank == 1) { // E2={3,11}, E3={4,12}
        EXPECT_EQ(e_local0, (std::set<float>{ 3.f, 11.f}));
        EXPECT_EQ(e_local1, (std::set<float>{ 4.f, 12.f}));
    } else if (g_rank == 2) { // E4={5,13}, E5={6,14}
        EXPECT_EQ(e_local0, (std::set<float>{ 5.f, 13.f}));
        EXPECT_EQ(e_local1, (std::set<float>{ 6.f, 14.f}));
    } else {                  // E6={7,15}, E7={8,16}
        EXPECT_EQ(e_local0, (std::set<float>{ 7.f, 15.f}));
        EXPECT_EQ(e_local1, (std::set<float>{ 8.f, 16.f}));
    }
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: expert-major + alignment ───────────────────────────────────────────

TEST_F(OutputLayoutTest, ExpertMajorWithAlignment) {
    constexpr size_t kAlign = 4;  // each expert zone padded to 4 tokens

    ncclEpHandleConfig cfg{};
    cfg.dispatch_output_layout               = NCCL_EP_OUTPUT_LAYOUT_EXPERT_MAJOR;
    cfg.dispatch_output_per_expert_alignment = kAlign;

    ncclEpHandle_t h = make_handle(&cfg);
    ASSERT_NE(h, nullptr);

    // GetNumRecvTokens must return the padded total:
    // 2 local experts × align(2 tokens, 4) = 2 × 4 = 8
    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandleGetNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 2u * static_cast<unsigned>(kAlign))
        << "Padded total: 2 local experts × 4 = 8";

    auto slots = run_dispatch(h, static_cast<int>(num_recv));

    // E_local0 zone = slots [0..3], E_local1 zone = slots [4..7]
    std::set<float> zone0, zone1;
    for (int s = 0; s < 4; ++s) if (slots[s] != 0.f) zone0.insert(slots[s]);
    for (int s = 4; s < 8; ++s) if (slots[s] != 0.f) zone1.insert(slots[s]);

    if (g_rank == 0) {
        EXPECT_EQ(zone0, (std::set<float>{ 1.f,  9.f}));
        EXPECT_EQ(zone1, (std::set<float>{ 2.f, 10.f}));
    } else if (g_rank == 1) {
        EXPECT_EQ(zone0, (std::set<float>{ 3.f, 11.f}));
        EXPECT_EQ(zone1, (std::set<float>{ 4.f, 12.f}));
    } else if (g_rank == 2) {
        EXPECT_EQ(zone0, (std::set<float>{ 5.f, 13.f}));
        EXPECT_EQ(zone1, (std::set<float>{ 6.f, 14.f}));
    } else {
        EXPECT_EQ(zone0, (std::set<float>{ 7.f, 15.f}));
        EXPECT_EQ(zone1, (std::set<float>{ 8.f, 16.f}));
    }
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: combine round-trip (rank-major) ─────────────────────────────────────
// dispatch + identity expert + combine must recover the original token values.

TEST_F(OutputLayoutTest, CombineRankMajor) {
    ncclEpHandle_t h = make_handle(nullptr);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandleGetNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 4u);

    auto combined = run_dispatch_combine(h, static_cast<int>(num_recv));

    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(combined[i], expected, 0.5f)
            << "rank " << g_rank << " token " << i;
    }
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: combine round-trip (expert-major, no alignment) ─────────────────────
// Expert-major changes dispatch output slot order; combine uses the remapped
// S2D to route expert outputs back correctly — result identical to rank-major.

TEST_F(OutputLayoutTest, CombineExpertMajor) {
    ncclEpHandleConfig cfg{};
    cfg.dispatch_output_layout               = NCCL_EP_OUTPUT_LAYOUT_EXPERT_MAJOR;
    cfg.dispatch_output_per_expert_alignment = 0;

    ncclEpHandle_t h = make_handle(&cfg);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandleGetNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 4u);

    auto combined = run_dispatch_combine(h, static_cast<int>(num_recv));

    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(combined[i], expected, 0.5f)
            << "rank " << g_rank << " token " << i;
    }
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: dispatch meta — offsets and padded counts ───────────────────────────

TEST_F(OutputLayoutTest, DispatchMeta) {
    constexpr size_t kAlign  = 4;
    const int        E_local = kNumExperts / g_nranks;  // local experts per rank = 2

    ncclEpHandleConfig cfg{};
    cfg.dispatch_output_layout               = NCCL_EP_OUTPUT_LAYOUT_EXPERT_MAJOR;
    cfg.dispatch_output_per_expert_alignment = kAlign;

    int64_t *d_off, *d_cnt;
    CUDA_ASSERT(cudaMalloc(&d_off, E_local * sizeof(int64_t)));
    CUDA_ASSERT(cudaMalloc(&d_cnt, E_local * sizeof(int64_t)));

    ncclNDTensor_t t_off, t_cnt;
    NCCL_ASSERT(ncclEpTensorCreate(g_ep_group, &t_off, 1, ncclInt64,
                                   NCCL_EP_TENSOR_TAG_NONE, d_off, E_local));
    NCCL_ASSERT(ncclEpTensorCreate(g_ep_group, &t_cnt, 1, ncclInt64,
                                   NCCL_EP_TENSOR_TAG_NONE, d_cnt, E_local));

    ncclEpDispatchMeta_t meta{nullptr, t_cnt, t_off};
    ncclEpHandle_t h = make_handle(&cfg, &meta);
    ASSERT_NE(h, nullptr);

    std::vector<int64_t> h_off(E_local), h_cnt(E_local);
    CUDA_ASSERT(cudaMemcpy(h_off.data(), d_off, E_local*sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(h_cnt.data(), d_cnt, E_local*sizeof(int64_t), cudaMemcpyDeviceToHost));

    // Each local expert receives 2 tokens (1 per rank), padded to 4.
    // offsets[0]=0, offsets[1]=4
    EXPECT_EQ(h_cnt[0], static_cast<int64_t>(kAlign)) << "local E0: 2 tokens padded to 4";
    EXPECT_EQ(h_cnt[1], static_cast<int64_t>(kAlign)) << "local E1: 2 tokens padded to 4";
    EXPECT_EQ(h_off[0], 0LL)                          << "local E0 starts at slot 0";
    EXPECT_EQ(h_off[1], static_cast<int64_t>(kAlign)) << "local E1 starts at slot 4";

    ncclEpTensorDestroy(g_ep_group, t_off);
    ncclEpTensorDestroy(g_ep_group, t_cnt);
    cudaFree(d_off); cudaFree(d_cnt);
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── TopK2MixedRoutingTest fixture ─────────────────────────────────────────────
// top-k=2 with a fixed routing mixing same-rank pairs (T0/T1) and cross-rank
// pairs (T2/T3):
//   T0 → E0, E1  (both rank 0 — same-rank pair)
//   T1 → E2, E3  (both rank 1 — same-rank pair)
//   T2 → E4, E6  (rank 2 and rank 3 — cross-rank pair)
//   T3 → E5, E7  (rank 2 and rank 3 — cross-rank pair)
//
// Receives (all 4 source ranks use identical routing):
//   Ranks 0,1: 4 tokens (T_{g_rank} from each source; same-rank pair → no duplication)
//   Ranks 2,3: 8 tokens (T2 and T3 from each source; one token per local expert)

static constexpr int kTopK2 = 2;

class TopK2MixedRoutingTest : public ::testing::Test {
protected:
    ncclNDTensor_t topk_idx2_ = nullptr;
    int64_t*       d_topk2_   = nullptr;

    void SetUp() override {
        CUDA_ASSERT(cudaMalloc(&d_topk2_, kNumTokens * kTopK2 * sizeof(int64_t)));
        const int64_t routing[kNumTokens * kTopK2] = {
            0, 1,  // T0 → E0 (rank 0 local-E0), E1 (rank 0 local-E1)  — same-rank pair
            2, 3,  // T1 → E2 (rank 1 local-E0), E3 (rank 1 local-E1)  — same-rank pair
            4, 6,  // T2 → E4 (rank 2 local-E0), E6 (rank 3 local-E0)  — cross-rank pair
            5, 7,  // T3 → E5 (rank 2 local-E1), E7 (rank 3 local-E1)  — cross-rank pair
        };
        CUDA_ASSERT(cudaMemcpy(d_topk2_, routing, sizeof(routing), cudaMemcpyHostToDevice));
        NCCL_ASSERT(ncclEpTensorCreate(g_ep_group, &topk_idx2_, 2, ncclInt64,
                                       NCCL_EP_TENSOR_TAG_TOPK_IDX,
                                       d_topk2_, kNumTokens, kTopK2));
    }

    void TearDown() override {
        if (topk_idx2_) ncclEpTensorDestroy(g_ep_group, topk_idx2_);
        if (d_topk2_)   cudaFree(d_topk2_);
    }

    ncclEpHandle_t make_handle2(const ncclEpHandleConfig* cfg,
                                 ncclEpDispatchMeta_t* meta = nullptr) {
        ncclEpHandle_t h = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(&h, g_ep_group, topk_idx2_,
                                     cfg, meta, g_stream, false), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        return h;
    }

    std::vector<float> run_dispatch2(ncclEpHandle_t handle, int num_recv) {
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh)
                h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }

        nv_bfloat16 *d_tok, *d_recv;
        float*       d_weights;
        EXPECT_EQ(cudaMalloc(&d_tok,     kNumTokens * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv,    num_recv   * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMemset(d_recv, 0,  num_recv   * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_weights, kNumTokens * kTopK2  * sizeof(float)),       cudaSuccess);

        EXPECT_EQ(cudaMemcpy(d_tok, h_tok.data(), kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
        std::vector<float> h_w(kNumTokens * kTopK2, 1.0f);
        EXPECT_EQ(cudaMemcpy(d_weights, h_w.data(), kNumTokens*kTopK2*sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

        ncclNDTensor_t t_tok, t_recv, t_w;
        EXPECT_EQ(ncclEpTensorCreate(g_ep_group, &t_tok,  2, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS,       d_tok,     kNumTokens, kHidden), ncclSuccess);
        EXPECT_EQ(ncclEpTensorCreate(g_ep_group, &t_recv, 2, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS,       d_recv,    num_recv,   kHidden), ncclSuccess);
        EXPECT_EQ(ncclEpTensorCreate(g_ep_group, &t_w,    2, ncclFloat32,  NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, d_weights, kNumTokens, kTopK2), ncclSuccess);

        const ncclNDTensor_t inputs[]  = {t_tok, t_w, topk_idx2_};
        const ncclNDTensor_t outputs[] = {t_recv};
        ncclEpDispatchConfig_t dcfg{};
        EXPECT_EQ(ncclEpDispatch(handle, inputs, 3, outputs, 1, nullptr, 0, 0, &dcfg, g_stream), ncclSuccess);
        EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        std::vector<nv_bfloat16> h_recv(num_recv * kHidden);
        EXPECT_EQ(cudaMemcpy(h_recv.data(), d_recv, num_recv*kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

        ncclEpTensorDestroy(g_ep_group, t_tok);
        ncclEpTensorDestroy(g_ep_group, t_recv);
        ncclEpTensorDestroy(g_ep_group, t_w);
        cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_weights);

        std::vector<float> vals(num_recv);
        for (int s = 0; s < num_recv; ++s)
            vals[s] = bf16_val(h_recv[s * kHidden]);
        return vals;
    }

    // Full set of token values this rank expects to receive.
    // Ranks 0,1: T_{g_rank} from each of the 4 source ranks (same-rank pair, no duplication).
    // Ranks 2,3: T2 and T3 from each of the 4 source ranks (cross-rank pair).
    std::set<float> expected_recv_set() const {
        std::set<float> s;
        if (g_rank <= 1) {
            for (int src = 0; src < g_nranks; ++src)
                s.insert(static_cast<float>(src * kNumTokens + g_rank + 1));
        } else {
            for (int src = 0; src < g_nranks; ++src) {
                s.insert(static_cast<float>(src * kNumTokens + 3)); // T2
                s.insert(static_cast<float>(src * kNumTokens + 4)); // T3
            }
        }
        return s;
    }

    // Expected non-zero tokens in local-E0 zone (expert-major).
    // Ranks 0,1: T_{g_rank} only (break-on-first-match).
    // Ranks 2,3: T2 values (E4/E6 are both local-E0 on their respective ranks).
    std::set<float> expected_e0_zone() const {
        int tok_idx = (g_rank <= 1) ? g_rank : 2; // T_{g_rank} or T2
        std::set<float> s;
        for (int src = 0; src < g_nranks; ++src)
            s.insert(static_cast<float>(src * kNumTokens + tok_idx + 1));
        return s;
    }

    // Expected tokens in local-E1 zone (expert-major, LCP warp active).
    // Ranks 0,1: same as E0 zone (LCP warp copies primary slot → secondary slot).
    // Ranks 2,3: T3 values (E5/E7 are both local-E1 on their respective ranks).
    std::set<float> expected_e1_zone() const {
        if (g_rank <= 1) return expected_e0_zone(); // LCP: E1 = copy of E0
        std::set<float> s;
        for (int src = 0; src < g_nranks; ++src)
            s.insert(static_cast<float>(src * kNumTokens + 4)); // T3
        return s;
    }
};

// ── Test: rank-major — correct recv counts; no duplication for same-rank pairs ─

TEST_F(TopK2MixedRoutingTest, RankMajorLayout) {
    ncclEpHandle_t h = make_handle2(nullptr);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandleGetNumRecvTokens(h, &num_recv));
    // Same-rank pair (ranks 0,1): token sent once despite 2 local experts targeted.
    // Cross-rank pair (ranks 2,3): T2 and T3 each from 4 sources = 8 tokens.
    const unsigned int expected = (g_rank <= 1) ? 4u : 8u;
    EXPECT_EQ(num_recv, expected);

    auto slots = run_dispatch2(h, static_cast<int>(num_recv));
    std::set<float> got(slots.begin(), slots.end());
    EXPECT_EQ(got, expected_recv_set());
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: expert-major no-align — E1 duplicated from E0 for same-rank pairs ────

TEST_F(TopK2MixedRoutingTest, ExpertMajorNoAlign) {
    ncclEpHandleConfig cfg{};
    cfg.dispatch_output_layout               = NCCL_EP_OUTPUT_LAYOUT_EXPERT_MAJOR;
    cfg.dispatch_output_per_expert_alignment = 0;

    ncclEpHandle_t h = make_handle2(&cfg);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandleGetNumRecvTokens(h, &num_recv));
    // Same-rank pair: LCP warp allocates+fills E0=4 and E1=4 → total 8.
    // Cross-rank pair: E0=4, E1=4 → total 8.
    EXPECT_EQ(num_recv, 8u);

    auto slots = run_dispatch2(h, static_cast<int>(num_recv));
    // E0 zone [0..3] and E1 zone [4..7] both present; set union = expected_recv_set().
    std::set<float> got(slots.begin(), slots.end());
    EXPECT_EQ(got, expected_recv_set());
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: expert-major + align=4 — LCP fills E1; padding zeros trailing slots ─
// Same-rank pair (ranks 0,1): LCP warp copies E0 tokens into E1; no padding needed
//   (actual_counts[1]=4 == alignment, pad=0).
// Cross-rank pair (ranks 2,3): both zones filled with distinct tokens (no padding needed).

TEST_F(TopK2MixedRoutingTest, ExpertMajorAlignZeroPadding) {
    constexpr size_t kAlign = 4;

    ncclEpHandleConfig cfg{};
    cfg.dispatch_output_layout               = NCCL_EP_OUTPUT_LAYOUT_EXPERT_MAJOR;
    cfg.dispatch_output_per_expert_alignment = kAlign;

    ncclEpHandle_t h = make_handle2(&cfg);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandleGetNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 2u * static_cast<unsigned>(kAlign))
        << "2 local experts × align=4 = 8 slots (all ranks)";

    auto slots = run_dispatch2(h, static_cast<int>(num_recv));

    std::set<float> e0_got, e1_got;
    for (size_t s = 0;      s < kAlign;   ++s) if (slots[s] != 0.f) e0_got.insert(slots[s]);
    for (size_t s = kAlign; s < 2*kAlign; ++s) if (slots[s] != 0.f) e1_got.insert(slots[s]);

    EXPECT_EQ(e0_got, expected_e0_zone());
    // E1 zone: LCP warp fills with copies of E0 for same-rank pairs; cross-rank gets real E1 tokens.
    EXPECT_EQ(e1_got, expected_e1_zone());

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: LCP warp — slot-by-slot E0/E1 equality for same-rank pairs ─────────
// Uses no alignment so zone boundaries are clean (E0=[0..3], E1=[4..7]).
// Same-rank pair (ranks 0,1): every E1 slot must equal its matching E0 slot.
// Cross-rank pair (ranks 2,3): E0 and E1 slots are distinct token values.

TEST_F(TopK2MixedRoutingTest, ExpertMajorDupTokens) {
    ncclEpHandleConfig cfg{};
    cfg.dispatch_output_layout               = NCCL_EP_OUTPUT_LAYOUT_EXPERT_MAJOR;
    cfg.dispatch_output_per_expert_alignment = 0;

    ncclEpHandle_t h = make_handle2(&cfg);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandleGetNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 8u) << "2 experts × 4 tokens each = 8 slots";

    auto slots = run_dispatch2(h, static_cast<int>(num_recv));
    // slots[0..3] = E0 zone, slots[4..7] = E1 zone.
    if (g_rank <= 1) {
        // Same-rank pair: LCP warp must have written identical values into E1.
        for (int i = 0; i < 4; ++i)
            EXPECT_EQ(slots[i], slots[i + 4])
                << "E0[" << i << "]=" << slots[i] << " != E1[" << i << "]=" << slots[i+4];
    } else {
        // Cross-rank pair: E0 and E1 carry different token indices (T2 vs T3).
        std::set<float> e0_vals(slots.begin(), slots.begin() + 4);
        std::set<float> e1_vals(slots.begin() + 4, slots.end());
        // E0 ∩ E1 must be empty (T2 values ≠ T3 values).
        std::set<float> intersection;
        for (auto v : e0_vals) if (e1_vals.count(v)) intersection.insert(v);
        EXPECT_TRUE(intersection.empty())
            << "E0 and E1 zones share values for cross-rank pair (should be disjoint)";
        EXPECT_EQ(e0_vals, expected_e0_zone());
        EXPECT_EQ(e1_vals, expected_e1_zone());
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_layout_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
