/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Config.h"

#if GSPLAT_BUILD_3DGS

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAStream.h>

#    include "Common.h"
#    include "Dispatch.h"
#    include "Rasterization.h"
#    include "RasterizeToPixels3DGSDevice.cuh"
#    include "Utils.cuh"

namespace gsplat
{
using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;

constexpr int kMaxContribPerPixel = 128;
constexpr float kOneMinusAlphaEps = 1e-9f;

template<uint32_t CDIM, uint32_t TILE_SIZE, uint32_t CTA_SIZE>
__global__ void __launch_bounds__(CTA_SIZE) rasterize_to_pixels_importance_3dgs_kernel(
    const uint32_t N,
    const uint32_t n_isects,
    const vec2 *__restrict__ means2d,      // [I, N, 2]
    const vec3 *__restrict__ conics,       // [I, N, 3]
    const float *__restrict__ colors,      // [I, N, CDIM]
    const float *__restrict__ opacities,   // [I, N]
    const float *__restrict__ backgrounds, // [I, CDIM]
    const bool *__restrict__ masks,        // [I, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t I,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t block_offset,
    const int32_t *__restrict__ isect_offsets, // [I, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,   // [n_isects]
    float *__restrict__ importance_scores      // [I, N]
)
{
    constexpr uint32_t BATCH_SIZE        = CTA_SIZE;
    constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE; // (TILE, CTA) = (16,256)->1, (4,16)->1
    constexpr uint32_t ROW_STRIDE        = CTA_SIZE / TILE_SIZE;
    constexpr uint32_t TILE_MASK         = TILE_SIZE - 1;
    constexpr uint32_t TILE_SHIFT        = __builtin_ctz(TILE_SIZE);
    constexpr uint32_t ALL_DONE          = (1u << PIXELS_PER_THREAD) - 1u;
    static_assert(
        (TILE_SIZE & (TILE_SIZE - 1)) == 0, "TILE_SIZE must be a power of 2 (TILE_MASK/TILE_SHIFT rely on this)"
    );
    static_assert(PIXELS_PER_THREAD > 0, "PIXELS_PER_THREAD == 0 - CTA_SIZE must not exceed TILE_SIZE * TILE_SIZE");

    const uint32_t linear_block_index = blockIdx.x + block_offset;
    const uint32_t tiles_per_image    = tile_width * tile_height;
    const int32_t image_id            = linear_block_index / tiles_per_image;
    const uint32_t tile_linear        = linear_block_index % tiles_per_image;
    const uint32_t grid_width         = tile_width;
    const uint32_t grid_height        = tile_height;

    const uint32_t tile_x = tile_linear % grid_width;
    const uint32_t tile_y = tile_linear / grid_width;
    const int32_t tile_id = tile_y * grid_width + tile_x;

    const uint32_t tid      = threadIdx.x;
    const uint32_t thread_x = tid & TILE_MASK;
    const uint32_t thread_y = tid >> TILE_SHIFT;

    // Narrow per-image buffers to the image handled by this tile.
    isect_offsets     += image_id * grid_height * grid_width;
    importance_scores += image_id * N;
    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }
    if(masks != nullptr)
    {
        masks += image_id * grid_height * grid_width;
        // The mask is uniform for this CTA's tile, so it is safe to exit
        // before any block-wide synchronization. Scores start at zero.
        if(!masks[tile_id])
        {
            return;
        }
    }

    // Convert from tile-local thread coordinates to pixel-center coordinates.
    const uint32_t out_x = tile_x * TILE_SIZE + thread_x;
    const float px       = static_cast<float>(out_x) + 0.5f;

    uint32_t out_y[PIXELS_PER_THREAD];
    float py[PIXELS_PER_THREAD];
#    pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        out_y[p] = tile_y * TILE_SIZE + thread_y + p * ROW_STRIDE;
        py[p]    = static_cast<float>(out_y[p]) + 0.5f;
    }

    // Out-of-bounds pixels cannot return because all threads must
    // participate in the block-wide synchronization below.
    uint32_t done_mask = (out_x >= image_width) ? ALL_DONE : 0;
#    pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        if(out_y[p] >= image_height)
        {
            done_mask |= (1u << p);
        }
    }

    // Every thread in the tile replays the same depth-sorted Gaussian list.
    const int32_t range_start = isect_offsets[tile_id];
    const int32_t range_end
        = (image_id == static_cast<int32_t>(I) - 1) && (tile_id == static_cast<int32_t>(grid_width * grid_height) - 1)
            ? n_isects
            : isect_offsets[tile_id + 1];
    const uint32_t num_batches = (range_end - range_start + BATCH_SIZE - 1) / BATCH_SIZE;

    extern __shared__ int s[];
    int32_t *id_batch      = reinterpret_cast<int32_t *>(s);                          // [BATCH_SIZE]
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[BATCH_SIZE]);         // [BATCH_SIZE]
    vec3 *conic_batch      = reinterpret_cast<vec3 *>(&xy_opacity_batch[BATCH_SIZE]); // [BATCH_SIZE]

    // Save the accepted front-to-back contributors so each can later be
    // removed from the final color without rerasterizing the tile.
    uint32_t history_ids[PIXELS_PER_THREAD][kMaxContribPerPixel];
    vec4 history_data[PIXELS_PER_THREAD][kMaxContribPerPixel];
    vec3 partial_colors[PIXELS_PER_THREAD][kMaxContribPerPixel];
    uint32_t num_contributors[PIXELS_PER_THREAD] = {0u};

    // Accumulate the regular front-to-back render while recording history.
    float T[PIXELS_PER_THREAD];
    float color_accum[PIXELS_PER_THREAD][CDIM] = {0.0f};
#    pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        T[p] = 1.0f;
    }

#    pragma unroll 1
    for(uint32_t b = 0; b < num_batches; ++b)
    {
        // Stage one Gaussian per thread in shared memory for this batch.
        const uint32_t batch_start = range_start + BATCH_SIZE * b;
        const uint32_t idx         = batch_start + tid;
        if(idx < range_end)
        {
            const int32_t global_id = flatten_ids[idx];
            id_batch[tid]           = global_id;
            const vec2 xy           = means2d[global_id];
            const float opac        = opacities[global_id];
            xy_opacity_batch[tid]   = {xy.x, xy.y, opac};
            conic_batch[tid]        = conics[global_id];
        }

        if constexpr(CTA_SIZE <= 32)
        {
            __syncwarp();
        }
        else
        {
            __syncthreads();
        }

        // Composite the staged Gaussians in their sorted order.
        const uint32_t batch_size = min(BATCH_SIZE, static_cast<uint32_t>(range_end - batch_start));
        for(uint32_t t = 0; (t < batch_size) && (done_mask != ALL_DONE); ++t)
        {
            const vec3 conic   = conic_batch[t];
            const vec3 xy_opac = xy_opacity_batch[t];
            const float opac   = xy_opac.z;
            const float dx     = xy_opac.x - px;

#    pragma unroll
            for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
            {
                if(done_mask & (1u << p))
                {
                    continue;
                }

                const float dy          = xy_opac.y - py[p];
                const GaussianWeight gw = eval_gaussian_weight(conic, dx, dy, opac);
                if(!gw.valid)
                {
                    continue;
                }

                const float alpha  = gw.alpha;
                const float next_T = T[p] * (1.0f - alpha);
                // Stop once remaining contributions are negligible or the
                // bounded replay history is full.
                if(next_T <= TRANSMITTANCE_THRESHOLD || num_contributors[p] >= kMaxContribPerPixel)
                {
                    done_mask |= (1u << p);
                    continue;
                }

                const int32_t global_id      = id_batch[t];
                const uint32_t local_id      = static_cast<uint32_t>(global_id % static_cast<int32_t>(N));
                const uint32_t history_idx   = num_contributors[p];
                history_ids[p][history_idx]  = local_id;
                history_data[p][history_idx] = {
                    colors[global_id * CDIM + 0],
                    colors[global_id * CDIM + 1],
                    colors[global_id * CDIM + 2],
                    alpha,
                };
                const float weight = alpha * T[p];
#    pragma unroll
                for(uint32_t ch = 0; ch < CDIM; ++ch)
                {
                    color_accum[p][ch] += colors[global_id * CDIM + ch] * weight;
                }
                ++num_contributors[p];
                T[p] = next_T;
            }
        }

        // Synchronize before reusing shared memory and stop if the tile is done.
        if(__syncthreads_count(done_mask == ALL_DONE) >= BATCH_SIZE)
        {
            break;
        }
    }

#    pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        if(out_x >= image_width || out_y[p] >= image_height)
        {
            continue;
        }
        if(num_contributors[p] == 0)
        {
            continue;
        }

#    pragma unroll
        for(uint32_t ch = 0; ch < CDIM; ++ch)
        {
            color_accum[p][ch] += T[p] * (backgrounds == nullptr ? 0.0f : backgrounds[ch]);
        }

        // Build prefix colors, then remove each contributor in turn. The
        // residual after contributor k is divided by (1 - alpha_k) to
        // reconstruct the color that would remain if it were absent.
        float replay_T       = 1.0f;
        vec3 current_sum     = {0.0f, 0.0f, 0.0f};
        const vec3 final_col = {color_accum[p][0], color_accum[p][1], color_accum[p][2]};
        for(uint32_t i = 0; i < num_contributors[p]; ++i)
        {
            const vec4 hist       = history_data[p][i];
            const vec3 color      = {hist.x, hist.y, hist.z};
            const float alpha     = hist.w;
            current_sum.x        += replay_T * alpha * color.x;
            current_sum.y        += replay_T * alpha * color.y;
            current_sum.z        += replay_T * alpha * color.z;
            partial_colors[p][i]  = current_sum;
            replay_T             *= 1.0f - alpha;
        }

        for(uint32_t k = 0; k < num_contributors[p]; ++k)
        {
            const uint32_t splat_id = history_ids[p][k];
            const float alpha_k     = history_data[p][k].w;

            if(alpha_k >= 1.0f || !isfinite(alpha_k))
            {
                continue;
            }

            const vec3 partial_after        = partial_colors[p][k];
            const vec3 partial_before       = (k == 0) ? vec3{0.0f, 0.0f, 0.0f} : partial_colors[p][k - 1];
            const float inv_one_minus_alpha = 1.0f / (1.0f - alpha_k + kOneMinusAlphaEps);

            if(!isfinite(inv_one_minus_alpha))
            {
                continue;
            }

            const vec3 removed_color = {
                partial_before.x + inv_one_minus_alpha * (final_col.x - partial_after.x),
                partial_before.y + inv_one_minus_alpha * (final_col.y - partial_after.y),
                partial_before.z + inv_one_minus_alpha * (final_col.z - partial_after.z),
            };
            // Squared color change is this pixel's importance contribution.
            const vec3 delta = {
                removed_color.x - final_col.x,
                removed_color.y - final_col.y,
                removed_color.z - final_col.z,
            };
            const float delta_se = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
            if(isfinite(delta_se))
            {
                // Multiple pixels can contribute to the same Gaussian.
                atomicAdd(&importance_scores[splat_id], delta_se);
            }
        }
    }
}

void launch_rasterize_to_pixels_importance_3dgs_kernel(
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    at::Tensor importance_scores
)
{
    const uint32_t N        = means2d.size(-2);
    const uint32_t I        = importance_scores.numel() / N;
    const uint32_t grid_h   = tile_offsets.size(-2);
    const uint32_t grid_w   = tile_offsets.size(-1);
    const uint32_t n_isects = flatten_ids.size(0);
    const uint32_t n_tiles  = I * grid_h * grid_w;
    const dim3 grid         = {n_tiles, 1, 1};

    const int32_t channels = colors.size(-1);
    TORCH_CHECK_VALUE(
        SupportedChannels::contains(channels),
        "Unsupported number of color channels: ",
        channels,
        ". To add support, rebuild gsplat with this channel count included "
        "in -DGSPLAT_NUM_CHANNELS=... (see gsplat/cuda/csrc/Config.h)."
    );
    TORCH_CHECK_VALUE(
        channels == 3,
        "rasterize_to_pixels_importance_3dgs currently supports "
        "RGB features only; got ",
        channels,
        " channels."
    );

    auto launch_kernel = [&]<typename ChannelsT>()
    {
        constexpr uint32_t CDIM = ChannelsT::value;

        auto launch_variant = [&]<uint32_t TILE_SIZE, uint32_t CTA_SIZE>()
        {
            const dim3 threads       = dim3{CTA_SIZE, 1, 1};
            const int64_t shmem_size = CTA_SIZE * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));

            if(cudaFuncSetAttribute(
                   rasterize_to_pixels_importance_3dgs_kernel<CDIM, TILE_SIZE, CTA_SIZE>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   shmem_size
               )
               != cudaSuccess)
            {
                AT_ERROR(
                    "Failed to set maximum shared memory size (requested ",
                    shmem_size,
                    " bytes), try lowering tile_size."
                );
            }

            rasterize_to_pixels_importance_3dgs_kernel<CDIM, TILE_SIZE, CTA_SIZE>
                <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    n_isects,
                    reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
                    reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                    colors.const_data_ptr<float>(),
                    opacities.const_data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
                    masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
                    image_width,
                    image_height,
                    I,
                    grid_w,
                    grid_h,
                    0,
                    tile_offsets.const_data_ptr<int32_t>(),
                    flatten_ids.const_data_ptr<int32_t>(),
                    importance_scores.data_ptr<float>()
                );
        };

        if(tile_size == 16)
        {
            launch_variant.template operator()<16, 256>();
        }
        else if(tile_size == 4)
        {
            launch_variant.template operator()<4, 16>();
        }
        else
        {
            AT_ERROR("Unsupported tile_size ", tile_size, "; supported values are {4, 16}.");
        }
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernel));
    TORCH_CHECK(
        dispatched,
        "dispatch failed: no matching compile-time "
        "instantiation for runtime parameters"
    );
}
} // namespace gsplat

#endif
