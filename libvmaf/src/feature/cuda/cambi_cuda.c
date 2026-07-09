/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *  Copyright 2025 NVIDIA Corporation.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <common.h>
#include <errno.h>
#include <picture_cuda.h>
#include <stdio.h>
#include <string.h>

#include "common/macros.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "luminance_tools.h"
#include "mem.h"
#include "mkdirp.h"
#include "picture.h"
#include "cambi.h"
#include "cambi_helpers.h"

extern const unsigned char cambi_ptx[];

typedef struct CambiCudaBuffers {
    VmafCudaBuffer *c_values;
    VmafCudaBuffer *c_values_histograms;
    VmafCudaBuffer *diffs_to_consider;
    VmafCudaBuffer *tvi_for_diff;
    VmafCudaBuffer *derivative_buffer;
    VmafCudaBuffer *diff_weights;
    VmafCudaBuffer *all_diffs;
    VmafPicture pic;
    float *c_values_host;
    size_t c_values_host_size;
} CambiCudaBuffers;


typedef struct CambiCudaState {
    CUevent event, finished;
    CUstream str, host_stream;
    CUfunction preprocess_u8_s, preprocess_u16_s;
    CUfunction preprocess_u8, preprocess_u16;
    CUfunction preprocess_nodither_u16, preprocess_nodither_u16s;
    CUfunction calculate_derivate;
    CUfunction calculate_spatial_mask;
    CUfunction decimate_kernel;
    CUfunction decimate_and_filter_kernel;
    CUfunction filter_kernel;
    CUfunction calculate_c_values_kernel;
    CambiCudaBuffers device_buffers;
    void *write_score_parameters;
} CambiCudaState;

typedef struct write_score_parameters_cambi {
    VmafFeatureCollector *feature_collector;
    CambiState *s;
    unsigned index;
    int scaled_widths[NUM_SCALES];
    int scaled_heights[NUM_SCALES];
    double topk;
    uint16_t window_size;
} write_score_parameters_cambi;

extern VmafFeatureExtractor vmaf_fex_cambi;

#define SWAP_FLOATS(x, y) { float temp = x; x = y; y = temp; }

static inline int clip(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

static double average_topk_elements(const float *arr, int topk_elements) {
    double sum = 0;
    for (int i = 0; i < topk_elements; i++)
        sum += arr[i];
    return (double)sum / topk_elements;
}

static void quick_select(float *arr, int n, int k) {
    if (n == k) return;
    int left = 0;
    int right = n - 1;
    while (left < right) {
        float pivot = arr[k];
        int i = left;
        int j = right;
        do {
            while (arr[i] > pivot) i++;
            while (arr[j] < pivot) j--;
            if (i <= j) {
                SWAP_FLOATS(arr[i], arr[j]);
                i++;
                j--;
            }
        } while (i <= j);
        if (j < k) left = i;
        if (k < i) right = j;
    }
}

static double spatial_pooling(float *c_values, double topk, unsigned width, unsigned height) {
    int num_elements = height * width;
    int topk_num_elements = clip(topk * num_elements, 1, num_elements);
    quick_select(c_values, num_elements, topk_num_elements);
    return average_topk_elements(c_values, topk_num_elements);
}

static inline uint16_t get_pixels_in_window(uint16_t window_length) {
    uint16_t odd_length = 2 * (window_length >> 1) + 1;
    return odd_length * odd_length;
}

static double weight_scores_per_scale(double *scores_per_scale, uint16_t normalization) {
    double score = 0.0;
    for (unsigned scale = 0; scale < NUM_SCALES; scale++)
        score += (scores_per_scale[scale] * g_scale_weights[scale]);
    return score / normalization;
}

static int allocate_aligned_and_upload_buffer(void *host_ptr, size_t size,
                                              VmafCudaBuffer **device_ptr,
                                              VmafCudaState *cuda_state) {
    int err = vmaf_cuda_buffer_alloc(cuda_state, device_ptr, size);
    if (err) return err;
    CHECK_CUDA(cuda_state->f, cuMemcpyHtoDAsync((*device_ptr)->data, host_ptr, size,
               cuda_state->str));
    return 0;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h) {
    (void) pix_fmt;
    int err = vmaf_fex_cambi.init(fex, pix_fmt, bpc, w, h);
    if (err) return err;

    CambiState *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    s->cambi_cuda_state = calloc(1, sizeof(CambiCudaState));
    CambiCudaState *cu_s = s->cambi_cuda_state;

    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&cu_s->str, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&cu_s->host_stream, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cu_f, cuEventCreate(&cu_s->event, CU_EVENT_DEFAULT));
    CHECK_CUDA(cu_f, cuEventCreate(&cu_s->finished, CU_EVENT_DEFAULT));
    CHECK_CUDA(cu_f, cuEventRecord(cu_s->finished, cu_s->str));
    CUmodule module;
    CHECK_CUDA(cu_f, cuModuleLoadData(&module, cambi_ptx));

    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->preprocess_u8, module, "preprocess_uint8_t_false"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->preprocess_u16, module, "preprocess_uint16_t_false"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->preprocess_u8_s, module, "preprocess_uint8_t_true"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->preprocess_u16_s, module, "preprocess_uint16_t_true"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->preprocess_nodither_u16, module, "preprocess_nodither_uint16_t_false"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->preprocess_nodither_u16s, module, "preprocess_nodither_uint16_t_true"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->calculate_derivate, module, "calculate_derivate"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->calculate_spatial_mask, module, "calculate_spatial_mask"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->decimate_kernel, module, "decimate_kernel"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->decimate_and_filter_kernel, module, "decimate_and_filter_kernel_true"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->filter_kernel, module, "decimate_and_filter_kernel_false"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->calculate_c_values_kernel, module, "calculate_c_values_kernel"));

    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    for (unsigned i = 0; i < PICS_BUFFER_SIZE; i++)
        err |= vmaf_picture_unref(&s->pics[i]);
    if (err) return err;

    int alloc_w = s->full_ref ? MAX(s->src_width, s->enc_width) : s->enc_width;
    int alloc_h = s->full_ref ? MAX(s->src_height, s->enc_height) : s->enc_height;
    for (unsigned i = 0; i < PICS_BUFFER_SIZE; i++) {
        VmafPicture pic;
        VmafCudaCookie cookie;
        cookie.bpc = 10;
        // only allocate the luma plane
        cookie.pix_fmt = VMAF_PIX_FMT_YUV400P;
        cookie.h = alloc_h;
        cookie.w = alloc_w;
        cookie.state = fex->cu_state;
        err |= vmaf_cuda_picture_alloc(&pic, &cookie);
        err |= vmaf_picture_ref(&s->pics[i], &pic);
    }
    /// addtitional image for decimated and filtered
    VmafPicture pic;
    VmafCudaCookie cookie;
    cookie.bpc = 10;
    // only allocate the luma plane
    cookie.pix_fmt = VMAF_PIX_FMT_YUV400P;
    cookie.h = alloc_h;
    cookie.w = alloc_w;
    cookie.state = fex->cu_state;
    err |= vmaf_cuda_picture_alloc(&pic, &cookie);
    err |= vmaf_picture_ref(&cu_s->device_buffers.pic, &pic);

    if (err) return err;

    const uint16_t num_diffs = 1 << s->max_log_contrast;
    const size_t c_values_size = ALIGN_CEIL(alloc_w * sizeof(float)) * alloc_h;
    const size_t histograms_size =
        ALIGN_CEIL(alloc_w * s->buffers.v_band_size * sizeof(uint16_t));
    const size_t diffs_size = ALIGN_CEIL(sizeof(uint16_t)) * num_diffs;
    const size_t weights_size = ALIGN_CEIL(sizeof(int)) * num_diffs;
    const size_t all_diffs_size = ALIGN_CEIL(sizeof(int)) * (2 * num_diffs + 1);

    err |= vmaf_cuda_buffer_alloc(fex->cu_state, &cu_s->device_buffers.c_values, c_values_size);
    // TODO see if these buffers should rather be strided to better accomodate odd resolutions
    size_t derivative_size = alloc_h * alloc_w * sizeof(uint8_t);
    err = vmaf_cuda_buffer_alloc(fex->cu_state, &cu_s->device_buffers.derivative_buffer, derivative_size);
    err |= allocate_aligned_and_upload_buffer(s->buffers.c_values_histograms, histograms_size,
                                              &cu_s->device_buffers.c_values_histograms, fex->cu_state);
    err |= allocate_aligned_and_upload_buffer(s->buffers.diffs_to_consider, diffs_size,
                                              &cu_s->device_buffers.diffs_to_consider, fex->cu_state);
    err |= allocate_aligned_and_upload_buffer(s->buffers.tvi_for_diff, diffs_size,
                                              &cu_s->device_buffers.tvi_for_diff, fex->cu_state);
    err |= allocate_aligned_and_upload_buffer(s->buffers.diff_weights, weights_size,
                                              &cu_s->device_buffers.diff_weights, fex->cu_state);
    // TODO make these buffers constant memory on the device
    err |= allocate_aligned_and_upload_buffer(s->buffers.all_diffs, all_diffs_size,
                                              &cu_s->device_buffers.all_diffs, fex->cu_state);
    if (err) return err;

    cu_s->device_buffers.c_values_host_size = alloc_w * alloc_h * sizeof(float) * NUM_SCALES;
    err = vmaf_cuda_buffer_host_alloc(fex->cu_state, (void**)&cu_s->device_buffers.c_values_host,
                                      cu_s->device_buffers.c_values_host_size);
    if (err) return err;

    cu_s->write_score_parameters = malloc(sizeof(write_score_parameters_cambi));
    if (!cu_s->write_score_parameters) return -ENOMEM;

    ((write_score_parameters_cambi *) cu_s->write_score_parameters)->s = s;
    return err;
}

static void write_scores(write_score_parameters_cambi *params) {
    VmafFeatureCollector *feature_collector = params->feature_collector;
    CambiState *s = params->s;
    CambiCudaState *cu_s = s->cambi_cuda_state;
    float *c_values_host = cu_s->device_buffers.c_values_host;

    double scores_per_scale[NUM_SCALES];
    size_t offset = 0;
    for (unsigned scale = 0; scale < NUM_SCALES; scale++) {
        int w = params->scaled_widths[scale];
        int h = params->scaled_heights[scale];
        scores_per_scale[scale] = spatial_pooling(c_values_host + offset, params->topk, w, h);
        offset += w * h;
    }

    uint16_t pixels_in_window = get_pixels_in_window(params->window_size);
    double dist_score = weight_scores_per_scale(scores_per_scale, pixels_in_window);

    int err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "Cambi_feature_cambi_score",
        MIN(dist_score, s->cambi_max_val), params->index
    );
    if (err) return;

    if (s->full_ref) {
        double src_score = 0.0;
        err = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "cambi_source",
            MIN(src_score, s->cambi_max_val), params->index
        );
        if (err) return;

        double combined_score = MAX(0, dist_score - src_score);
        err = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "cambi_full_reference",
            MIN(combined_score, s->cambi_max_val), params->index
        );
        if (err) return;
    }
}


static int preprocess(VmafFeatureExtractor *fex, CambiState *s, VmafPicture *pic, int width, int height) {
    CudaFunctions *cu_f = fex->cu_state->f;
    CambiCudaState *cuda_state = s->cambi_cuda_state;

    // this is done to ensure that the CPU does not overwrite the buffer params for 'write_scores
    CHECK_CUDA(cu_f, cuEventSynchronize(cuda_state->finished));

    {
        // preprocess
        int block_dim_x = 32;
        int block_dim_y = 8;
        void *kernelParams[] = {
            pic, &s->pics[0], &width, &height, &s->enc_bitdepth
        };
        bool is_scaling = !(pic->w[0] == (unsigned)width && pic->h[0] == (unsigned)height);
        int grid_dim_x = DIV_ROUND_UP(width, block_dim_x);
        int grid_dim_y = DIV_ROUND_UP(height, block_dim_y);

        if (s->enc_bitdepth <= 8) {
            if (is_scaling) {
                CHECK_CUDA(cu_f, cuLaunchKernel(cuda_state->preprocess_u8_s, grid_dim_x, grid_dim_y, 1,
                               block_dim_x, block_dim_y, 1,
                               0, vmaf_cuda_picture_get_stream(pic), kernelParams, NULL));
            } else {
                CHECK_CUDA(cu_f, cuLaunchKernel(cuda_state->preprocess_u8, grid_dim_x, grid_dim_y, 1,
                               block_dim_x, block_dim_y, 1,
                               0, vmaf_cuda_picture_get_stream(pic), kernelParams, NULL));
            }
        } else if (s->enc_bitdepth < 10) {
            if (is_scaling) {
                CHECK_CUDA(cu_f, cuLaunchKernel(cuda_state->preprocess_u16_s, grid_dim_x, grid_dim_y, 1,
                               block_dim_x, block_dim_y, 1,
                               0, vmaf_cuda_picture_get_stream(pic), kernelParams, NULL));
            } else {
                CHECK_CUDA(cu_f, cuLaunchKernel(cuda_state->preprocess_u16, grid_dim_x, grid_dim_y, 1,
                               block_dim_x, block_dim_y, 1,
                               0, vmaf_cuda_picture_get_stream(pic), kernelParams, NULL));
            }
        } else {
            if (is_scaling) {
                CHECK_CUDA(cu_f, cuLaunchKernel(cuda_state->preprocess_nodither_u16s, grid_dim_x, grid_dim_y, 1,
                               block_dim_x, block_dim_y, 1,
                               0, vmaf_cuda_picture_get_stream(pic), kernelParams, NULL));
            } else {
                CHECK_CUDA(cu_f, cuLaunchKernel(cuda_state->preprocess_nodither_u16, grid_dim_x, grid_dim_y, 1,
                               block_dim_x, block_dim_y, 1,
                               0, vmaf_cuda_picture_get_stream(pic), kernelParams, NULL));
            }
        }
    }

    return 0;
}

static void get_spatial_mask(CudaFunctions *cu_f, CambiCudaState *cu_s, const VmafPicture *image, VmafPicture *mask,
                             VmafCudaBuffer *derivative_buffer, int width, int height) {
    uint16_t mask_index = get_mask_index(width, height, MASK_FILTER_SIZE);

    int block_dim_x = 16;
    int block_dim_y = 16;
    int grid_dim_x = DIV_ROUND_UP(width, 16);
    int grid_dim_y = DIV_ROUND_UP(height, 16);

    // calculate derivative
    // derivative looks at right hand side pixel and bottom pixel to determine if they are the same
    // derivate = 0 if the same and 1 if they are different
    void *derivative_params[] = {
        (void*)image, (void*)derivative_buffer, &width, &height
    };
    CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->calculate_derivate,
                                   grid_dim_x, grid_dim_y, 1,
                                   block_dim_x, block_dim_y, 1,
                                   0, cu_s->str,
                                   derivative_params, NULL));

    // the spatial mask is a binary mask it is calculated based on the sum of derivative values (dp)
    // if the dp value is larget than the mask_index calculated by get_mask_index then the mask value is 1 otherwise 0
    void *mask_params[] = {
        (void*)mask, (void*)derivative_buffer,
        &width, &height, &mask_index
    };
    CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->calculate_spatial_mask,
                                   grid_dim_x, grid_dim_y, 1,
                                   block_dim_x, block_dim_y, 1,
                                   0, cu_s->str,
                                   mask_params, NULL));
}


static void decimate_mask(CudaFunctions *cu_f, CambiCudaState *cu_s,
                              VmafPicture *mask, int scaled_width, int scaled_height) {
    int block_dim_x = 16;
    int block_dim_y = 16;
    int grid_dim_x = DIV_ROUND_UP(scaled_width, block_dim_x);
    int grid_dim_y = DIV_ROUND_UP(scaled_height, block_dim_y);

    uint16_t* data = (uint16_t*)mask->data[0];
    int stride = mask->stride[0] / sizeof(uint16_t);

    void *params[] = {&data, &mask->w[0], &mask->h[0],
                    &scaled_width, &scaled_height, &stride};
    CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->decimate_kernel,
                                   grid_dim_x, grid_dim_y, 1,
                                   block_dim_x, block_dim_y, 1,
                                   0, cu_s->str, params, NULL));

}

static void decimate_and_filter_mode(CudaFunctions *cu_f, CambiCudaState *cu_s,
                              VmafPicture *image, VmafPicture *image_decimate,
                              int scaled_width, int scaled_height) {
    int block_dim_x = 16;
    int block_dim_y = 16;
    int grid_dim_x = DIV_ROUND_UP(scaled_width, block_dim_x);
    int grid_dim_y = DIV_ROUND_UP(scaled_height, block_dim_y);

    uint16_t* data = (uint16_t*)image->data[0];
    uint16_t* data_out = (uint16_t*)image_decimate->data[0];
    int stride = image->stride[0] / sizeof(uint16_t);
    int decimate_stride = image_decimate->stride[0] / sizeof(uint16_t);

    void *params[] = {&data, &data_out, &image->w[0], &image->h[0], &scaled_width, &scaled_height,
                      &stride, &decimate_stride};
    size_t smem_size = sizeof(uint16_t) * (block_dim_x + 2) * (block_dim_y + 2) ;
    bool decimate = !(image->w[0] == (unsigned)scaled_width && image->h[0] == (unsigned)scaled_height);
    if (decimate) {
        CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->decimate_and_filter_kernel,
                                       grid_dim_x, grid_dim_y, 1,
                                       block_dim_x, block_dim_y, 1,
                                       smem_size, cu_s->str, params, NULL));
    } else {
        CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->filter_kernel,
                                       grid_dim_x, grid_dim_y, 1,
                                       block_dim_x, block_dim_y, 1,
                                       smem_size   , cu_s->str, params, NULL));
    }
}

static int cambi_score(CudaFunctions *cu_f, CambiState *s, VmafPicture *pics, uint16_t window_size, double topk,
                       uint16_t num_diffs, uint16_t vlt_luma,
                       int width, int height, bool cambi_high_res_speedup,
                       bool write_heatmaps, VmafFeatureCollector *feature_collector, int frame) {
    CambiCudaState *cu_s = s->cambi_cuda_state;

    VmafPicture *image = &pics[0];
    VmafPicture *mask = &pics[1];
    CambiCudaBuffers *buffers = &cu_s->device_buffers;
    VmafPicture *image_decimate = &buffers->pic;

    int scaled_width = width;
    int scaled_height = height;
    get_spatial_mask(cu_f, cu_s, image, mask, buffers->derivative_buffer, width, height);
    write_score_parameters_cambi current_data;
    current_data.index = frame;
    current_data.feature_collector = feature_collector;
    current_data.topk = topk;
    current_data.s = s;
    current_data.window_size = window_size;
    size_t c_values_offset = 0;
    // sync since we are overwriting the host c values inside the loop
    CHECK_CUDA(cu_f, cuStreamSynchronize(cu_s->host_stream));

    for (unsigned scale = 0; scale < NUM_SCALES; scale++) {
        if (scale > 0 || cambi_high_res_speedup) {
            scaled_width = (scaled_width + 1) >> 1;
            scaled_height = (scaled_height + 1) >> 1;
            decimate_and_filter_mode(cu_f, cu_s, image_decimate, image_decimate, scaled_width, scaled_height);
            decimate_mask(cu_f, cu_s, mask, scaled_width, scaled_height);
        } else {
            decimate_and_filter_mode(cu_f, cu_s, image, image_decimate, scaled_width, scaled_height);
        }

        current_data.scaled_widths[scale] = scaled_width;
        current_data.scaled_heights[scale] = scaled_height;

        int block_dim = 16;
        int grid_dim_x = DIV_ROUND_UP(scaled_width, block_dim);
        int grid_dim_y = DIV_ROUND_UP(scaled_height, block_dim);

        void *c_values_params[] = {
            (void*)image_decimate, (void*)mask, (void*)buffers->c_values,
            &scaled_width, &scaled_height, &window_size,
            (void*)buffers->tvi_for_diff, (void*)buffers->diff_weights, (void*)buffers->all_diffs,
            &num_diffs, &vlt_luma
        };
        CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->calculate_c_values_kernel,
                                       grid_dim_x, grid_dim_y, 1,
                                       block_dim, block_dim, 1,
                                       0, cu_s->str, c_values_params, NULL));

        size_t scale_size = scaled_width * scaled_height * sizeof(float);
        CUdeviceptr src = buffers->c_values->data;
        void * dst = buffers->c_values_host + (c_values_offset / sizeof(float));
        CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(dst, src, scale_size, cu_s->str));
        if (write_heatmaps) {
            CHECK_CUDA(cu_f, cuStreamSynchronize(cu_s->str));
            int err = dump_c_values(s->heatmaps_files, buffers->c_values_host, scaled_width, scaled_height, scale, window_size,
                                    num_diffs, s->buffers.diff_weights, frame);
            if (err) return err;
        }
        c_values_offset += scale_size;
    }
    CHECK_CUDA(cu_f, cuEventRecord(cu_s->finished, cu_s->str));
    CHECK_CUDA(cu_f, cuStreamWaitEvent(cu_s->host_stream, cu_s->finished, CU_EVENT_WAIT_DEFAULT));
    *((write_score_parameters_cambi*)cu_s->write_score_parameters) = current_data;
    CHECK_CUDA(cu_f, cuLaunchHostFunc(cu_s->host_stream, (CUhostFn*)write_scores, cu_s->write_score_parameters));
    return 0;
}

static int preprocess_and_extract_cambi(VmafFeatureExtractor *fex, CambiState *s,
    VmafPicture *pic, bool is_src, VmafFeatureCollector *feature_collector, int frame) {

    CudaFunctions *cu_f = fex->cu_state->f;

    int width = is_src ? s->src_width : s->enc_width;
    int height = is_src ? s->src_height : s->enc_height;

    int err = preprocess(fex, s, pic, width, height);
    if (err) return err;

    int window_size = is_src ? s->src_window_size : s->window_size;
    int num_diffs = 1 << s->max_log_contrast;

    double topk;
    if (s->topk != DEFAULT_CAMBI_TOPK_POOLING) {
        topk = s->topk;
    } else {
        topk = s->cambi_topk;
    }

    bool write_heatmaps = s->heatmaps_path && !is_src;
    err = cambi_score(cu_f, s, s->pics, window_size, topk, num_diffs, s->vlt_luma,
                      width, height, (bool) s->cambi_high_res_speedup, write_heatmaps,
                      feature_collector, frame);

    return err;
}


static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector) {
    (void) ref_pic_90;
    (void) dist_pic_90;

    CambiState *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;

    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));

    int err = preprocess_and_extract_cambi(fex, s, dist_pic, false, feature_collector, index);
    if (err) return err;

    if (s->full_ref) {
        err = preprocess_and_extract_cambi(fex, s, ref_pic, true, feature_collector, index);
        if (err) return err;
    }
    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));
    return 0;
}


static int flush(VmafFeatureExtractor *fex,
        VmafFeatureCollector *feature_collector)
{
    (void)feature_collector;

    CambiState *s = fex->priv;
    CambiCudaState *cu_s = s->cambi_cuda_state;
    CudaFunctions *cu_f = fex->cu_state->f;

    CHECK_CUDA(cu_f, cuStreamSynchronize(cu_s->str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(cu_s->host_stream));
    return 1;
}

static int close(VmafFeatureExtractor *fex) {
    CambiState *s = fex->priv;
    CambiCudaState *cu_s = s->cambi_cuda_state;

    int err = 0;
    for (unsigned i = 0; i < PICS_BUFFER_SIZE; i++) {
        err |= vmaf_picture_unref(&s->pics[i]);
    }
    err |= vmaf_picture_unref(&cu_s->device_buffers.pic);

    aligned_free(s->buffers.tvi_for_diff);
    aligned_free(s->buffers.c_values);
    aligned_free(s->buffers.c_values_histograms);
    aligned_free(s->buffers.mask_dp);
    aligned_free(s->buffers.filter_mode_buffer);
    aligned_free(s->buffers.diffs_to_consider);
    aligned_free(s->buffers.diff_weights);
    aligned_free(s->buffers.all_diffs);
    aligned_free(s->buffers.derivative_buffer);

    if (s->heatmaps_path) {
        for (int scale = 0; scale < NUM_SCALES; scale++) {
            fclose(s->heatmaps_files[scale]);
        }
    }

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    if (cu_s->device_buffers.c_values)
        vmaf_cuda_buffer_free(fex->cu_state, cu_s->device_buffers.c_values);
    if (cu_s->device_buffers.c_values_histograms)
        vmaf_cuda_buffer_free(fex->cu_state, cu_s->device_buffers.c_values_histograms);
    if (cu_s->device_buffers.diffs_to_consider)
        vmaf_cuda_buffer_free(fex->cu_state, cu_s->device_buffers.diffs_to_consider);
    if (cu_s->device_buffers.tvi_for_diff)
        vmaf_cuda_buffer_free(fex->cu_state, cu_s->device_buffers.tvi_for_diff);
    if (cu_s->device_buffers.derivative_buffer)
        vmaf_cuda_buffer_free(fex->cu_state, cu_s->device_buffers.derivative_buffer);
    if (cu_s->device_buffers.diff_weights)
        vmaf_cuda_buffer_free(fex->cu_state, cu_s->device_buffers.diff_weights);
    if (cu_s->device_buffers.all_diffs)
        vmaf_cuda_buffer_free(fex->cu_state, cu_s->device_buffers.all_diffs);
    if (cu_s->device_buffers.c_values_host)
        vmaf_cuda_buffer_host_free(fex->cu_state, cu_s->device_buffers.c_values_host);
    if (cu_s->write_score_parameters) free(cu_s->write_score_parameters);
    if (s->cambi_cuda_state) free(s->cambi_cuda_state);

    return err;
}

VmafFeatureExtractor vmaf_fex_cambi_cuda = {
    .name = "cambi_cuda",
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close,
    .options = options,
    .priv_size = sizeof(CambiState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
};
