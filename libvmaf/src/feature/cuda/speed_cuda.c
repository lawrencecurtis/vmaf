/**
 *
 *  Copyright 2016-2025 Netflix, Inc.
 *  Copyright 2025 NVIDIA Corporation
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
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "common/macros.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "mem.h"
#include "picture.h"
#include "speed_helpers.h"
#include "vif_tools.h"

// PTX bytecode for CUDA kernels (generated from speed.cu)
extern const unsigned char speed_ptx[];

// =============================================================================
// CUDA Buffer Definitions
// =============================================================================

typedef struct SpeedCudaBuffers {
  // Frame buffers (largest allocations)
  // Shape: [alloc_height, alloc_width] in floats
  VmafCudaBuffer *frame_ref;
  VmafCudaBuffer *frame_dis;
  VmafCudaBuffer *frame_temp;  // For decimated image
  VmafCudaBuffer *filter_temp; // For intermediate filtering (3rd buffer needed
                               // for separable filter)

  // Additional frame buffers for temporal (previous frames)
  VmafCudaBuffer *frame_ref_prev;
  VmafCudaBuffer *frame_dis_prev;

  // Block means for covariance computation
  // Shape: [elements_in_block] = [25] floats
  VmafCudaBuffer *block_means;

  // Covariance matrix
  // Shape: [elements_in_block, elements_in_block] = [25, 25] floats
  VmafCudaBuffer *cov_matrix;

  // Eigenvalues of covariance matrix
  // Shape: [elements_in_block] = [25] floats
  VmafCudaBuffer *eigenvalues;

  // Independent term for linear system (Y in KX = Y)
  // Shape: [elements_in_block, num_blocks] = [25, num_blocks] floats
  VmafCudaBuffer *independent_term;

  // Solution to linear system (X in KX = Y)
  // Shape: [elements_in_block, num_blocks] = [25, num_blocks] floats
  VmafCudaBuffer *linear_solution;

  // Result buffers for reference
  // Shape: [num_blocks] floats each
  VmafCudaBuffer *entropies_ref;
  VmafCudaBuffer *variances_ref;

  // Result buffers for distorted
  // Shape: [num_blocks] floats each
  VmafCudaBuffer *entropies_dis;
  VmafCudaBuffer *variances_dis;

  // Filter coefficients
  // Shape: [max_filter_width] floats
  VmafCudaBuffer *filter_coeffs;
  VmafCudaBuffer *antialias_coeffs;

  // cuSOLVER workspace for eigenvalue and QR operations
  VmafCudaBuffer *cusolver_workspace;

  // Host pinned memory for async score readback
  float *score_host;
  size_t score_host_size;
} SpeedCudaBuffers;

// =============================================================================
// CUDA State
// =============================================================================

typedef struct SpeedCudaState {
  // CUDA streams and events
  CUstream str;
  CUstream host_stream;
  CUevent event;
  CUevent finished;

  // Kernel handles - Preprocessing
  CUfunction preprocess_u8_kernel;
  CUfunction preprocess_u16_kernel;
  CUfunction preprocess_prescale_u8_kernel;
  CUfunction preprocess_prescale_u16_kernel;

  // Kernel handles - Filtering
  CUfunction filter_vertical_f32_kernel;
  CUfunction filter_horizontal_f32_kernel;
  CUfunction decimate_16x_kernel;

  // Kernel handles - Image operations
  CUfunction subtract_image_kernel;

  // Kernel handles - Covariance computation
  CUfunction compute_block_means_kernel;
  CUfunction compute_covariance_kernel;

  // Kernel handles - Linear algebra
  CUfunction compute_eigenvalues_kernel;
  CUfunction solve_linear_system_kernel;
  CUfunction check_matrix_regular_kernel;

  // Kernel handles - Block-wise operations
  CUfunction compute_independent_term_kernel;
  CUfunction pointwise_product_div_kernel;
  CUfunction sum_columns_kernel;
  CUfunction update_entropy_kernel;

  // Kernel handles - Score computation
  CUfunction compute_speed_score_kernel;

  // Device buffers
  SpeedCudaBuffers buffers;

  // Filter parameters
  int antialias_filter_width;
  int operating_filter_width;

  // Parameters for async score write callback
  void *write_score_parameters;
} SpeedCudaState;

// =============================================================================
// Write Score Parameters (for async callback)
// =============================================================================

typedef struct WriteScoreParametersChroma {
  VmafFeatureCollector *feature_collector;
  VmafDictionary *feature_name_dict;
  SpeedCudaBuffers *buffers;
  unsigned index;
  int num_blocks;
  float max_val;
} WriteScoreParametersChroma;

typedef struct WriteScoreParametersTemporal {
  VmafFeatureCollector *feature_collector;
  VmafDictionary *feature_name_dict;
  SpeedCudaBuffers *buffers;
  unsigned index;
  int num_blocks;
  float max_val;
} WriteScoreParametersTemporal;

// =============================================================================
// Async Write Score Functions
// =============================================================================

static void write_scores_chroma(WriteScoreParametersChroma *params) {
  // Scores are in score_host: [score_u, score_v]
  float score_u = params->buffers->score_host[0];
  float score_v = params->buffers->score_host[1];

  // Finalize scores (divide by num_blocks)
  if (params->num_blocks > 0) {
    score_u /= params->num_blocks;
    score_v /= params->num_blocks;
  }

  float score_uv = (score_u + score_v) / 2.0f;

  vmaf_feature_collector_append_with_dict(
      params->feature_collector, params->feature_name_dict,
      "Speed_chroma_feature_speed_chroma_u_score",
      fmin(score_u, params->max_val), params->index);
  vmaf_feature_collector_append_with_dict(
      params->feature_collector, params->feature_name_dict,
      "Speed_chroma_feature_speed_chroma_v_score",
      fmin(score_v, params->max_val), params->index);
  vmaf_feature_collector_append_with_dict(
      params->feature_collector, params->feature_name_dict,
      "Speed_chroma_feature_speed_chroma_uv_score",
      fmin(score_uv, params->max_val), params->index);
}

static void write_scores_temporal(WriteScoreParametersTemporal *params) {
  // Score is in score_host[0]
  float score = params->buffers->score_host[0];

  // Finalize score (divide by num_blocks)
  if (params->num_blocks > 0) {
    score /= params->num_blocks;
  }

  vmaf_feature_collector_append_with_dict(
      params->feature_collector, params->feature_name_dict,
      "Speed_temporal_feature_speed_temporal_score",
      fmin(score, params->max_val), params->index);
}

// =============================================================================
// Helper: Allocate CUDA buffer
// =============================================================================

static int alloc_cuda_buffer(VmafCudaState *cu_state, VmafCudaBuffer **buf,
                             size_t size) {
  int err = vmaf_cuda_buffer_alloc(cu_state, buf, size);
  if (err) {
    vmaf_log(VMAF_LOG_LEVEL_ERROR,
             "speed_cuda: failed to allocate CUDA buffer of size %zu", size);
  }
  return err;
}

// =============================================================================
// Helper: Load CUDA kernels
// =============================================================================

static int load_cuda_kernels(SpeedCudaState *cu_s, CudaFunctions *cu_f,
                             CUcontext ctx) {
  CHECK_CUDA(cu_f, cuCtxPushCurrent(ctx));

  // Load PTX module and get kernel functions
  CUmodule module;
  CHECK_CUDA(cu_f, cuModuleLoadData(&module, speed_ptx));

  // Get preprocessing kernel handles
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->preprocess_u8_kernel, module,
                                       "preprocess_kernel_u8"));
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->preprocess_u16_kernel, module,
                                       "preprocess_kernel_u16"));
  CHECK_CUDA(cu_f,
             cuModuleGetFunction(&cu_s->preprocess_prescale_u8_kernel, module,
                                 "preprocess_prescale_kernel_u8"));
  CHECK_CUDA(cu_f,
             cuModuleGetFunction(&cu_s->preprocess_prescale_u16_kernel, module,
                                 "preprocess_prescale_kernel_u16"));

  // Get filtering kernel handles
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->filter_vertical_f32_kernel,
                                       module, "filter_vertical_f32"));
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->filter_horizontal_f32_kernel,
                                       module, "filter_horizontal_f32"));
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->decimate_16x_kernel, module,
                                       "decimate_16x_kernel"));

  // Get image operation kernel handles
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->subtract_image_kernel, module,
                                       "subtract_image_kernel"));

  // Get covariance computation kernel handles
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->compute_block_means_kernel,
                                       module, "compute_block_means_kernel"));
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->compute_covariance_kernel, module,
                                       "compute_covariance_kernel"));

  // Get linear algebra kernel handles
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->compute_eigenvalues_kernel,
                                       module, "compute_eigenvalues_kernel"));
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->solve_linear_system_kernel,
                                       module, "solve_linear_system_kernel"));
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->check_matrix_regular_kernel,
                                       module, "check_matrix_regular_kernel"));

  // Get block-wise operation kernel handles
  CHECK_CUDA(cu_f,
             cuModuleGetFunction(&cu_s->compute_independent_term_kernel, module,
                                 "compute_independent_term_kernel"));
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->pointwise_product_div_kernel,
                                       module, "pointwise_product_div_kernel"));
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->sum_columns_kernel, module,
                                       "sum_columns_kernel"));
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->update_entropy_kernel, module,
                                       "update_entropy_kernel"));

  // Get score computation kernel handles
  CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->compute_speed_score_kernel,
                                       module, "compute_speed_score_kernel"));

  CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));
  return 0;
}

// =============================================================================
// Helper: Allocate common CUDA buffers
// =============================================================================

static int alloc_common_cuda_buffers(VmafFeatureExtractor *fex,
                                     SpeedCudaState *cu_s, SpeedDimensions *dim,
                                     size_t float_stride) {
  int err = 0;
  size_t frame_size = float_stride * dim->alloc_height;
  SpeedCudaBuffers *bufs = &cu_s->buffers;

  err |= alloc_cuda_buffer(fex->cu_state, &bufs->frame_ref, frame_size);
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->frame_dis, frame_size);
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->frame_temp, frame_size);
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->filter_temp, frame_size);

  err |= alloc_cuda_buffer(fex->cu_state, &bufs->block_means,
                           sizeof(float) * dim->elements_in_block);
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->cov_matrix,
                           sizeof(float) * dim->elements_in_block *
                               dim->elements_in_block);
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->eigenvalues,
                           sizeof(float) * dim->elements_in_block);

  err |= alloc_cuda_buffer(fex->cu_state, &bufs->independent_term,
                           sizeof(float) * dim->elements_in_block *
                               dim->num_blocks);
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->linear_solution,
                           sizeof(float) * dim->elements_in_block *
                               dim->num_blocks);

  err |= alloc_cuda_buffer(fex->cu_state, &bufs->entropies_ref,
                           sizeof(float) * dim->num_blocks);
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->variances_ref,
                           sizeof(float) * dim->num_blocks);
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->entropies_dis,
                           sizeof(float) * dim->num_blocks);
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->variances_dis,
                           sizeof(float) * dim->num_blocks);

  // Allocate filter coefficient buffers (max filter width = 128)
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->filter_coeffs,
                           sizeof(float) * 128);
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->antialias_coeffs,
                           sizeof(float) * 128);

  // Allocate cuSOLVER workspace
  size_t cusolver_workspace_size =
      sizeof(float) * (dim->elements_in_block * dim->elements_in_block * 4 +
                       dim->elements_in_block * dim->num_blocks);
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->cusolver_workspace,
                           cusolver_workspace_size);

  if (err)
    return err;

  // Allocate host pinned memory for async score readback
  // Space for 2 scores (U and V for chroma, or 1 for temporal)
  bufs->score_host_size = sizeof(float) * 2;
  err = vmaf_cuda_buffer_host_alloc(fex->cu_state, (void **)&bufs->score_host,
                                    bufs->score_host_size);
  if (err)
    return err;

  return 0;
}

/**
 * Upload Gaussian filter coefficients to GPU
 */
static int upload_filter_coefficients(VmafFeatureExtractor *fex,
                                      SpeedCudaState *cu_s, float kernelscale) {
  CudaFunctions *cu_f = fex->cu_state->f;
  SpeedCudaBuffers *bufs = &cu_s->buffers;

  // Compute antialias filter (for scale 1 -> scale SPEED_NUM_SCALES transition)
  float antialias_filter[128];
  speed_get_antialias_filter(antialias_filter, SPEED_NUM_SCALES, kernelscale);
  cu_s->antialias_filter_width = vif_get_filter_size(1, kernelscale);

  // Compute operating scale filter
  float filter[128];
  vif_get_filter(filter, SPEED_NUM_SCALES, kernelscale);
  cu_s->operating_filter_width =
      vif_get_filter_size(SPEED_NUM_SCALES, kernelscale);

  // Upload to GPU
  CUdeviceptr antialias_ptr, filter_ptr;
  vmaf_cuda_buffer_get_dptr(bufs->antialias_coeffs, &antialias_ptr);
  vmaf_cuda_buffer_get_dptr(bufs->filter_coeffs, &filter_ptr);

  CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
  CHECK_CUDA(cu_f, cuMemcpyHtoD(antialias_ptr, antialias_filter,
                                sizeof(float) * cu_s->antialias_filter_width));
  CHECK_CUDA(cu_f, cuMemcpyHtoD(filter_ptr, filter,
                                sizeof(float) * cu_s->operating_filter_width));
  CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

  return 0;
}

// =============================================================================
// Helper: Free CUDA buffers
// =============================================================================

static void free_cuda_buffers(VmafCudaState *cu_state, SpeedCudaState *cu_s) {
  if (!cu_s)
    return;

  SpeedCudaBuffers *bufs = &cu_s->buffers;

  if (bufs->frame_ref)
    vmaf_cuda_buffer_free(cu_state, bufs->frame_ref);
  if (bufs->frame_dis)
    vmaf_cuda_buffer_free(cu_state, bufs->frame_dis);
  if (bufs->frame_temp)
    vmaf_cuda_buffer_free(cu_state, bufs->frame_temp);
  if (bufs->filter_temp)
    vmaf_cuda_buffer_free(cu_state, bufs->filter_temp);
  if (bufs->frame_ref_prev)
    vmaf_cuda_buffer_free(cu_state, bufs->frame_ref_prev);
  if (bufs->frame_dis_prev)
    vmaf_cuda_buffer_free(cu_state, bufs->frame_dis_prev);
  if (bufs->block_means)
    vmaf_cuda_buffer_free(cu_state, bufs->block_means);
  if (bufs->cov_matrix)
    vmaf_cuda_buffer_free(cu_state, bufs->cov_matrix);
  if (bufs->eigenvalues)
    vmaf_cuda_buffer_free(cu_state, bufs->eigenvalues);
  if (bufs->independent_term)
    vmaf_cuda_buffer_free(cu_state, bufs->independent_term);
  if (bufs->linear_solution)
    vmaf_cuda_buffer_free(cu_state, bufs->linear_solution);
  if (bufs->entropies_ref)
    vmaf_cuda_buffer_free(cu_state, bufs->entropies_ref);
  if (bufs->variances_ref)
    vmaf_cuda_buffer_free(cu_state, bufs->variances_ref);
  if (bufs->entropies_dis)
    vmaf_cuda_buffer_free(cu_state, bufs->entropies_dis);
  if (bufs->variances_dis)
    vmaf_cuda_buffer_free(cu_state, bufs->variances_dis);
  if (bufs->filter_coeffs)
    vmaf_cuda_buffer_free(cu_state, bufs->filter_coeffs);
  if (bufs->antialias_coeffs)
    vmaf_cuda_buffer_free(cu_state, bufs->antialias_coeffs);
  if (bufs->cusolver_workspace)
    vmaf_cuda_buffer_free(cu_state, bufs->cusolver_workspace);

  if (bufs->score_host)
    vmaf_cuda_buffer_host_free(cu_state, bufs->score_host);

  if (cu_s->write_score_parameters)
    free(cu_s->write_score_parameters);

  free(cu_s);
}

// =============================================================================
// Kernel Launch Helpers
// =============================================================================

#define BLOCK_DIM 16
#ifndef DIV_ROUND_UP
#define DIV_ROUND_UP(x, y) (((x) + (y) - 1) / (y))
#endif

/**
 * Launch preprocessing kernel to convert input to float
 */
static int launch_preprocess(VmafFeatureExtractor *fex, SpeedCudaState *cu_s,
                             VmafPicture *pic, VmafCudaBuffer *output,
                             int channel, int width, int height, int out_stride,
                             float offset) {
  CudaFunctions *cu_f = fex->cu_state->f;

  CUdeviceptr input_ptr = (CUdeviceptr)pic->data[channel];
  CUdeviceptr output_ptr;
  vmaf_cuda_buffer_get_dptr(output, &output_ptr);

  int bpc = pic->bpc;

  // Input stride is in bytes from VmafPicture, convert to elements
  // For u8: stride in bytes = stride in elements
  // For u16: stride in bytes / 2 = stride in elements
  int in_stride = (bpc <= 8) ? pic->stride[channel]
                             : pic->stride[channel] / sizeof(uint16_t);

  int grid_x = DIV_ROUND_UP(width, BLOCK_DIM);
  int grid_y = DIV_ROUND_UP(height, BLOCK_DIM);

  CUfunction kernel =
      (bpc <= 8) ? cu_s->preprocess_u8_kernel : cu_s->preprocess_u16_kernel;

  void *params[] = {&input_ptr, &output_ptr,
                    &width, &height, &in_stride,
                    &out_stride, &bpc, &offset};

  CHECK_CUDA(cu_f, cuLaunchKernel(kernel,
    grid_x, grid_y, 1,
    BLOCK_DIM, BLOCK_DIM, 1,
    0, cu_s->str, params, NULL));

  return 0;
}

/**
 * Process a single frame: filter, decimate, local mean subtraction
 * This is the GPU equivalent of filter_and_downscale()
 *
 * Uses 3 buffers to avoid race conditions in separable filtering:
 * - frame: input/output buffer
 * - temp: intermediate buffer (holds decimated image)
 * - filter_temp: extra buffer for filter intermediate
 */
static int process_frame(VmafFeatureExtractor *fex, SpeedCudaState *cu_s,
                         SpeedDimensions *dim, VmafCudaBuffer *frame,
                         VmafCudaBuffer *temp, int stride_px) {
  CudaFunctions *cu_f = fex->cu_state->f;
  SpeedCudaBuffers *bufs = &cu_s->buffers;

  int scaled_w = dim->scaled_width;
  int scaled_h = dim->scaled_height;

  // Get filter coefficients pointers
  CUdeviceptr antialias_ptr, filter_ptr;
  vmaf_cuda_buffer_get_dptr(bufs->antialias_coeffs, &antialias_ptr);
  vmaf_cuda_buffer_get_dptr(bufs->filter_coeffs, &filter_ptr);

  // Get buffer pointers
  CUdeviceptr frame_ptr, temp_ptr, filter_temp_ptr;
  vmaf_cuda_buffer_get_dptr(frame, &frame_ptr);
  vmaf_cuda_buffer_get_dptr(temp, &temp_ptr);
  vmaf_cuda_buffer_get_dptr(bufs->filter_temp, &filter_temp_ptr);

  int antialias_width = cu_s->antialias_filter_width;
  int grid_x = DIV_ROUND_UP(scaled_w, BLOCK_DIM);
  int grid_y = DIV_ROUND_UP(scaled_h, BLOCK_DIM);

  // =========================================================================
  // Step 1: Antialias filter (separable Gaussian)
  // Input: frame, Output: frame (via temp)
  // =========================================================================

  // Vertical: frame -> temp
  void *aa_v_params[] = {&frame_ptr, &temp_ptr,  &antialias_ptr,
                         &scaled_w, &scaled_h,  &stride_px, &antialias_width};
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->filter_vertical_f32_kernel, grid_x,
                                  grid_y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0,
                                  cu_s->str, aa_v_params, NULL));

  // Horizontal: temp -> frame
  void *aa_h_params[] = {&temp_ptr, &frame_ptr, &antialias_ptr,  &scaled_w,
                         &scaled_h, &stride_px, &antialias_width};
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->filter_horizontal_f32_kernel, grid_x,
                                  grid_y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0,
                                  cu_s->str, aa_h_params, NULL));

  // =========================================================================
  // Step 2: Decimate 16x
  // Input: frame (antialias filtered), Output: temp (decimated)
  // =========================================================================

  int op_w = dim->operating_width;
  int op_h = dim->operating_height;
  int op_grid_x = DIV_ROUND_UP(op_w, BLOCK_DIM);
  int op_grid_y = DIV_ROUND_UP(op_h, BLOCK_DIM);

  void *dec_params[] = {&frame_ptr, &temp_ptr,
                        &scaled_w, &scaled_h, &stride_px, // input sizes
                        &op_w,     &op_h,     &stride_px}; // output sizes
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->decimate_16x_kernel,
                          op_grid_x,op_grid_y, 1,
                                  BLOCK_DIM, BLOCK_DIM, 1, 0,
                                  cu_s->str, dec_params, NULL));

  // =========================================================================
  // Step 3: Local mean filter (separable Gaussian) + subtraction
  // We need: decimated - filter(decimated)
  //
  // Buffer usage with 3 buffers:
  // - temp: holds decimated image (preserve this!)
  // - frame: intermediate for vertical output
  // - filter_temp: holds local mean (horizontal output)
  // =========================================================================

  int op_filter_width = cu_s->operating_filter_width;

  // Vertical: temp (decimated) -> frame (intermediate)
  // temp is preserved (only read)
  void *lm_v_params[] = {&temp_ptr, &frame_ptr, &filter_ptr,     &op_w,
                         &op_h,     &stride_px, &op_filter_width};
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->filter_vertical_f32_kernel, op_grid_x,
                                  op_grid_y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0,
                                  cu_s->str, lm_v_params, NULL));

  // Horizontal: frame (intermediate) -> filter_temp (local mean)
  // frame is now available for final result
  void *lm_h_params[] = {&frame_ptr, &filter_temp_ptr, &filter_ptr,     &op_w,
                         &op_h,      &stride_px,       &op_filter_width};
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->filter_horizontal_f32_kernel, op_grid_x,
                                  op_grid_y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0,
                                  cu_s->str, lm_h_params, NULL));

  // =========================================================================
  // Step 4: Local mean subtraction
  // Compute: temp (decimated) - filter_temp (local mean) -> frame (result)
  //
  // Our subtract kernel does: A -= B (A = A - B)
  // We want: frame = temp - filter_temp
  //
  // First copy temp -> frame, then frame -= filter_temp
  // =========================================================================

  // Copy decimated to frame
  size_t copy_size = op_h * stride_px * sizeof(float);
  CHECK_CUDA(cu_f,
             cuMemcpyDtoDAsync(frame_ptr, temp_ptr, copy_size, cu_s->str));

  // Subtract: frame = frame - filter_temp = decimated - local_mean
  void *sub_params[] = {&frame_ptr, &filter_temp_ptr, &op_w, &op_h, &stride_px};
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->subtract_image_kernel, op_grid_x,
                                  op_grid_y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0,
                                  cu_s->str, sub_params, NULL));

  return 0;
}

/**
 * Compute SpEED parameters (entropy and variance) for one frame.
 * This is the GPU equivalent of est_params()
 */
static int compute_speed_params(VmafFeatureExtractor *fex, SpeedCudaState *cu_s,
                                SpeedDimensions *dim, VmafCudaBuffer *frame,
                                VmafCudaBuffer *entropies,
                                VmafCudaBuffer *variances, float sigma_nn,
                                int stride_px) {
  CudaFunctions *cu_f = fex->cu_state->f;
  SpeedCudaBuffers *bufs = &cu_s->buffers;

  int block_size = dim->block_size;
  int elements_in_block = dim->elements_in_block;
  int num_blocks = dim->num_blocks;
  int num_blocks_h = dim->num_blocks_horizontal;
  int num_blocks_v = dim->num_blocks_vertical;
  int submatrix_w = dim->submatrix_width;
  int submatrix_h = dim->submatrix_height;
  int truncated_w = dim->truncated_width;
  int truncated_h = dim->truncated_height;

  CUdeviceptr frame_ptr, means_ptr, cov_ptr, eigen_ptr;
  CUdeviceptr indep_ptr, solution_ptr, entropy_ptr, variance_ptr, workspace_ptr;

  vmaf_cuda_buffer_get_dptr(frame, &frame_ptr);
  vmaf_cuda_buffer_get_dptr(bufs->block_means, &means_ptr);
  vmaf_cuda_buffer_get_dptr(bufs->cov_matrix, &cov_ptr);
  vmaf_cuda_buffer_get_dptr(bufs->eigenvalues, &eigen_ptr);
  vmaf_cuda_buffer_get_dptr(bufs->independent_term, &indep_ptr);
  vmaf_cuda_buffer_get_dptr(bufs->linear_solution, &solution_ptr);
  vmaf_cuda_buffer_get_dptr(entropies, &entropy_ptr);
  vmaf_cuda_buffer_get_dptr(variances, &variance_ptr);
  vmaf_cuda_buffer_get_dptr(bufs->cusolver_workspace, &workspace_ptr);

  int op_w = dim->operating_width;
  int op_h = dim->operating_height;

  // Step 1: Compute block means (single block kernel, 5x5 threads)
  void *means_params[] = {&frame_ptr, &means_ptr,  &op_w,        &op_h,
                          &stride_px, &block_size, &submatrix_w, &submatrix_h};
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->compute_block_means_kernel, 1, 1, 1,
                                  block_size, block_size, 1, 0, cu_s->str,
                                  means_params, NULL));

  // Step 2: Compute covariance matrix (25x25 threads)
  int cov_grid = DIV_ROUND_UP(elements_in_block, BLOCK_DIM);
  void *cov_params[] = {&frame_ptr,  &means_ptr,   &cov_ptr,
                        &op_w,       &op_h,        &stride_px,
                        &block_size, &submatrix_w, &submatrix_h};
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->compute_covariance_kernel, cov_grid,
                                  cov_grid, 1, BLOCK_DIM, BLOCK_DIM, 1, 0,
                                  cu_s->str, cov_params, NULL));

  // Step 3: Compute eigenvalues (single block kernel)
  void *eigen_params[] = {&cov_ptr, &eigen_ptr, &workspace_ptr,
                          &elements_in_block};
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->compute_eigenvalues_kernel, 1, 1, 1, 1,
                                  1, 1, 0, cu_s->str, eigen_params, NULL));

  // Step 4: Compute independent term
  int total_elements = elements_in_block * num_blocks;
  int indep_blocks = DIV_ROUND_UP(total_elements, 256);
  void *indep_params[] = {&frame_ptr, &indep_ptr,  &truncated_w, &truncated_h,
                          &stride_px, &block_size, &num_blocks,  &num_blocks_h};
  CHECK_CUDA(cu_f,
             cuLaunchKernel(cu_s->compute_independent_term_kernel, indep_blocks,
                            1, 1, 256, 1, 1, 0, cu_s->str, indep_params, NULL));

  // Step 5: Solve linear system KX = Y (single block kernel)
  void *solve_params[] = {&cov_ptr,       &indep_ptr,         &solution_ptr,
                          &workspace_ptr, &elements_in_block, &num_blocks};
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->solve_linear_system_kernel, 1, 1, 1, 1,
                                  1, 1, 0, cu_s->str, solve_params, NULL));

  // Step 6: Pointwise product and division Z = (X * Y) / elements_in_block
  int pw_blocks = DIV_ROUND_UP(total_elements, 256);
  void *pw_params[] = {&solution_ptr, &indep_ptr, &elements_in_block,
                       &num_blocks};
  CHECK_CUDA(cu_f,
             cuLaunchKernel(cu_s->pointwise_product_div_kernel, pw_blocks, 1, 1,
                            256, 1, 1, 0, cu_s->str, pw_params, NULL));

  // Step 7: Sum columns to get variances
  int sum_blocks = DIV_ROUND_UP(num_blocks, 256);
  void *sum_params[] = {&solution_ptr, &variance_ptr, &elements_in_block,
                        &num_blocks};
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->sum_columns_kernel, sum_blocks, 1, 1,
                                  256, 1, 1, 0, cu_s->str, sum_params, NULL));

  // Step 8: Compute entropy for each block
  int ent_blocks = DIV_ROUND_UP(num_blocks, 256);
  void *ent_params[] = {&variance_ptr, &eigen_ptr,         &entropy_ptr,
                        &sigma_nn,     &elements_in_block, &num_blocks_h,
                        &num_blocks_v};
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->update_entropy_kernel, ent_blocks, 1, 1,
                                  256, 1, 1, 0, cu_s->str, ent_params, NULL));

  return 0;
}

/**
 * Compute final SpEED score from ref and dis results
 * Writes the unfinalized sum to score_host[score_slot] asynchronously.
 * The finalization (divide by num_blocks) is done in the host callback.
 */
static int compute_speed_score_async(VmafFeatureExtractor *fex,
                                     SpeedCudaState *cu_s, SpeedDimensions *dim,
                                     float sigma_nn, float nn_floor,
                                     int weight_var_mode, int score_slot) {
  CudaFunctions *cu_f = fex->cu_state->f;
  SpeedCudaBuffers *bufs = &cu_s->buffers;

  int num_blocks = dim->num_blocks;
  int elements_in_block = dim->elements_in_block;

  CUdeviceptr ref_ent_ptr, ref_var_ptr, dis_ent_ptr, dis_var_ptr;
  vmaf_cuda_buffer_get_dptr(bufs->entropies_ref, &ref_ent_ptr);
  vmaf_cuda_buffer_get_dptr(bufs->variances_ref, &ref_var_ptr);
  vmaf_cuda_buffer_get_dptr(bufs->entropies_dis, &dis_ent_ptr);
  vmaf_cuda_buffer_get_dptr(bufs->variances_dis, &dis_var_ptr);

  // Zero the score location in workspace
  float zero = 0.0f;
  CUdeviceptr workspace_ptr;
  vmaf_cuda_buffer_get_dptr(bufs->cusolver_workspace, &workspace_ptr);
  CUdeviceptr score_ptr = workspace_ptr; // Use first float in workspace as temp
  CHECK_CUDA(cu_f,
             cuMemcpyHtoDAsync(score_ptr, &zero, sizeof(float), cu_s->str));

  // Launch score kernel with reduction
  int score_blocks = DIV_ROUND_UP(num_blocks, 256);
  void *score_params[] = {&ref_ent_ptr, &ref_var_ptr,     &dis_ent_ptr,
                          &dis_var_ptr, &score_ptr,       &sigma_nn,
                          &nn_floor,    &weight_var_mode, &elements_in_block,
                          &num_blocks};
  CHECK_CUDA(cu_f,
             cuLaunchKernel(cu_s->compute_speed_score_kernel, score_blocks, 1,
                            1, 256, 1, 1, sizeof(float) * 32, cu_s->str,
                            score_params, NULL));

  // Copy unfinalized score to score_host[score_slot]
  // Finalization (divide by num_blocks) will be done in host callback
  CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(&bufs->score_host[score_slot], score_ptr,
                                     sizeof(float), cu_s->str));

  return 0;
}

/**
 * Extract SpEED score for a single channel (async version).
 * Writes the unfinalized score to score_host[score_slot].
 */
static int
extract_channel_cuda_async(VmafFeatureExtractor *fex, SpeedState *speed_state,
                           SpeedOptions *speed_options, VmafPicture *ref_pic,
                           VmafPicture *dis_pic, int channel, int score_slot) {
  SpeedCudaState *cu_s = speed_state->speed_cuda_state;
  SpeedDimensions *dim = &speed_state->dimensions;
  SpeedCudaBuffers *bufs = &cu_s->buffers;

  int stride_px = speed_state->float_stride / sizeof(float);
  float offset = -128.0f;
  float sigma_nn = (float)speed_options->speed_sigma_nn;

  // Preprocess ref
  launch_preprocess(fex, cu_s, ref_pic, bufs->frame_ref, channel,
                    dim->original_width, dim->original_height, stride_px,
                    offset);

  // Preprocess dis
  launch_preprocess(fex, cu_s, dis_pic, bufs->frame_dis, channel,
                    dim->original_width, dim->original_height, stride_px,
                    offset);

  // Process ref: filter, decimate, local mean subtraction
  process_frame(fex, cu_s, dim, bufs->frame_ref, bufs->frame_temp, stride_px);

  // Process dis: filter, decimate, local mean subtraction
  process_frame(fex, cu_s, dim, bufs->frame_dis, bufs->frame_temp, stride_px);

  // Compute SpEED parameters for ref
  compute_speed_params(fex, cu_s, dim, bufs->frame_ref, bufs->entropies_ref,
                       bufs->variances_ref, sigma_nn, stride_px);

  // Compute SpEED parameters for dis
  compute_speed_params(fex, cu_s, dim, bufs->frame_dis, bufs->entropies_dis,
                       bufs->variances_dis, sigma_nn, stride_px);

  // Compute final score (async, writes to score_host[score_slot])
  compute_speed_score_async(fex, cu_s, dim, sigma_nn,
                            (float)speed_options->speed_nn_floor,
                            speed_options->speed_weight_var_mode, score_slot);

  return 0;
}

// =============================================================================
// =============================================================================
//
//                         SPEED CHROMA CUDA
//
// =============================================================================
// =============================================================================

// Using unified SpeedChromaState from speed_helpers.h
// CPU buffers (cpu_frame_buffer_*) remain NULL, CUDA buffers are in
// speed_cuda_state

static int init_chroma(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                       unsigned bpc, unsigned w, unsigned h) {
  (void)bpc;

  // Adjust dimensions for chroma subsampling
  switch (pix_fmt) {
  case VMAF_PIX_FMT_UNKNOWN:
  case VMAF_PIX_FMT_YUV400P:
    return -EINVAL;
  case VMAF_PIX_FMT_YUV420P:
    w /= 2;
    h /= 2;
    break;
  case VMAF_PIX_FMT_YUV422P:
    w /= 2;
    break;
  case VMAF_PIX_FMT_YUV444P:
    break;
  }

  SpeedChromaState *s = fex->priv;
  CudaFunctions *cu_f = fex->cu_state->f;
  int err = 0;

  // Initialize SpeedOptions from feature state
  s->speed_options = (SpeedOptions){
      .speed_kernelscale = s->speed_chroma_kernelscale,
      .speed_prescale = s->speed_chroma_prescale,
      .speed_prescale_method = s->speed_chroma_prescale_method,
      .speed_sigma_nn = s->speed_chroma_sigma_nn,
      .speed_nn_floor = s->speed_chroma_nn_floor,
      .speed_weight_var_mode = s->speed_weight_var_mode,
  };

  // Initialize dimensions
  err = speed_init_dimensions(&s->speed_state.dimensions, w, h,
                              s->speed_options.speed_prescale);
  if (err) {
    vmaf_log(VMAF_LOG_LEVEL_ERROR, "speed_chroma_cuda: image too small");
    return err;
  }

  SpeedDimensions *dim = &s->speed_state.dimensions;

  // Validate options
  if (!speed_validate_kernelscale(s->speed_options.speed_kernelscale)) {
    vmaf_log(VMAF_LOG_LEVEL_ERROR, "speed_chroma_cuda: invalid kernelscale");
    return -EINVAL;
  }

  SpeedScalingMethod scaling_method;
  if (speed_get_scaling_method(s->speed_options.speed_prescale_method,
                               &scaling_method)) {
    vmaf_log(VMAF_LOG_LEVEL_ERROR,
             "speed_chroma_cuda: invalid prescale_method");
    return -EINVAL;
  }

  // Allocate CUDA state
  s->speed_state.speed_cuda_state = calloc(1, sizeof(SpeedCudaState));
  if (!s->speed_state.speed_cuda_state)
    return -ENOMEM;

  SpeedCudaState *cu_s = s->speed_state.speed_cuda_state;

  // Create streams and events
  CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
  CHECK_CUDA(cu_f,
             cuStreamCreateWithPriority(&cu_s->str, CU_STREAM_NON_BLOCKING, 0));
  CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&cu_s->host_stream,
                                              CU_STREAM_NON_BLOCKING, 0));
  CHECK_CUDA(cu_f, cuEventCreate(&cu_s->event, CU_EVENT_DEFAULT));
  CHECK_CUDA(cu_f, cuEventCreate(&cu_s->finished, CU_EVENT_DEFAULT));
  CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

  // Load kernels
  err = load_cuda_kernels(cu_s, cu_f, fex->cu_state->ctx);
  if (err)
    return err;

  // Calculate stride and allocate buffers
  s->speed_state.float_stride = ALIGN_CEIL(dim->alloc_width * sizeof(float));
  err = alloc_common_cuda_buffers(fex, cu_s, dim, s->speed_state.float_stride);
  if (err)
    return err;

  // Upload filter coefficients to GPU
  err = upload_filter_coefficients(fex, cu_s,
                                   (float)s->speed_options.speed_kernelscale);
  if (err)
    return err;

  // Allocate write score parameters
  cu_s->write_score_parameters = malloc(sizeof(WriteScoreParametersChroma));
  if (!cu_s->write_score_parameters)
    return -ENOMEM;

  // Initialize feature name dictionary
  s->feature_name_dict = vmaf_feature_name_dict_from_provided_features(
      fex->provided_features, fex->options, s);
  if (!s->feature_name_dict)
    return -ENOMEM;

  return 0;
}

static int extract_chroma(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                          VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                          VmafPicture *dist_pic_90, unsigned index,
                          VmafFeatureCollector *feature_collector) {
  (void)ref_pic_90;
  (void)dist_pic_90;

  SpeedChromaState *s = fex->priv;
  CudaFunctions *cu_f = fex->cu_state->f;
  SpeedCudaState *cu_s = s->speed_state.speed_cuda_state;
  SpeedDimensions *dim = &s->speed_state.dimensions;

  CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));

  // Wait for previous frame's host callback to complete
  CHECK_CUDA(cu_f, cuStreamSynchronize(cu_s->host_stream));

  // Process U channel (channel 1) -> score_host[0]
  extract_channel_cuda_async(fex, &s->speed_state, &s->speed_options, ref_pic,
                             dist_pic, 1, 0);

  // Process V channel (channel 2) -> score_host[1]
  extract_channel_cuda_async(fex, &s->speed_state, &s->speed_options, ref_pic,
                             dist_pic, 2, 1);

  // Record event when GPU work is done
  CHECK_CUDA(cu_f, cuEventRecord(cu_s->finished, cu_s->str));

  // Make host_stream wait for GPU work to complete
  CHECK_CUDA(cu_f, cuStreamWaitEvent(cu_s->host_stream, cu_s->finished,
                                     CU_EVENT_WAIT_DEFAULT));

  // Prepare parameters for async score write
  WriteScoreParametersChroma *params =
      (WriteScoreParametersChroma *)cu_s->write_score_parameters;
  params->feature_collector = feature_collector;
  params->feature_name_dict = s->feature_name_dict;
  params->buffers = &cu_s->buffers;
  params->index = index;
  params->num_blocks = dim->num_blocks;
  params->max_val = (float)s->speed_chroma_max_val;

  // Launch async host function to write scores
  CHECK_CUDA(cu_f, cuLaunchHostFunc(cu_s->host_stream,
                                    (CUhostFn*)write_scores_chroma,
                                    cu_s->write_score_parameters));

  CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));
  return 0;
}

static int flush_chroma(VmafFeatureExtractor *fex,
                        VmafFeatureCollector *feature_collector) {
  (void)feature_collector;

  SpeedChromaState *s = fex->priv;
  SpeedCudaState *cu_s = s->speed_state.speed_cuda_state;
  CudaFunctions *cu_f = fex->cu_state->f;

  CHECK_CUDA(cu_f, cuStreamSynchronize(cu_s->str));
  CHECK_CUDA(cu_f, cuStreamSynchronize(cu_s->host_stream));

  return 1;
}

static int close_chroma(VmafFeatureExtractor *fex) {
  SpeedChromaState *s = fex->priv;

  free_cuda_buffers(fex->cu_state, s->speed_state.speed_cuda_state);
  s->speed_state.speed_cuda_state = NULL;

  if (s->feature_name_dict)
    vmaf_dictionary_free(&s->feature_name_dict);

  return 0;
}

static const VmafOption options_chroma[] = {
    SPEED_OPTION_KERNELSCALE(SpeedChromaState, speed_chroma_),
    SPEED_OPTION_PRESCALE(SpeedChromaState, speed_chroma_),
    SPEED_OPTION_PRESCALE_METHOD(SpeedChromaState, speed_chroma_),
    SPEED_OPTION_SIGMA_NN(SpeedChromaState, speed_chroma_),
    SPEED_OPTION_NN_FLOOR(SpeedChromaState, speed_chroma_),
    SPEED_OPTION_MAX_VAL(SpeedChromaState, speed_chroma_),
    SPEED_OPTION_WEIGHT_VAR_MODE(SpeedChromaState, speed_weight_var_mode),
    {0}};

VmafFeatureExtractor vmaf_fex_speed_chroma_cuda = {
    .name = "speed_chroma_cuda",
    .init = init_chroma,
    .extract = extract_chroma,
    .flush = flush_chroma,
    .close = close_chroma,
    .options = options_chroma,
    .priv_size = sizeof(SpeedChromaState),
    .provided_features = provided_features_chroma,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
};

// =============================================================================
// =============================================================================
//
//                         SPEED TEMPORAL CUDA
//
// =============================================================================
// =============================================================================

// Using unified SpeedTemporalState from speed_helpers.h
// CPU buffers (cpu_frame_buffer_*) remain NULL, CUDA buffers are in
// speed_cuda_state

static int init_temporal(VmafFeatureExtractor *fex,
                         enum VmafPixelFormat pix_fmt, unsigned bpc, unsigned w,
                         unsigned h) {
  (void)pix_fmt;
  (void)bpc;

  SpeedTemporalState *s = fex->priv;
  CudaFunctions *cu_f = fex->cu_state->f;
  int err = 0;

  // Initialize SpeedOptions from feature state
  s->speed_options = (SpeedOptions){
      .speed_kernelscale = s->speed_temporal_kernelscale,
      .speed_prescale = s->speed_temporal_prescale,
      .speed_prescale_method = s->speed_temporal_prescale_method,
      .speed_sigma_nn = s->speed_temporal_sigma_nn,
      .speed_nn_floor = s->speed_temporal_nn_floor,
      .speed_weight_var_mode = 0, // Not used for temporal
  };

  // Initialize dimensions
  err = speed_init_dimensions(&s->speed_state.dimensions, w, h,
                              s->speed_options.speed_prescale);
  if (err) {
    vmaf_log(VMAF_LOG_LEVEL_ERROR, "speed_temporal_cuda: image too small");
    return err;
  }

  SpeedDimensions *dim = &s->speed_state.dimensions;

  // Validate options
  if (!speed_validate_kernelscale(s->speed_options.speed_kernelscale)) {
    vmaf_log(VMAF_LOG_LEVEL_ERROR, "speed_temporal_cuda: invalid kernelscale");
    return -EINVAL;
  }

  SpeedScalingMethod scaling_method;
  if (speed_get_scaling_method(s->speed_options.speed_prescale_method,
                               &scaling_method)) {
    vmaf_log(VMAF_LOG_LEVEL_ERROR,
             "speed_temporal_cuda: invalid prescale_method");
    return -EINVAL;
  }

  // Allocate CUDA state
  s->speed_state.speed_cuda_state = calloc(1, sizeof(SpeedCudaState));
  if (!s->speed_state.speed_cuda_state)
    return -ENOMEM;

  SpeedCudaState *cu_s = s->speed_state.speed_cuda_state;

  // Create streams and events
  CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
  CHECK_CUDA(cu_f,
             cuStreamCreateWithPriority(&cu_s->str, CU_STREAM_NON_BLOCKING, 0));
  CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&cu_s->host_stream,
                                              CU_STREAM_NON_BLOCKING, 0));
  CHECK_CUDA(cu_f, cuEventCreate(&cu_s->event, CU_EVENT_DEFAULT));
  CHECK_CUDA(cu_f, cuEventCreate(&cu_s->finished, CU_EVENT_DEFAULT));
  CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

  // Load kernels
  err = load_cuda_kernels(cu_s, cu_f, fex->cu_state->ctx);
  if (err)
    return err;

  // Calculate stride and allocate buffers
  s->speed_state.float_stride = ALIGN_CEIL(dim->alloc_width * sizeof(float));
  size_t frame_size = s->speed_state.float_stride * dim->alloc_height;

  err = alloc_common_cuda_buffers(fex, cu_s, dim, s->speed_state.float_stride);
  if (err)
    return err;

  // Allocate additional buffers for temporal (previous frames)
  SpeedCudaBuffers *bufs = &cu_s->buffers;
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->frame_ref_prev, frame_size);
  err |= alloc_cuda_buffer(fex->cu_state, &bufs->frame_dis_prev, frame_size);
  if (err)
    return err;

  // Upload filter coefficients to GPU
  err = upload_filter_coefficients(fex, cu_s,
                                   (float)s->speed_options.speed_kernelscale);
  if (err)
    return err;

  // Initialize frame index
  s->frame_index = 0;

  // Allocate write score parameters
  cu_s->write_score_parameters = malloc(sizeof(WriteScoreParametersTemporal));
  if (!cu_s->write_score_parameters)
    return -ENOMEM;

  // Initialize feature name dictionary
  s->feature_name_dict = vmaf_feature_name_dict_from_provided_features(
      fex->provided_features, fex->options, s);
  if (!s->feature_name_dict)
    return -ENOMEM;

  return 0;
}

static int extract_temporal(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                            VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                            VmafPicture *dist_pic_90, unsigned index,
                            VmafFeatureCollector *feature_collector) {
  (void)ref_pic_90;
  (void)dist_pic_90;

  SpeedTemporalState *s = fex->priv;
  CudaFunctions *cu_f = fex->cu_state->f;
  SpeedCudaState *cu_s = s->speed_state.speed_cuda_state;
  SpeedCudaBuffers *bufs = &cu_s->buffers;
  SpeedDimensions *dim = &s->speed_state.dimensions;

  CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));

  // Wait for previous frame's host callback to complete
  CHECK_CUDA(cu_f, cuStreamSynchronize(cu_s->host_stream));

  int stride_px = s->speed_state.float_stride / sizeof(float);
  float offset = -128.0f;

  // Use cyclic buffer pattern: even frames use frame_ref/dis, odd use
  // frame_ref_prev/dis_prev
  int cyclic_index = s->frame_index % 2;
  VmafCudaBuffer *curr_ref =
      cyclic_index ? bufs->frame_ref_prev : bufs->frame_ref;
  VmafCudaBuffer *curr_dis =
      cyclic_index ? bufs->frame_dis_prev : bufs->frame_dis;
  VmafCudaBuffer *prev_ref =
      cyclic_index ? bufs->frame_ref : bufs->frame_ref_prev;
  VmafCudaBuffer *prev_dis =
      cyclic_index ? bufs->frame_dis : bufs->frame_dis_prev;

  // Preprocess current ref frame to curr_ref
  launch_preprocess(fex, cu_s, ref_pic, curr_ref, 0, dim->original_width,
                    dim->original_height, stride_px, offset);

  // Preprocess current dis frame to curr_dis
  launch_preprocess(fex, cu_s, dist_pic, curr_dis, 0, dim->original_width,
                    dim->original_height, stride_px, offset);

  // For frame 0, just store the frames and output 0
  if (s->frame_index == 0) {
    int err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict,
        "Speed_temporal_feature_speed_temporal_score", 0.0, index);
    s->frame_index++;
    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));
    return err;
  }

  // Compute frame differences: prev = prev - curr
  // This computes the temporal difference (previous frame - current frame)
  int w = dim->original_width;
  int h = dim->original_height;

  CUdeviceptr prev_ref_ptr, prev_dis_ptr, curr_ref_ptr, curr_dis_ptr;
  vmaf_cuda_buffer_get_dptr(prev_ref, &prev_ref_ptr);
  vmaf_cuda_buffer_get_dptr(prev_dis, &prev_dis_ptr);
  vmaf_cuda_buffer_get_dptr(curr_ref, &curr_ref_ptr);
  vmaf_cuda_buffer_get_dptr(curr_dis, &curr_dis_ptr);

  int grid_x = DIV_ROUND_UP(w, BLOCK_DIM);
  int grid_y = DIV_ROUND_UP(h, BLOCK_DIM);

  // ref_diff = prev_ref - curr_ref
  void *ref_sub_params[] = {&prev_ref_ptr, &curr_ref_ptr, &w, &h, &stride_px};
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->subtract_image_kernel, grid_x, grid_y,
                                  1, BLOCK_DIM, BLOCK_DIM, 1, 0, cu_s->str,
                                  ref_sub_params, NULL));

  // dis_diff: if use_ref_diff, dis_diff = prev_dis - curr_ref, else dis_diff =
  // prev_dis - curr_dis
  CUdeviceptr dis_sub_src =
      s->speed_temporal_use_ref_diff ? curr_ref_ptr : curr_dis_ptr;
  void *dis_sub_params[] = {&prev_dis_ptr, &dis_sub_src, &w, &h, &stride_px};
  CHECK_CUDA(cu_f, cuLaunchKernel(cu_s->subtract_image_kernel, grid_x, grid_y,
                                  1, BLOCK_DIM, BLOCK_DIM, 1, 0, cu_s->str,
                                  dis_sub_params, NULL));

  // Process ref diff: filter, decimate, local mean subtraction
  // Use prev_ref (which now contains the difference) and frame_temp as temp
  process_frame(fex, cu_s, dim, prev_ref, bufs->frame_temp, stride_px);

  // Process dis diff
  process_frame(fex, cu_s, dim, prev_dis, bufs->frame_temp, stride_px);

  // Compute SpEED parameters for ref diff
  float sigma_nn = (float)s->speed_options.speed_sigma_nn;
  compute_speed_params(fex, cu_s, dim, prev_ref, bufs->entropies_ref,
                       bufs->variances_ref, sigma_nn, stride_px);

  // Compute SpEED parameters for dis diff
  compute_speed_params(fex, cu_s, dim, prev_dis, bufs->entropies_dis,
                       bufs->variances_dis, sigma_nn, stride_px);

  // Compute final score (async, writes to score_host[0])
  compute_speed_score_async(
      fex, cu_s, dim, sigma_nn, (float)s->speed_options.speed_nn_floor, 0,
      0); // weight_var_mode = 0 for temporal, score_slot = 0

  // Record event when GPU work is done
  CHECK_CUDA(cu_f, cuEventRecord(cu_s->finished, cu_s->str));

  // Make host_stream wait for GPU work to complete
  CHECK_CUDA(cu_f, cuStreamWaitEvent(cu_s->host_stream, cu_s->finished,
                                     CU_EVENT_WAIT_DEFAULT));

  // Prepare parameters for async score write
  WriteScoreParametersTemporal *params =
      (WriteScoreParametersTemporal *)cu_s->write_score_parameters;
  params->feature_collector = feature_collector;
  params->feature_name_dict = s->feature_name_dict;
  params->buffers = &cu_s->buffers;
  params->index = index;
  params->num_blocks = dim->num_blocks;
  params->max_val = (float)s->speed_temporal_max_val;

  // Launch async host function to write scores
  CHECK_CUDA(cu_f, cuLaunchHostFunc(cu_s->host_stream,
                                    (CUhostFn*)write_scores_temporal,
                                    cu_s->write_score_parameters));

  s->frame_index++;
  CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));
  return 0;
}

static int flush_temporal(VmafFeatureExtractor *fex,
                          VmafFeatureCollector *feature_collector) {
  (void)feature_collector;

  SpeedTemporalState *s = fex->priv;
  SpeedCudaState *cu_s = s->speed_state.speed_cuda_state;
  CudaFunctions *cu_f = fex->cu_state->f;

  CHECK_CUDA(cu_f, cuStreamSynchronize(cu_s->str));
  CHECK_CUDA(cu_f, cuStreamSynchronize(cu_s->host_stream));

  return 1;
}

static int close_temporal(VmafFeatureExtractor *fex) {
  SpeedTemporalState *s = fex->priv;

  free_cuda_buffers(fex->cu_state, s->speed_state.speed_cuda_state);
  s->speed_state.speed_cuda_state = NULL;

  if (s->feature_name_dict)
    vmaf_dictionary_free(&s->feature_name_dict);

  return 0;
}

static const VmafOption options_temporal[] = {
    SPEED_OPTION_KERNELSCALE(SpeedTemporalState, speed_temporal_),
    SPEED_OPTION_PRESCALE(SpeedTemporalState, speed_temporal_),
    SPEED_OPTION_PRESCALE_METHOD(SpeedTemporalState, speed_temporal_),
    SPEED_OPTION_SIGMA_NN(SpeedTemporalState, speed_temporal_),
    SPEED_OPTION_NN_FLOOR(SpeedTemporalState, speed_temporal_),
    SPEED_OPTION_MAX_VAL(SpeedTemporalState, speed_temporal_),
    SPEED_OPTION_USE_REF_DIFF(SpeedTemporalState, speed_temporal_use_ref_diff),
    {0}};

VmafFeatureExtractor vmaf_fex_speed_temporal_cuda = {
    .name = "speed_temporal_cuda",
    .init = init_temporal,
    .extract = extract_temporal,
    .flush = flush_temporal,
    .close = close_temporal,
    .options = options_temporal,
    .priv_size = sizeof(SpeedTemporalState),
    .provided_features = provided_features_temporal,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA | VMAF_FEATURE_EXTRACTOR_TEMPORAL,
};
