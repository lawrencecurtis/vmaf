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

// CUDA kernels for SpEED feature extractor
// Compiled to PTX only - all kernel launches are done from speed_cuda.c

#include <cuda_runtime.h>
#include <stdint.h>

// =============================================================================
// Constants
// =============================================================================

#define SPEED_BLOCK_SIZE 5
#define SPEED_ELEMENTS_IN_BLOCK 25
#define SPEED_NUM_SCALES 4
#define SPEED_EIGENVALUE_EPS 1e-6f
#define SPEED_EIGENVALUE_MAX_ITERS 500

// Thread block dimensions for 2D kernels
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// =============================================================================
// Helper Device Functions
// =============================================================================

/**
 * Reflect index at boundaries (mirror boundary condition)
 * Formula matches CPU: 2 * max - idx - 1
 */
__device__ __forceinline__ int reflect_index(int idx, int max_val) {
    if (idx < 0) return -idx;
    if (idx >= max_val) return 2 * max_val - idx - 1;  // Fix: was -2, should be -1
    return idx;
}

/**
 * Warp-level sum reduction for floats
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


/**
 * Block-level sum reduction using shared memory
 */
__device__ float block_reduce_sum(float val, float* shared_data) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane == 0) {
        shared_data[wid] = val;
    }
    __syncthreads();

    // Final reduction by first warp
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? shared_data[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }

    return val;
}

// =============================================================================
// Kernel 1: Preprocessing Kernels (templated)
// =============================================================================

/**
 * Convert uint8 input to float with offset (no prescale)
 */
extern "C" __global__ void preprocess_kernel_u8(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int width, int height, int in_stride, int out_stride,
    int bpc, float offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float val = static_cast<float>(input[y * in_stride + x]);
    output[y * out_stride + x] = val + offset;
}

/**
 * Convert uint16 input to float with offset (no prescale)
 */
extern "C" __global__ void preprocess_kernel_u16(
    const uint16_t* __restrict__ input,
    float* __restrict__ output,
    int width, int height, int in_stride, int out_stride,
    int bpc, float offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // in_stride is in bytes, convert to elements
    int in_stride_elem = in_stride / sizeof(uint16_t);
    float val = static_cast<float>(input[y * in_stride_elem + x]);
    output[y * out_stride + x] = val + offset;
}

/**
 * Prescale uint8 input with nearest-neighbor interpolation and convert to float
 */
extern "C" __global__ void preprocess_prescale_kernel_u8(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int in_w, int in_h, int in_stride,
    int out_w, int out_h, int out_stride,
    int bpc, float offset)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= out_w || oy >= out_h) return;

    // Nearest neighbor sampling
    float scale_x = (float)out_w / in_w;
    float scale_y = (float)out_h / in_h;
    float fx = (ox + 0.5f) / scale_x - 0.5f;
    float fy = (oy + 0.5f) / scale_y - 0.5f;

    int ix = min(max((int)(fx + 0.5f), 0), in_w - 1);
    int iy = min(max((int)(fy + 0.5f), 0), in_h - 1);

    float val = static_cast<float>(input[iy * in_stride + ix]);
    output[oy * out_stride + ox] = val + offset;
}

/**
 * Prescale uint16 input with nearest-neighbor interpolation and convert to float
 */
extern "C" __global__ void preprocess_prescale_kernel_u16(
    const uint16_t* __restrict__ input,
    float* __restrict__ output,
    int in_w, int in_h, int in_stride,
    int out_w, int out_h, int out_stride,
    int bpc, float offset)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= out_w || oy >= out_h) return;

    // in_stride is in bytes, convert to elements
    int in_stride_elem = in_stride / sizeof(uint16_t);

    // Nearest neighbor sampling
    float scale_x = (float)out_w / in_w;
    float scale_y = (float)out_h / in_h;
    float fx = (ox + 0.5f) / scale_x - 0.5f;
    float fy = (oy + 0.5f) / scale_y - 0.5f;

    int ix = min(max((int)(fx + 0.5f), 0), in_w - 1);
    int iy = min(max((int)(fy + 0.5f), 0), in_h - 1);

    float val = static_cast<float>(input[iy * in_stride_elem + ix]);
    output[oy * out_stride + ox] = val + offset;
}

// =============================================================================
// Kernel 2: Gaussian Filter (Separable) - Float input only
// The input is already converted to float by preprocess kernels
// =============================================================================

/**
 * Separable Gaussian filter - Vertical pass (float input)
 * Writes to output buffer with same stride as input.
 */
extern "C" __global__ void filter_vertical_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ filter,
    int width, int height, int stride,
    int filter_width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_filter = filter_width / 2;
    float sum = 0.0f;

    for (int fy = 0; fy < filter_width; fy++) {
        int sy = y - half_filter + fy;
        // Mirror boundary condition
        sy = reflect_index(sy, height);
        sum += input[sy * stride + x] * filter[fy];
    }

    output[y * stride + x] = sum;
}

/**
 * Separable Gaussian filter - Horizontal pass (float input)
 * Input and output use the same stride.
 */
extern "C" __global__ void filter_horizontal_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ filter,
    int width, int height, int stride,
    int filter_width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_filter = filter_width / 2;
    float sum = 0.0f;

    for (int fx = 0; fx < filter_width; fx++) {
        int sx = x - half_filter + fx;
        // Mirror boundary condition
        sx = reflect_index(sx, width);
        sum += input[y * stride + sx] * filter[fx];
    }

    output[y * stride + x] = sum;
}

// =============================================================================
// Kernel 3: Decimation (16x)
// =============================================================================

extern "C" __global__ void decimate_16x_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_width, int in_height, int in_stride,
    int out_width, int out_height, int out_stride)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= out_width || oy >= out_height) return;

    // Take every 16th pixel (for SPEED_NUM_SCALES = 4, decimate = 2^4 = 16)
    int ix = ox << SPEED_NUM_SCALES;  // ox * 16
    int iy = oy << SPEED_NUM_SCALES;  // oy * 16

    output[oy * out_stride + ox] = input[iy * in_stride + ix];
}

// =============================================================================
// Kernel 4: Image Subtraction (Local Mean Subtraction)
// =============================================================================

extern "C" __global__ void subtract_image_kernel(
    float* __restrict__ image_a,
    const float* __restrict__ image_b,
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * stride + x;
        image_a[idx] -= image_b[idx];
    }
}

// =============================================================================
// Kernel 5: Block Means Computation
// =============================================================================

/**
 * Compute mean for each of the 25 block offset positions.
 * Each thread computes one mean for position (start_row, start_col).
 */
extern "C" __global__ void compute_block_means_kernel(
    const float* __restrict__ image,
    float* __restrict__ means,
    int width, int height, int stride,
    int block_size, int submatrix_w, int submatrix_h)
{
    int start_row = threadIdx.y;
    int start_col = threadIdx.x;

    if (start_row >= block_size || start_col >= block_size) return;

    double sum = 0.0;

    for (int i = 0; i < submatrix_h; i++) {
        for (int j = 0; j < submatrix_w; j++) {
            sum += image[(start_row + i) * stride + (start_col + j)];
        }
    }

    means[start_row * block_size + start_col] =
        (float)(sum / (submatrix_w * submatrix_h));
}

// =============================================================================
// Kernel 6: Covariance Matrix Computation
// =============================================================================

/**
 * Compute the 25x25 covariance matrix.
 * Each thread computes one entry cov[x_idx][y_idx].
 * Only computes upper triangle + diagonal, then mirrors.
 */
extern "C" __global__ void compute_covariance_kernel(
    const float* __restrict__ image,
    const float* __restrict__ means,
    float* __restrict__ cov_matrix,
    int width, int height, int stride,
    int block_size, int submatrix_w, int submatrix_h)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    int elements_in_block = block_size * block_size;

    if (x_idx >= elements_in_block || y_idx > x_idx) return;

    int start_row_x = x_idx / block_size;
    int start_col_x = x_idx % block_size;
    int start_row_y = y_idx / block_size;
    int start_col_y = y_idx % block_size;

    double mean_x = means[start_row_x * block_size + start_col_x];
    double mean_y = means[start_row_y * block_size + start_col_y];

    double cov = 0.0;

    for (int i = 0; i < submatrix_h; i++) {
        for (int j = 0; j < submatrix_w; j++) {
            double val_x = image[(start_row_x + i) * stride + (start_col_x + j)];
            double val_y = image[(start_row_y + i) * stride + (start_col_y + j)];
            cov += (val_x - mean_x) * (val_y - mean_y);
        }
    }

    float covariance = (float)(cov / (submatrix_w * submatrix_h));

    // Store in both positions (symmetric matrix)
    cov_matrix[x_idx * elements_in_block + y_idx] = covariance;
    cov_matrix[y_idx * elements_in_block + x_idx] = covariance;
}

// =============================================================================
// Kernel 7: Eigenvalue Computation (Tridiagonal QR Iteration)
// =============================================================================

/**
 * Single-block kernel to compute eigenvalues of a 25x25 symmetric matrix.
 * Uses Householder tridiagonalization followed by QR iteration.
 *
 * This is a single-threaded kernel since eigenvalue computation is inherently
 * sequential. For 25x25 matrices, this is fast enough.
 */
extern "C" __global__ void compute_eigenvalues_kernel(
    const float* __restrict__ cov_matrix,
    float* __restrict__ eigenvalues,
    float* __restrict__ workspace,
    int matrix_size)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // Workspace layout:
    // A: matrix_size * matrix_size
    // d: matrix_size (diagonal)
    // sd: matrix_size (subdiagonal)
    // v: matrix_size (temp vector)
    // x: matrix_size (temp vector)
    float* A = workspace;
    float* d = A + matrix_size * matrix_size;
    float* sd = d + matrix_size;
    float* v = sd + matrix_size;
    float* x = v + matrix_size;

    // Copy matrix to workspace
    for (int i = 0; i < matrix_size * matrix_size; i++) {
        A[i] = cov_matrix[i];
    }

    // Handle special case
    if (matrix_size == 1) {
        eigenvalues[0] = A[0];
        return;
    }

    // =========================================================================
    // Step 1: Tridiagonalization using Householder transformations
    // =========================================================================

    for (int i = 0; i < matrix_size - 2; i++) {
        // Compute column norm below diagonal
        float xnorm_sq = 0.0f;
        for (int j = i + 2; j < matrix_size; j++) {
            float val = A[j * matrix_size + i];
            xnorm_sq += val * val;
        }
        float xnorm = sqrtf(xnorm_sq);

        if (xnorm < SPEED_EIGENVALUE_EPS) {
            continue;  // Skip if already zero
        }

        float alpha = A[(i + 1) * matrix_size + i];
        float sign = (alpha >= 0.0f) ? 1.0f : -1.0f;
        float beta = -sign * sqrtf(alpha * alpha + xnorm_sq);
        float tau = (beta - alpha) / beta;
        float s = alpha - beta;

        if (fabsf(s) > SPEED_EIGENVALUE_EPS) {
            // Scale the column
            for (int j = i + 1; j < matrix_size; j++) {
                A[j * matrix_size + i] /= s;
            }
            A[(i + 1) * matrix_size + i] = beta;
        }

        if (tau == 0.0f) continue;

        // Copy column into v
        for (int j = i + 1; j < matrix_size; j++) {
            v[j] = A[j * matrix_size + i];
        }
        v[i + 1] = 1.0f;

        // x = tau * A * v
        for (int j = i + 1; j < matrix_size; j++) {
            x[j] = 0.0f;
            for (int k = i + 1; k < matrix_size; k++) {
                x[j] += tau * A[j * matrix_size + k] * v[k];
            }
        }

        // Compute x = x - 0.5 * tau * (x.v) * v
        float xv = 0.0f;
        for (int j = i + 1; j < matrix_size; j++) {
            xv += x[j] * v[j];
        }
        float alpha2 = -0.5f * tau * xv;
        for (int j = i + 1; j < matrix_size; j++) {
            x[j] += alpha2 * v[j];
        }

        // A = A - v * x' - x * v'
        for (int j = i + 1; j < matrix_size; j++) {
            for (int k = i + 1; k < matrix_size; k++) {
                A[j * matrix_size + k] -= x[j] * v[k] + v[j] * x[k];
            }
        }
    }

    // Extract diagonal and subdiagonal
    for (int i = 0; i < matrix_size; i++) {
        d[i] = A[i * matrix_size + i];
    }
    for (int i = 0; i < matrix_size - 1; i++) {
        sd[i] = A[(i + 1) * matrix_size + i];
    }

    // =========================================================================
    // Step 2: QR iteration on tridiagonal matrix
    // =========================================================================

    // Chop small elements
    for (int i = 0; i < matrix_size - 1; i++) {
        if (fabsf(sd[i]) < SPEED_EIGENVALUE_EPS * (fabsf(d[i]) + fabsf(d[i + 1]))) {
            sd[i] = 0.0f;
        }
    }

    // QR iteration
    int b = matrix_size - 1;
    int iter = 0;

    while (b > 0 && iter < SPEED_EIGENVALUE_MAX_ITERS) {
        if (sd[b - 1] == 0.0f) {
            b--;
            continue;
        }

        // Find largest unreduced block
        int a = b - 1;
        while (a > 0 && sd[a - 1] != 0.0f) {
            a--;
        }

        int n_block = b - a + 1;
        float* d_block = d + a;
        float* sd_block = sd + a;

        // Implicit QR step with Wilkinson shift
        float ta = d_block[n_block - 2];
        float tb = d_block[n_block - 1];
        float tab = sd_block[n_block - 2];
        float dt = (ta - tb) / 2.0f;
        float mu;

        if (dt > 0) {
            mu = tb - tab * (tab / (dt + sqrtf(dt * dt + tab * tab)));
        } else if (dt == 0) {
            mu = tb - fabsf(tab);
        } else {
            mu = tb + tab * (tab / (-dt + sqrtf(dt * dt + tab * tab)));
        }

        // Implicit QR step - chase the bulge
        float x_val = d_block[0] - mu;
        float z_val = sd_block[0];

        // Track previous iteration values for bulge chasing
        float ak = 0.0f, bk = 0.0f, zk = 0.0f;
        float ap = d_block[0];
        float bp = sd_block[0];
        float aq = d_block[1];
        float bq = (n_block > 2) ? sd_block[1] : 0.0f;

        for (int k = 0; k < n_block - 1; k++) {
            // Givens rotation to zero out z_val
            float c, s;
            if (z_val == 0.0f) {
                c = 1.0f;
                s = 0.0f;
            } else if (fabsf(z_val) > fabsf(x_val)) {
                float t = -x_val / z_val;
                s = 1.0f / sqrtf(1.0f + t * t);
                c = s * t;
            } else {
                float t = -z_val / x_val;
                c = 1.0f / sqrtf(1.0f + t * t);
                s = c * t;
            }

            // Compute update for sd[k-1] using previous iteration's values
            float bk1 = c * bk - s * zk;

            // Apply rotation to 2x2 block
            float ap1 = c * (c * ap - s * bp) + s * (s * aq - c * bp);
            float bp1 = c * (s * ap + c * bp) - s * (s * bp + c * aq);
            float zp1 = -s * bq;
            float aq1 = s * (s * ap + c * bp) + c * (s * bp + c * aq);
            float bq1 = c * bq;

            // Store results for next iteration
            ak = ap1;
            bk = bp1;
            zk = zp1;
            ap = aq1;
            bp = bq1;

            // Read next values
            if (k < n_block - 2) aq = d_block[k + 2];
            if (k < n_block - 3) bq = sd_block[k + 2];

            // Update matrix
            d_block[k] = ak;
            if (k > 0) sd_block[k - 1] = bk1;
            if (k < n_block - 2) sd_block[k + 1] = bp;

            x_val = bk;
            z_val = zk;
        }

        // Final updates
        d_block[n_block - 1] = ap;
        sd_block[n_block - 2] = bk;

        // Chop small elements in block
        for (int i = 0; i < n_block - 1; i++) {
            if (fabsf(sd_block[i]) < SPEED_EIGENVALUE_EPS *
                (fabsf(d_block[i]) + fabsf(d_block[i + 1]))) {
                sd_block[i] = 0.0f;
            }
        }

        iter++;
    }

    // Copy diagonal to eigenvalues
    for (int i = 0; i < matrix_size; i++) {
        eigenvalues[i] = d[i];
    }
}

// =============================================================================
// Kernel 8: Independent Term Computation
// =============================================================================

/**
 * Extract Y matrix from image for the linear system KX = Y.
 * Maps image pixels to their block positions.
 */
extern "C" __global__ void compute_independent_term_kernel(
    const float* __restrict__ image,
    float* __restrict__ independent_term,
    int truncated_width, int truncated_height, int stride,
    int block_size, int num_blocks, int num_blocks_h)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int elements_in_block = block_size * block_size;
    int total_elements = elements_in_block * num_blocks;

    if (tid >= total_elements) return;

    // Determine which element this is
    int out_row = tid / num_blocks;  // Which element in block (0-24)
    int out_col = tid % num_blocks;  // Which block (0 to num_blocks-1)

    int start_i = out_row / block_size;
    int start_j = out_row % block_size;

    int block_row = out_col / num_blocks_h;
    int block_col = out_col % num_blocks_h;

    int i = start_i + block_row * block_size;
    int j = start_j + block_col * block_size;

    if (i < truncated_height && j < truncated_width) {
        independent_term[out_row * num_blocks + out_col] = image[i * stride + j];
    }
}

// =============================================================================
// Kernel 9: Linear System Solve (QR Decomposition)
// =============================================================================

/**
 * Solve KX = Y using QR decomposition.
 * Single-block kernel for 25x25 matrix with multiple right-hand sides.
 *
 * For better performance on large num_blocks, this could be batched.
 */
extern "C" __global__ void solve_linear_system_kernel(
    const float* __restrict__ cov_matrix,
    const float* __restrict__ independent_term,
    float* __restrict__ solution,
    float* __restrict__ workspace,
    int matrix_size, int num_blocks)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // Workspace layout:
    // A: matrix_size * matrix_size
    // Q: matrix_size * matrix_size
    // R: matrix_size * matrix_size
    // tmp1: matrix_size * matrix_size
    // tmp2: matrix_size * matrix_size
    // vec: matrix_size
    // QtY: matrix_size * num_blocks
    float* A = workspace;
    float* Q = A + matrix_size * matrix_size;
    float* R = Q + matrix_size * matrix_size;
    float* tmp1 = R + matrix_size * matrix_size;
    float* tmp2 = tmp1 + matrix_size * matrix_size;
    float* vec = tmp2 + matrix_size * matrix_size;

    int size = matrix_size;

    // Copy cov_matrix to A
    for (int i = 0; i < size * size; i++) {
        A[i] = cov_matrix[i];
    }

    // =========================================================================
    // QR decomposition using Householder reflections
    // =========================================================================

    // Initialize Q as identity
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            Q[i * size + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Copy A to tmp2 (working copy)
    for (int i = 0; i < size * size; i++) {
        tmp2[i] = A[i];
    }

    for (int k = 0; k < size - 1; k++) {
        // Create minor matrix in tmp2
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i < k || j < k) {
                    tmp2[i * size + j] = (i == j) ? 1.0f : 0.0f;
                }
            }
        }

        // Extract k-th column into vec
        for (int i = 0; i < size; i++) {
            vec[i] = tmp2[i * size + k];
        }

        // Compute norm
        float norm = 0.0f;
        for (int i = 0; i < size; i++) {
            norm += vec[i] * vec[i];
        }
        norm = sqrtf(norm);

        // Modify vec - use ORIGINAL matrix A for sign (matches CPU implementation)
        float sign = (A[k * size + k] >= 0.0f) ? 1.0f : -1.0f;
        vec[k] += sign * norm;

        // Normalize vec
        float vec_norm = 0.0f;
        for (int i = 0; i < size; i++) {
            vec_norm += vec[i] * vec[i];
        }
        vec_norm = sqrtf(vec_norm);

        if (vec_norm > SPEED_EIGENVALUE_EPS) {
            for (int i = 0; i < size; i++) {
                vec[i] /= vec_norm;
            }
        }

        // Compute tmp_q = I - 2*v*v'
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                tmp1[i * size + j] = ((i == j) ? 1.0f : 0.0f) - 2.0f * vec[i] * vec[j];
            }
        }

        // tmp2 = tmp_q * tmp2
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0.0f;
                for (int p = 0; p < size; p++) {
                    sum += tmp1[i * size + p] * tmp2[p * size + j];
                }
                R[i * size + j] = sum;  // Use R as temp storage
            }
        }
        for (int i = 0; i < size * size; i++) {
            tmp2[i] = R[i];
        }

        // Q = tmp_q * Q
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0.0f;
                for (int p = 0; p < size; p++) {
                    sum += tmp1[i * size + p] * Q[p * size + j];
                }
                R[i * size + j] = sum;  // Use R as temp storage
            }
        }
        for (int i = 0; i < size * size; i++) {
            Q[i] = R[i];
        }
    }

    // R = Q * A
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int p = 0; p < size; p++) {
                sum += Q[i * size + p] * A[p * size + j];
            }
            R[i * size + j] = sum;
        }
    }

    // NOTE: CPU transposes Q twice (once in matrix_qr_decomposition, once in solve_linear_system)
    // which effectively leaves Q unchanged. So we skip the transpose here to match CPU behavior.
    // The Q matrix at this point is already in the correct form for solving.

    // =========================================================================
    // Solve R * X = Q * Y for each column (Q is NOT transposed to match CPU)
    // =========================================================================

    for (int col = 0; col < num_blocks; col++) {
        // Compute Q * Y[:, col] (Q is not transposed to match CPU's double-transpose behavior)
        float QtY[SPEED_ELEMENTS_IN_BLOCK];
        for (int i = 0; i < size; i++) {
            QtY[i] = 0.0f;
            for (int j = 0; j < size; j++) {
                QtY[i] += Q[i * size + j] * independent_term[j * num_blocks + col];
            }
        }

        // Back substitution: R * x = QtY
        for (int i = size - 1; i >= 0; i--) {
            float denom = R[i * size + i];
            if (fabsf(denom) < SPEED_EIGENVALUE_EPS) {
                solution[i * num_blocks + col] = 0.0f;
                continue;
            }

            float sum = QtY[i];
            for (int j = i + 1; j < size; j++) {
                sum -= R[i * size + j] * solution[j * num_blocks + col];
            }
            solution[i * num_blocks + col] = sum / denom;
        }
    }
}

// =============================================================================
// Kernel 10: Pointwise Product and Division
// =============================================================================

extern "C" __global__ void pointwise_product_div_kernel(
    float* __restrict__ solution,
    const float* __restrict__ independent_term,
    int elements_in_block, int num_blocks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = elements_in_block * num_blocks;

    if (idx < total) {
        solution[idx] = (solution[idx] * independent_term[idx]) /
                        (float)elements_in_block;
    }
}

// =============================================================================
// Kernel 11: Column Sum (Reduction)
// =============================================================================

extern "C" __global__ void sum_columns_kernel(
    const float* __restrict__ z_matrix,
    float* __restrict__ variances,
    int elements_in_block, int num_blocks)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < num_blocks) {
        float sum = 0.0f;
        for (int row = 0; row < elements_in_block; row++) {
            sum += z_matrix[row * num_blocks + col];
        }
        variances[col] = sum;
    }
}

// =============================================================================
// Kernel 12: Entropy Update
// =============================================================================

extern "C" __global__ void update_entropy_kernel(
    const float* __restrict__ variances,
    const float* __restrict__ eigenvalues,
    float* __restrict__ entropies,
    float sigma_nn,
    int elements_in_block, int num_blocks_h, int num_blocks_v)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = num_blocks_h * num_blocks_v;

    if (block_idx < num_blocks) {
        float entropy = 0.0f;
        float s_val = variances[block_idx];

        // Sum over all eigenvalues
        for (int k = 0; k < elements_in_block; k++) {
            float L = eigenvalues[k];
            L = fmaxf(L, 0.0f);  // Clamp negative eigenvalues
            entropy += log2f(L * s_val + sigma_nn) + log2f(2.0f * M_PI * M_E);
        }

        entropies[block_idx] = entropy;
    }
}

// =============================================================================
// Kernel 13: Check Matrix Regularity
// =============================================================================

extern "C" __global__ void check_matrix_regular_kernel(
    const float* __restrict__ eigenvalues,
    int* __restrict__ is_regular,
    int elements_in_block)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    *is_regular = 1;
    for (int i = 0; i < elements_in_block; i++) {
        if (eigenvalues[i] < SPEED_EIGENVALUE_EPS) {
            *is_regular = 0;
            return;
        }
    }
}

// =============================================================================
// Kernel 14: Speed Score Computation (Parallel Reduction)
// =============================================================================

extern "C" __global__ void compute_speed_score_kernel(
    const float* __restrict__ ref_entropies,
    const float* __restrict__ ref_variances,
    const float* __restrict__ dis_entropies,
    const float* __restrict__ dis_variances,
    float* __restrict__ score,
    float sigma_nn, float nn_floor,
    int weight_var_mode, int elements_in_block, int num_blocks)
{
    __shared__ float shared_sum[32];  // For warp reduction

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute base_entropy threshold
    float base_entropy = elements_in_block *
        (log2f((1.0f + nn_floor) * sigma_nn) + log2f(2.0f * M_PI * M_E));

    float local_score = 0.0f;

    // Grid-stride loop
    for (int i = tid; i < num_blocks; i += blockDim.x * gridDim.x) {
        float ref_ent = ref_entropies[i];
        float dis_ent = dis_entropies[i];

        // Skip if both below threshold
        if (ref_ent < base_entropy && dis_ent < base_entropy) {
            continue;
        }

        float ref_var = ref_variances[i];
        float dis_var = dis_variances[i];

        float spatial_ref, spatial_dis;

        // Select weighting mode
        switch (weight_var_mode) {
            case 0:
            default:
                spatial_ref = ref_ent * log2f(1.0f + ref_var);
                spatial_dis = dis_ent * log2f(1.0f + dis_var);
                break;
            case 1:
                spatial_ref = ref_ent * log2f(1.0f + ref_var);
                spatial_dis = dis_ent * log2f(1.0f + ref_var);
                break;
            case 2:
                spatial_ref = ref_ent * log2f(1.0f + dis_var);
                spatial_dis = dis_ent * log2f(1.0f + dis_var);
                break;
            case 3:
                spatial_ref = ref_ent * log2f(1.0f + (ref_var + dis_var) / 2.0f);
                spatial_dis = dis_ent * log2f(1.0f + (ref_var + dis_var) / 2.0f);
                break;
            case 4:
                spatial_ref = ref_ent * log2f(1.0f + ref_var);
                spatial_dis = dis_ent * log2f(1.0f + (ref_var + dis_var) / 2.0f);
                break;
            case 5:
                spatial_ref = ref_ent * log2f(1.0f + ref_var);
                spatial_dis = dis_ent * log2f(1.0f + (0.75f * ref_var + 0.25f * dis_var));
                break;
            case 6:
                spatial_ref = ref_ent * log2f(1.0f + ref_var);
                spatial_dis = dis_ent * log2f(1.0f + (0.25f * ref_var + 0.75f * dis_var));
                break;
        }

        local_score += fabsf(spatial_ref - spatial_dis);
    }

    // Block-level reduction
    local_score = block_reduce_sum(local_score, shared_sum);

    // Write result from first thread of each block
    if (threadIdx.x == 0) {
        atomicAdd(score, local_score);
    }
}
