#include "common.h"
#include "cuda_helper.cuh"
#include "cambi.h"

#include <iostream>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template<typename T, bool is_scaling>
__device__ uint16_t get_input_pixel_10b(const T* input_image, int x, int y, int stride, float ratio_x, float ratio_y, int bpc) {
    unsigned ori_x = x;
    unsigned ori_y = y;
    if constexpr (is_scaling) {
        const float start_x = ratio_x / 2.f - 0.5f;
        const float start_y = ratio_y / 2.f - 0.5f;
        ori_x = (int)(start_x + (x * ratio_x + 0.5f));
        ori_y = (int)(start_y + (y * ratio_y + 0.5f));
    }

    const uint16_t max_val = (1 << bpc) - 1;
    uint16_t pixel_value = input_image[ori_y * stride / sizeof(T) + ori_x];
    uint16_t value_10b;
    assert(pixel_value <= max_val);  // Ensure pixel value is within valid range
    if (bpc >= 10 && std::is_same<T, uint16_t>::value) {
        int shift_factor = bpc - 10;
        int rounding_offset = shift_factor == 0 ? 0 : 1 << (shift_factor - 1);
        value_10b = (pixel_value + rounding_offset) >> shift_factor;
    } else if (bpc == 9 && std::is_same<T, uint16_t>::value) {
        value_10b = pixel_value << 1;
    } else if (bpc <= 8 && std::is_same<T, uint8_t>::value) {
        int shift_factor = 10 - bpc;
        value_10b = pixel_value << shift_factor;
    }
    return value_10b;
}


template<typename T, bool is_scaling>
__device__ void preprocess(const VmafPicture image, VmafPicture preprocessed, int width, int height, int enc_bitdepth) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    __restrict__ T* input_image = (T*)image.data[0];
    __restrict__ uint16_t* output_image = (uint16_t*)preprocessed.data[0];

    float ratio_x = (float)image.w[0] / width;
    float ratio_y = (float)image.h[0] / height;
    int out_stride = preprocessed.stride[0] >> 1 ; // in uint16_t

    if (x < width && y < height) {
        uint16_t pixel_value = get_input_pixel_10b<T, is_scaling>(input_image, x, y, image.stride[0], ratio_x, ratio_y, image.bpc);
        // anti dither
        uint16_t pixel_value_rhs     = __shfl_down_sync(0xFFFFFFFF,  pixel_value, 1);
        if (threadIdx.x % 32 == 31) {
            // last thread in warp
            pixel_value_rhs     = get_input_pixel_10b<T, is_scaling>(input_image, x + 1, y, image.stride[0], ratio_x, ratio_y, image.bpc);
        }

        uint16_t pixel_value_low     = 0;
        uint16_t pixel_value_low_rhs = 0;
        if (y < height - 1) {
            pixel_value_low     = get_input_pixel_10b<T, is_scaling>(input_image, x, y + 1, image.stride[0], ratio_x, ratio_y, image.bpc);
            pixel_value_low_rhs = __shfl_down_sync(0xFFFFFFFF,  pixel_value_low, 1);
            if (threadIdx.x % 32 == 31) {
                // last thread in warp
                pixel_value_low_rhs     = get_input_pixel_10b<T, is_scaling>(input_image, x + 1, y + 1, image.stride[0], ratio_x, ratio_y, image.bpc);
            }
        }
        int shift = 2 - (y == height - 1) - (x == width - 1);
        output_image[y * out_stride + x] = \
            (pixel_value + pixel_value_rhs + pixel_value_low + pixel_value_low_rhs) >> shift;
    } else {
        // make sure that all threads have same execution path
        uint16_t pixel_value = 0;
        uint16_t pixel_value_rhs = __shfl_down_sync(0xFFFFFFFF,  pixel_value, 1);
        if (y < height - 1) {
            uint16_t pixel_value_low     = 0;
            uint16_t pixel_value_low_rhs = __shfl_down_sync(0xFFFFFFFF,  pixel_value_low, 1);
        }
    }
}

template<typename T, bool is_scaling>
__device__ void preprocess_nodither(const VmafPicture image, VmafPicture preprocessed, int width, int height, int enc_bitdepth) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    __restrict__ T* input_image = (T*)image.data[0];
    __restrict__ uint16_t* output_image = (uint16_t*)preprocessed.data[0];

    float ratio_x = (float)image.w[0] / width;
    float ratio_y = (float)image.h[0] / height;
    int out_stride = preprocessed.stride[0] >> 1 ; // in uint16_t

    if (x < width && y < height) {
        uint16_t pixel_value = get_input_pixel_10b<T, is_scaling>(input_image, x, y, image.stride[0], ratio_x, ratio_y, image.bpc);
        output_image[y * out_stride + x] = pixel_value;
    }
}

#define TEMPALTE_PREPROCESS(type, scaling)                                                                             \
__global__ void preprocess_##type##_##scaling(const VmafPicture image, VmafPicture preprocessed, int width, int height, int enc_bitdepth) {   \
    preprocess<type, scaling>(image, preprocessed, width, height, enc_bitdepth);                                       \
}

#define TEMPALTE_PREPROCESS_NODITHER(type, scaling)                                                                             \
__global__ void preprocess_nodither_##type##_##scaling(const VmafPicture image, VmafPicture preprocessed, int width, int height, int enc_bitdepth) {   \
    preprocess_nodither<type, scaling>(image, preprocessed, width, height, enc_bitdepth);                                       \
}



__device__ uint16_t mode3(uint16_t a, uint16_t b, uint16_t c) {
    if (a == b || a == c) return a;
    if (b == c) return b;
    return min(a, min(b, c));
};

template<bool apply_decimation>
__device__ uint16_t read_pixel(const uint16_t* __restrict__ input, int i, int j, int in_stride) {
    if (apply_decimation) {
        return input[(i << 1) * in_stride + (j << 1)];
    } else {
        return input[i * in_stride + j];
    }
}


template<bool apply_decimation, int blockDimX = 16, int blockDimY = 16>
__device__ void decimate_and_filter_kernel(
    const uint16_t* __restrict__ input,
    uint16_t* __restrict__ output,
    int in_width, int in_height,
    int out_width, int out_height,
    int in_stride, int out_stride)
{
	assert(blockDimX == blockDim.x && blockDimY == blockDim.y); // assumed in shared memory allocation
    int i = blockIdx.y * blockDimY + threadIdx.y;
    int j = blockIdx.x * blockDimX + threadIdx.x;

    if (i >= out_height || j >= out_width) return;

    uint16_t value = read_pixel<apply_decimation>(input, i, j, in_stride);
	constexpr int pad = 1;
    __shared__ uint16_t smem[2 * pad + blockDimY][2 * pad + blockDimX];

    int tx = threadIdx.x + pad;
    int ty = threadIdx.y + pad;
    smem[ty][tx] = value;

	// edges
    if (threadIdx.x == 0) {
        if (j > 0) {
            smem[ty][0] = read_pixel<apply_decimation>(input, i, j - 1, in_stride);
        } else {
            smem[ty][0] = 0;
        }
    }
    if (threadIdx.x == blockDimX - 1) {
        if (j < (in_width - 1)) {
            smem[ty][tx + 1] = read_pixel<apply_decimation>(input, i, j + 1, in_stride);
        } else {
            smem[ty][tx + 1] = 0;
        }
    }
    if (threadIdx.y == 0) {
        if (i > 0) {
            smem[0][tx] = read_pixel<apply_decimation>(input, i - 1, j, in_stride);
        } else {
            smem[0][tx] = 0;
        }
    }
    if (threadIdx.y == blockDimY - 1) {
        if (i < (in_height - 1)) {
            smem[ty + 1][tx] = read_pixel<apply_decimation>(input, i + 1, j, in_stride);
        } else {
            smem[ty + 1][tx] = 0;
        }
    }

	// corners
	// top left corner
	if (threadIdx.x == 0 && threadIdx.y == 0 ) {
	    if (j > 0 && i > 0) {
	        smem[0][0] = read_pixel<apply_decimation>(input, i - 1,
                j - 1, in_stride);
	    } else {
	        smem[0][0] = 0;
	    }
	}
	// top right corner
	if (threadIdx.x == blockDimX - 1 && threadIdx.y == 0) {
	    if (j < (in_width - 1) && i > 0) {
	        smem[0][tx + 1] = \
                read_pixel<apply_decimation>(input, i - 1, j + 1, in_stride);
	    } else {
	        smem[0][tx + 1] = 0;
	    }
	}
	// bottom left corner
	if (threadIdx.x == 0 && threadIdx.y == blockDimY - 1) {
	    if (i < (in_height - 1) && j > 0) {
	        smem[ty + 1][0] = \
                read_pixel<apply_decimation>(input, i + 1, j - 1, in_stride);
	    } else {
	        smem[ty + 1][0] = 0;
	    }
	}
    // bottom right corner
	if (threadIdx.x == blockDimX - 1 && threadIdx.y == blockDimY - 1) {
	    if  (j < (in_width - 1) && i < (in_height - 1)) {
	        smem[ty + 1][tx + 1] = \
                read_pixel<apply_decimation>(input, i + 1, j + 1, in_stride);
	    } else {
	        smem[ty + 1][tx + 1] = 0;
	    }
	}
    __syncthreads();

    uint16_t result;
    if (i == 0 || i == (out_height - 1)) {
        result = smem[ty][tx]; // value;
    } else if (j == 0 || j == (out_width - 1)) {
        uint16_t top     = smem[ty - 1][tx];
        uint16_t middle  = smem[ty][tx]; // value;
        uint16_t bottom  = smem[ty + 1][tx];
        result = mode3(top, middle, bottom);
    } else {
        uint16_t top =    mode3(smem[ty - 1][tx - 1], smem[ty - 1][tx], smem[ty - 1][tx + 1]);
        uint16_t middle = mode3(smem[ty][tx - 1], smem[ty][tx], smem[ty][tx + 1]);
        uint16_t bottom = mode3(smem[ty + 1][tx - 1], smem[ty + 1][tx], smem[ty + 1][tx + 1]);
        result = mode3(top, middle, bottom);
	}
	output[i * out_stride + j] = result;
}

#define TEMPALTE_DECIMATE_FILTER(decimate)                                                                                                                                                       \
__global__ void decimate_and_filter_kernel_##decimate(const uint16_t* __restrict__ input, uint16_t* __restrict__ output, \
                                                      int in_width, int in_height, int out_width, int out_height, int in_stride, int out_stride) {  \
    decimate_and_filter_kernel<decimate>(input, output,in_width,  in_height,out_width, out_height,in_stride,out_stride);                      \
}


extern "C" __global__ void decimate_kernel(
    uint16_t* mask,
    int in_width, int in_height,
    int out_width, int out_height,
    int stride)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= out_height || j >= out_width) return;

	mask[i * stride + j] = read_pixel<true>(mask, i, j, stride);

}


extern "C" __global__ void calculate_derivate(const VmafPicture image,
    VmafCudaBuffer derivative_buffer,
    int width, int height) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= height || j >= width) return;

    const uint16_t* __restrict__ image_data = (const uint16_t*)image.data[0];
    uint8_t* __restrict__ derivative_out = (uint8_t*)derivative_buffer.data;

    int stride = image.stride[0] / sizeof(uint16_t);

    bool horizontal_equal = (j == width - 1) ||
                           (image_data[i * stride + j] == image_data[i * stride + j + 1]);

    bool vertical_equal = (i == height - 1) ||
                         (image_data[i * stride + j] == image_data[(i + 1) * stride + j]);

    derivative_out[i * width + j] = (horizontal_equal && vertical_equal);
}

extern "C" __global__ void calculate_spatial_mask(VmafPicture mask, VmafCudaBuffer derivative_buffer,
                                                  int width, int height, uint16_t mask_index) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= height || j >= width) return;

    uint8_t* __restrict__ derivative = (uint8_t*)derivative_buffer.data;
    uint16_t* __restrict__ mask_data = (uint16_t*)mask.data[0];

    int mask_stride = mask.stride[0] / sizeof(uint16_t);

    constexpr int pad_size = MASK_FILTER_SIZE >> 1;
    int top = max(0, i - (int)pad_size);
    int bottom = min(height - 1, i + (int)pad_size);
    int left = max(0, j - (int)pad_size);
    int right = min(width - 1, j + (int)pad_size);


    uint32_t sum = 0;
    for (int wi = top; wi <= bottom; wi++) {
        for (int wj = left; wj <= right; wj++) {
            sum += derivative[wi * width + wj];
        }
    }
    mask_data[i * mask_stride + j] = (sum > mask_index) ? 1 : 0;
}


extern "C" __global__ void calculate_c_values_kernel(
    const VmafPicture image, const VmafPicture mask,
    VmafCudaBuffer c_values_out,
    int width, int height, uint16_t window_size,
    VmafCudaBuffer tvi_for_diff_buf, VmafCudaBuffer diff_weights_buf, VmafCudaBuffer all_diffs_buf,
    uint16_t num_diffs, uint16_t vlt_luma)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= height || j >= width) return;

    const uint16_t* __restrict__ image_data = (const uint16_t*)image.data[0];
    const uint16_t* __restrict__ mask_data = (const uint16_t*)mask.data[0];
    float* __restrict__ c_values = (float*)c_values_out.data;
    const uint16_t* __restrict__ tvi_for_diff = (const uint16_t*)tvi_for_diff_buf.data;
    const int* __restrict__ diff_weights = (const int*)diff_weights_buf.data;
    const int* __restrict__ all_diffs = (const int*)all_diffs_buf.data;

    int img_stride = image.stride[0] / sizeof(uint16_t);
    int mask_stride = mask.stride[0] / sizeof(uint16_t);
    uint16_t pad_size = window_size >> 1;

    if (mask_data[i * mask_stride + j] == 0) {
        c_values[i * width + j] = 0.0f;
        return;
    }

    uint16_t value = image_data[i * img_stride + j];
    uint16_t value_offset = value + num_diffs;  // Offset value to match CPU implementation
	uint16_t local_diffs[2 * MAX_CAMBI_MAX_LOG_CONTRAST + 1] = {0};
    int top = max(0, i - (int)pad_size);
    int bottom = min(height - 1, i + (int)pad_size);
    int left = max(0, j - (int)pad_size);
    int right = min(width - 1, j + (int)pad_size);
    for (int wi = top; wi <= bottom; wi++) {
        for (int wj = left; wj <= right; wj++) {
            // Only count pixels where mask is non-zero, matching CPU implementation
            if (mask_data[wi * mask_stride + wj]) {
                for (int d = 0; d < 2 * num_diffs + 1; d++) {
                    if (image_data[wi * img_stride + wj] == value + all_diffs[d]) {
                        local_diffs[d]++;
                    }
                }
            }
		}
	}

    uint16_t p_0 = local_diffs[num_diffs];
    float c_value = 0.0f;

    for (uint16_t d = 0; d < num_diffs; d++) {
        if ((value_offset <= tvi_for_diff[d]) && ((value_offset + all_diffs[num_diffs + d + 1]) > vlt_luma)) {
            uint16_t p_1 = local_diffs[num_diffs + d + 1];
            uint16_t p_2 = local_diffs[num_diffs - d - 1];
            float val;
            if (p_1 > p_2) {
                val = (float)(diff_weights[d] * p_0 * p_1) / (p_1 + p_0);
            } else {
                val = (float)(diff_weights[d] * p_0 * p_2) / (p_2 + p_0);
            }
            if (val > c_value) {
                c_value = val;
            }
        }
    }

    c_values[i * width + j] = c_value;
}

extern "C" {
    TEMPALTE_PREPROCESS(uint8_t, true);
    TEMPALTE_PREPROCESS(uint8_t, false);
    TEMPALTE_PREPROCESS(uint16_t, true);
    TEMPALTE_PREPROCESS(uint16_t, false);
    TEMPALTE_PREPROCESS_NODITHER(uint16_t, true);
    TEMPALTE_PREPROCESS_NODITHER(uint16_t, false);

    TEMPALTE_DECIMATE_FILTER(true);
    TEMPALTE_DECIMATE_FILTER(false);
}
