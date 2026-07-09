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

#ifndef SPEED_HELPERS_H_
#define SPEED_HELPERS_H_

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

#include "opt.h"

// Forward declaration for VmafDictionary
typedef struct VmafDictionary VmafDictionary;

// =============================================================================
// Constants
// =============================================================================

#define DEFAULT_SPEED_BLOCK_SIZE (5)
#define SPEED_NUM_SQUARE_BUFFERS (5)
#define SPEED_NUM_RECT_BUFFERS (1)
#define SPEED_NUM_FRAME_BUFFERS (2)
#define SPEED_NUM_SCALES (4)
#define SPEED_EIGENVALUE_EPS (1e-6)
#define SPEED_EIGENVALUE_MAX_ITERS (500)

#define DEFAULT_SPEED_SIGMA_NN (0.29)
#define DEFAULT_SPEED_MAX_VAL (1000.0)
#define DEFAULT_SPEED_NN_FLOOR (0.0)
#define DEFAULT_SPEED_KERNELSCALE (1.0)
#define DEFAULT_SPEED_PRESCALE (1.0)
#define DEFAULT_SPEED_PRESCALE_METHOD ("nearest")

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

// Valid kernel scales (copied from vif_tools.h for standalone compilation)
#define SPEED_NUM_KERNELSCALES 21
static const float speed_valid_kernelscales[SPEED_NUM_KERNELSCALES] = {
    1.0f,
    1.0f / 2.0f,
    3.0f / 2.0f,
    2.0f,
    2.0f / 3.0f,
    24.0f / 10.0f,
    360.0f / 97.0f,
    4.0f / 3.0f,
    3.5f / 3.0f,
    3.75f / 3.0f,
    4.25f / 3.0f,
    5.0f / 3.0f,
    3.0f,
    1.0f / 2.25f,
    1.4746f,
    1.54f,
    1.6f,
    1.06667f,
    0.711111f,
    0.740740f,
    1.111111f,
};

// Scaling methods
typedef enum SpeedScalingMethod {
    SPEED_SCALE_NEAREST = 0,
    SPEED_SCALE_BICUBIC = 1,
    SPEED_SCALE_LANCZOS4 = 2,
    SPEED_SCALE_BILINEAR = 3,
} SpeedScalingMethod;

// =============================================================================
// Shared Structs
// =============================================================================

typedef struct SpeedDimensions {
    size_t original_height;
    size_t original_width;
    size_t scaled_height;
    size_t scaled_width;
    size_t alloc_height;
    size_t alloc_width;
    size_t operating_height;
    size_t operating_width;
    size_t block_size;
    size_t truncated_width;
    size_t truncated_height;
    size_t num_blocks_horizontal;
    size_t num_blocks_vertical;
    size_t num_blocks;
    size_t elements_in_block;
    size_t submatrix_width;
    size_t submatrix_height;
} SpeedDimensions;

typedef struct SpeedResultBuffers {
    float *entropies;
    float *variances;
} SpeedResultBuffers;

typedef struct SpeedBuffers {
    float *independent_term;
    float *linear_system_sol;
    float *cov_mat;
    float *eigenvalues;
    float *tmp_buffer;
} SpeedBuffers;

// Everything that is passed in as a feature option and is needed for
// SpEED computation
typedef struct SpeedOptions {
    double speed_kernelscale;
    double speed_prescale;
    char *speed_prescale_method;
    double speed_sigma_nn;
    double speed_nn_floor;
    int speed_weight_var_mode;
} SpeedOptions;

// Everything that is needed to compute SpEED given a pair of float buffers
// (ref, dis), except for what is provided in SpeedOptions
typedef struct SpeedState {
    SpeedDimensions dimensions;
    SpeedResultBuffers ref_results;
    SpeedResultBuffers dis_results;
    SpeedBuffers buffers;
    size_t float_stride;
    void *speed_cuda_state;  // Opaque pointer for CUDA state
} SpeedState;

// =============================================================================
// Unified Feature Extractor States
// These structs are shared between CPU and CUDA implementations.
// - CPU uses cpu_frame_buffer_* fields
// - CUDA uses speed_state.speed_cuda_state for device buffers
// =============================================================================

typedef struct SpeedChromaState {
    SpeedState speed_state;
    SpeedOptions speed_options;
    VmafDictionary *feature_name_dict;

    // Feature options (offsets must match for VmafOption)
    double speed_chroma_kernelscale;
    double speed_chroma_prescale;
    char *speed_chroma_prescale_method;
    double speed_chroma_sigma_nn;
    double speed_chroma_nn_floor;
    double speed_chroma_max_val;
    int speed_weight_var_mode;

    // CPU-only frame buffers (NULL for CUDA)
    float *cpu_frame_buffer_ref;
    float *cpu_frame_buffer_dis;
} SpeedChromaState;

typedef struct SpeedTemporalState {
    SpeedState speed_state;
    SpeedOptions speed_options;
    VmafDictionary *feature_name_dict;
    int frame_index;  // Track frame number for temporal differencing

    // Feature options (offsets must match for VmafOption)
    double speed_temporal_kernelscale;
    double speed_temporal_prescale;
    char *speed_temporal_prescale_method;
    double speed_temporal_sigma_nn;
    double speed_temporal_nn_floor;
    double speed_temporal_max_val;
    bool speed_temporal_use_ref_diff;

    // CPU-only frame buffers (NULL for CUDA)
    float *cpu_frame_buffer_ref[2];
    float *cpu_frame_buffer_dis[2];
} SpeedTemporalState;

// =============================================================================
// Static Inline Helper Functions (available in both CPU and CUDA compilation units)
// =============================================================================

/**
 * Validate kernel scale value.
 */
static inline bool speed_validate_kernelscale(float kernelscale) {
    for (int i = 0; i < SPEED_NUM_KERNELSCALES; i++) {
        if (fabsf(kernelscale - speed_valid_kernelscales[i]) < 1e-3f) {
            return true;
        }
    }
    return false;
}

/**
 * Get scaling method from string.
 * @return 0 on success, -EINVAL on error
 */
static inline int speed_get_scaling_method(const char *method_str,
                                           SpeedScalingMethod *method) {
    if (!strcmp(method_str, "nearest")) {
        *method = SPEED_SCALE_NEAREST;
    } else if (!strcmp(method_str, "bilinear")) {
        *method = SPEED_SCALE_BILINEAR;
    } else if (!strcmp(method_str, "bicubic")) {
        *method = SPEED_SCALE_BICUBIC;
    } else if (!strcmp(method_str, "lanczos4")) {
        *method = SPEED_SCALE_LANCZOS4;
    } else {
        return -EINVAL;
    }
    return 0;
}

/**
 * Initialize SpeedDimensions based on input width, height and prescale factor.
 *
 * @param dim        Output dimensions struct to populate
 * @param w          Input width
 * @param h          Input height
 * @param speed_prescale  Prescale factor (1.0 = no prescaling)
 * @return 0 on success, negative errno on error
 */
static inline int speed_init_dimensions(SpeedDimensions *dim, int w, int h,
                                        double speed_prescale) {
    dim->original_height = h;
    dim->original_width = w;
    dim->scaled_height = (size_t)(dim->original_height * speed_prescale + 0.5);
    dim->scaled_width = (size_t)(dim->original_width * speed_prescale + 0.5);
    dim->alloc_height = MAX(dim->original_height, dim->scaled_height);
    dim->alloc_width = MAX(dim->original_width, dim->scaled_width);
    dim->operating_height = dim->scaled_height >> SPEED_NUM_SCALES;
    dim->operating_width = dim->scaled_width >> SPEED_NUM_SCALES;
    dim->block_size = DEFAULT_SPEED_BLOCK_SIZE;
    dim->truncated_width =
        (dim->operating_width / dim->block_size) * dim->block_size;
    dim->truncated_height =
        (dim->operating_height / dim->block_size) * dim->block_size;
    dim->num_blocks_horizontal = dim->truncated_width / dim->block_size;
    dim->num_blocks_vertical = dim->truncated_height / dim->block_size;
    dim->num_blocks = dim->num_blocks_horizontal * dim->num_blocks_vertical;
    dim->elements_in_block = dim->block_size * dim->block_size;
    dim->submatrix_width = dim->truncated_width - dim->block_size + 1;
    dim->submatrix_height = dim->truncated_height - dim->block_size + 1;

    if (dim->truncated_height == 0 || dim->truncated_width == 0) {
        return -EINVAL;
    }
    return 0;
}

// =============================================================================
// Functions declared here, defined in speed.c (CPU-only)
// =============================================================================

/**
 * Initialize SpeedState including CPU buffer allocations.
 *
 * @param s    SpeedState to initialize
 * @param opt  SpeedOptions with configuration
 * @param w    Input width
 * @param h    Input height
 * @return 0 on success, negative errno on error
 */
int speed_init(SpeedState *s, SpeedOptions *opt, int w, int h);

/**
 * Free SpeedState CPU buffers.
 *
 * @param s    SpeedState to close
 * @return 0 on success
 */
int speed_close(SpeedState *s);

/**
 * Extract SpEED score from preprocessed ref and dis buffers.
 *
 * @param s     Initialized SpeedState
 * @param opt   SpeedOptions
 * @param ref   Reference float buffer (will be modified)
 * @param dis   Distorted float buffer (will be modified)
 * @param score Output score
 * @return 0 on success, negative errno on error
 */
int speed_extract_score(SpeedState *s, SpeedOptions *opt, float *ref,
                        float *dis, float *score);



static const char *provided_features_temporal[] = {
    "Speed_temporal_feature_speed_temporal_score",
    NULL
};

static const char *provided_features_chroma[] = {
    "Speed_chroma_feature_speed_chroma_u_score",
    "Speed_chroma_feature_speed_chroma_v_score",
    "Speed_chroma_feature_speed_chroma_uv_score",
    NULL
};


// =============================================================================
// VmafOption Macros for Feature Extractors
// =============================================================================

// Macro to generate VmafOption entries for speed options
// Usage: SPEED_OPTIONS_DEFINE(state_type, prefix)
// where prefix is the struct member prefix (e.g., speed_chroma_ or speed_temporal_)

#define SPEED_OPTION_KERNELSCALE(state_type, member) \
    { \
        .name = "speed_kernelscale", \
        .help = "scaling factor for the gaussian kernel (2.0 means " \
                "multiplying the standard deviation by 2 and enlarge " \
                "the kernel size accordingly", \
        .offset = offsetof(state_type, member##kernelscale), \
        .type = VMAF_OPT_TYPE_DOUBLE, \
        .default_val.d = DEFAULT_SPEED_KERNELSCALE, \
        .min = 0.1, \
        .max = 4.0, \
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM, \
        .alias = "ks", \
    }

#define SPEED_OPTION_PRESCALE(state_type, member) \
    { \
        .name = "speed_prescale", \
        .help = "scaling factor for the frame (2.0 means " \
                "making the image twice as large on each dimension)", \
        .offset = offsetof(state_type, member##prescale), \
        .type = VMAF_OPT_TYPE_DOUBLE, \
        .default_val.d = DEFAULT_SPEED_PRESCALE, \
        .min = 0.1, \
        .max = 4.0, \
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM, \
        .alias = "ps", \
    }

#define SPEED_OPTION_PRESCALE_METHOD(state_type, member) \
    { \
        .name = "speed_prescale_method", \
        .help = "scaling method for the frame, supported options: " \
                "[nearest, bilinear, bicubic, lanczos4]", \
        .offset = offsetof(state_type, member##prescale_method), \
        .type = VMAF_OPT_TYPE_STRING, \
        .default_val.s = DEFAULT_SPEED_PRESCALE_METHOD, \
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM, \
        .alias = "psm", \
    }

#define SPEED_OPTION_SIGMA_NN(state_type, member) \
    { \
        .name = "speed_sigma_nn", \
        .help = "standard deviation of neural noise", \
        .offset = offsetof(state_type, member##sigma_nn), \
        .type = VMAF_OPT_TYPE_DOUBLE, \
        .default_val.d = DEFAULT_SPEED_SIGMA_NN, \
        .min = 0.1, \
        .max = 2.0, \
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM, \
        .alias = "snn", \
    }

#define SPEED_OPTION_NN_FLOOR(state_type, member) \
    { \
        .name = "speed_nn_floor", \
        .help = "neural noise floor, expressed in percentage of sigma_nn", \
        .offset = offsetof(state_type, member##nn_floor), \
        .type = VMAF_OPT_TYPE_DOUBLE, \
        .default_val.d = DEFAULT_SPEED_NN_FLOOR, \
        .min = 0.0, \
        .max = 1.0, \
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM, \
        .alias = "nnf", \
    }

#define SPEED_OPTION_MAX_VAL(state_type, member) \
    { \
        .name = "speed_max_val", \
        .help = "maximum value allowed; larger values will be clipped to this value", \
        .offset = offsetof(state_type, member##max_val), \
        .type = VMAF_OPT_TYPE_DOUBLE, \
        .default_val.d = DEFAULT_SPEED_MAX_VAL, \
        .min = 0.0, \
        .max = 1000.0, \
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM, \
        .alias = "mxv", \
    }

#define SPEED_OPTION_WEIGHT_VAR_MODE(state_type, member) \
    { \
        .name = "speed_weight_var_mode", \
        .help = "different approaches to perform variance-based weighting", \
        .offset = offsetof(state_type, member), \
        .type = VMAF_OPT_TYPE_INT, \
        .default_val.i = 0, \
        .min = 0, \
        .max = 6, \
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM, \
        .alias = "wvm", \
    }

#define SPEED_OPTION_USE_REF_DIFF(state_type, member) \
    { \
        .name = "speed_use_ref_diff", \
        .help = "debug mode: enable additional output", \
        .offset = offsetof(state_type, member), \
        .type = VMAF_OPT_TYPE_BOOL, \
        .default_val.b = false, \
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM, \
        .alias = "urd", \
    }

#endif /* SPEED_HELPERS_H_ */

