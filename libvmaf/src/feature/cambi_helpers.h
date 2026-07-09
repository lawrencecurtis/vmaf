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

#pragma once

#define DEFAULT_CAMBI_TOPK_POOLING (0.6)
#define DEFAULT_CAMBI_WINDOW_SIZE (65)
#define DEFAULT_CAMBI_TVI (0.019)
#define DEFAULT_CAMBI_VLT (0.0)
#define DEFAULT_CAMBI_MAX_LOG_CONTRAST (2)
#define DEFAULT_CAMBI_FULL_REF_FLAG (false)
#define DEFAULT_CAMBI_EOTF ("bt1886")
#define DEFAULT_CAMBI_HIGH_RES_SPEEDUP (0)
#define DEFAULT_CAMBI_MAX_VAL (1000.0)
#define CAMBI_4K_HEIGHT (2160)
#define NUM_SCALES 5
#define PICS_BUFFER_SIZE 2
#define MASK_FILTER_SIZE 7

static const int g_scale_weights[NUM_SCALES] = {16, 8, 4, 2, 1};

static FORCE_INLINE uint16_t ceil_log2(uint32_t num) {
    if (num==0)
        return 0;

    uint32_t tmp = num - 1;
    uint16_t shift = 0;
    while (tmp>0) {
        tmp >>= 1;
        shift += 1;
    }
    return shift;
}

static FORCE_INLINE uint16_t get_mask_index(unsigned input_width, unsigned input_height,
                                            uint16_t filter_size) {
    uint32_t shifted_wh = (input_width >> 6) * (input_height >> 6);
    return (filter_size * filter_size + 3 * (ceil_log2(shifted_wh) - 11) - 1)>>1;
}

static int dump_c_values(FILE *heatmaps_files[], const float *c_values, int width, int height, int scale,
                         int window_size, const uint16_t num_diffs, const int *diff_weights, int frame) {
    int max_diff_weight = diff_weights[0];
    for (int i = 0; i < num_diffs; i++) {
        if (diff_weights[i] > max_diff_weight) {
            max_diff_weight = diff_weights[i];
        }
    }
    int max_c_value = max_diff_weight * window_size * window_size / 4;
    int max_16bit_value = (1 << 16) - 1;
    double scaling_value = (double)max_16bit_value / max_c_value;
    FILE *file = heatmaps_files[scale];
    uint16_t *to_write = malloc(width * sizeof(uint16_t));
    if (!to_write) return -ENOMEM;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            to_write[j] = (uint16_t)(scaling_value * c_values[i * width + j]);
        }
        ptrdiff_t offset = (frame * height + i) * width * sizeof(uint16_t);
        fseek(file, offset, SEEK_SET);
        fwrite((void*)to_write, sizeof(uint16_t), width, file);
    }
    free(to_write);
    return 0;
}

typedef struct CambiBuffers {
    float *c_values;
    uint32_t *mask_dp;
    uint16_t *c_values_histograms;
    uint16_t *filter_mode_buffer;
    uint16_t *diffs_to_consider;
    uint16_t *tvi_for_diff;
    uint16_t *derivative_buffer;
    int *diff_weights;
    int *all_diffs;
    uint16_t v_band_base;
    uint16_t v_band_size;
} CambiBuffers;

typedef void (*VmafDerivativeCalculator)(const uint16_t *image_data, uint16_t *derivative_buffer, int width, int height, int row, int stride);
typedef void (*VmafFilterMode)(const VmafPicture *image, int width, int height, uint16_t *buffer);
typedef void (*VmafDecimate)(VmafPicture *image, unsigned width, unsigned height);
typedef void (*VmafCalcCValues)(VmafPicture *pic, const VmafPicture *mask_pic,
                                float *c_values, uint16_t *histograms, uint16_t window_size,
                                const uint16_t num_diffs, const uint16_t *tvi_for_diff, uint16_t vlt_luma,
                                const int *diff_weights, const int *all_diffs, int width, int height);

typedef struct CambiState {
    VmafPicture pics[PICS_BUFFER_SIZE];
    unsigned enc_width;
    unsigned enc_height;
    unsigned enc_bitdepth;
    unsigned src_width;
    unsigned src_height;
    uint16_t window_size;
    uint16_t src_window_size;
    double topk;
    double cambi_topk;
    double tvi_threshold;
    double cambi_max_val;
    double cambi_vis_lum_threshold;
    uint16_t vlt_luma;
    uint16_t max_log_contrast;
    char *heatmaps_path;
    char *eotf;
    char *cambi_eotf;
    bool full_ref;
    int cambi_high_res_speedup;

    FILE *heatmaps_files[NUM_SCALES];
    VmafDerivativeCalculator derivative_callback;
    VmafCalcCValues calc_c_values_callback;
    VmafFilterMode filter_mode_callback;
    VmafDecimate decimate_callback;
    CambiBuffers buffers;
    VmafDictionary *feature_name_dict;
    void* cambi_cuda_state;
} CambiState;

enum CambiTVIBisectFlag {
    CAMBI_TVI_BISECT_TOO_SMALL,
    CAMBI_TVI_BISECT_CORRECT,
    CAMBI_TVI_BISECT_TOO_BIG
};


static const VmafOption options[] = {
    {
        .name = "cambi_max_val",
        .help = "maximum value allowed; larger values will be clipped to this value",
        .offset = offsetof(CambiState, cambi_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "cmxv",
    },
    {
        .name = "enc_width",
        .help = "Encoding width",
        .offset = offsetof(CambiState, enc_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 180,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encw",
    },
    {
        .name = "enc_height",
        .help = "Encoding height",
        .offset = offsetof(CambiState, enc_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 150,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ench",
    },
    {
        .name = "enc_bitdepth",
        .help = "Encoding bitdepth",
        .offset = offsetof(CambiState, enc_bitdepth),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 6,
        .max = 16,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encbd",
    },
    {
        .name = "src_width",
        .help = "Source width. Only used when full_ref=true.",
        .offset = offsetof(CambiState, src_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 320,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "srcw",
    },
    {
        .name = "src_height",
        .help = "Source height. Only used when full_ref=true.",
        .offset = offsetof(CambiState, src_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 200,
        .max = 4320,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "srch",
    },
    {
        .name = "window_size",
        .help = "Window size to compute CAMBI: 65 corresponds to ~1 degree at 4k",
        .offset = offsetof(CambiState, window_size),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_WINDOW_SIZE,
        .min = 15,
        .max = 127,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ws",
    },
    {
        .name = "topk",
        .help = "Ratio of pixels for the spatial pooling computation, must be 0 < topk <= 1.0",
        .offset = offsetof(CambiState, topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TOPK_POOLING,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_topk",
        .help = "Ratio of pixels for the spatial pooling computation, must be 0 < cambi_topk <= 1.0",
        .offset = offsetof(CambiState, cambi_topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TOPK_POOLING,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ctpk",
    },
    {
        .name = "tvi_threshold",
        .help = "Visibilty threshold for luminance ΔL < tvi_threshold*L_mean",
        .offset = offsetof(CambiState, tvi_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TVI,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "tvit",
    },
    {
        .name = "cambi_vis_lum_threshold",
        .help = "Luminance value below which we assume any banding is not visible",
        .offset = offsetof(CambiState, cambi_vis_lum_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_VLT,
        .min = 0.0,
        .max = 300.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "vlt",
    },
    {
        .name = "max_log_contrast",
        .help = "Maximum contrast in log luma level (2^max_log_contrast) at 10-bits, "
                "e.g., 2 is equivalent to 4 luma levels at 10-bit and 1 luma level at 8-bit. "
                "From 0 to 5: default 2 is recommended for banding from compression.",
        .offset = offsetof(CambiState, max_log_contrast),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_MAX_LOG_CONTRAST,
        .min = 0,
        .max = 5,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mlc",
    },
    {
        .name = "heatmaps_path",
        .help = "Path where heatmaps will be dumped.",
        .offset = offsetof(CambiState, heatmaps_path),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = NULL,
    },
    {
        .name = "full_ref",
        .help = "If true, CAMBI will be run in full-reference mode and will be computed on both the reference and distorted inputs",
        .offset = offsetof(CambiState, full_ref),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = DEFAULT_CAMBI_FULL_REF_FLAG,
    },
    {
        .name = "eotf",
        .help = "Determines the EOTF used to compute the visibility thresholds. Possible values: ['bt1886', 'pq']. Default: 'bt1886'",
        .offset = offsetof(CambiState, eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_CAMBI_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_eotf",
        .help = "Determines the EOTF used to compute the visibility thresholds. Possible values: ['bt1886', 'pq']. Default: 'bt1886'. If both eotf and cambi_eotf are set, cambi_eotf takes precedence.",
        .offset = offsetof(CambiState, cambi_eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_CAMBI_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ceot",
    },
    {
        .name = "cambi_high_res_speedup",
        .help = "Speed up the processing by downsampling post spatial mask for resolutions >= 1080p. "
                "Min speed-up resolution possible values: [1080, 1440, 2160, 0]. Default: 0 (not applied)"
                "Note some loss of accuracy is expected with this speedup.",
        .offset = offsetof(CambiState, cambi_high_res_speedup),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_HIGH_RES_SPEEDUP,
        .min = 0,
        .max = CAMBI_4K_HEIGHT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "hrs",
    },
    { 0 }
};

static const char *provided_features[] = {
    "Cambi_feature_cambi_score",
    NULL
};
