#include <obs-module.h>
#include "background-matting/include/c_api.h"
#include <media-io/video-scaler.h>

#define TFLITE_WIDTH  256
#define TFLITE_HEIGHT 256
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

struct background_mask_filter_data {
	obs_source_t *context;
	gs_effect_t *effect;

	uint8_t * rgb_int;
	float * rgb_f;
	float * output_probability;
	TfLiteTensor *input_tensor;
	TfLiteInterpreter *interpreter;
	uint32_t frame_width;
	uint32_t frame_height;
	uint32_t rgb_linesize;
	video_scaler_t *scalerToBGR;

	uint8_t *texturedata;
	gs_texture_t *tex;
	gs_eparam_t *mask;
	gs_eparam_t *texelSize_param;
	gs_eparam_t *step_param;
	gs_eparam_t *radius_param;
	gs_eparam_t *offset_param;
	gs_eparam_t *sigmaTexel_param;
	gs_eparam_t *sigmaColor_param;

	struct vec2* texelSize;
	float step;
	float radius;
	float offset;
	float sigmaTexel;
	float sigmaColor;

	double mask_value;
};

static const char *background_mask_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "background_mask";
}

static void destroyScalers(struct background_mask_filter_data *filter) {
	if (filter->scalerToBGR) {
		video_scaler_destroy(filter->scalerToBGR);
		filter->scalerToBGR = NULL;
	}
}

static void initializeScalers(
	uint32_t width,
	uint32_t height,
	enum video_format frameFormat,
	struct background_mask_filter_data *filter) {

	struct video_scale_info dst = {
		VIDEO_FORMAT_BGR3,
		TFLITE_WIDTH,
		TFLITE_HEIGHT,
		VIDEO_RANGE_DEFAULT,
		VIDEO_CS_DEFAULT
	};

	struct video_scale_info src = {
		frameFormat,
		width,
		height,
		VIDEO_RANGE_DEFAULT,
		VIDEO_CS_DEFAULT
	};

	destroyScalers(filter);
	video_scaler_create(&filter->scalerToBGR, &dst, &src, VIDEO_SCALE_DEFAULT);
	//	video_scaler_create(&filter->scalerFromBGR, &src, &dst, VIDEO_SCALE_DEFAULT);
}

static void convertFrameToBGR(
	struct obs_source_frame *frame,
	struct background_mask_filter_data *filter) {
	if (filter->scalerToBGR == NULL) {
		// Lazy initialize the frame scale & color converter
		initializeScalers(filter->frame_width, filter->frame_height, frame->format, filter);
	}

	video_scaler_scale(filter->scalerToBGR,
			   &(filter->rgb_int), &(filter->rgb_linesize),
			   frame->data, frame->linesize);
}

static void background_mask_destroy(void *data)
{
	struct background_mask_filter_data *filter = data;

	if (filter->effect) {
		obs_enter_graphics();
		gs_effect_destroy(filter->effect);
		obs_leave_graphics();
	}

	if (filter->rgb_int) {
		bfree(filter->rgb_int);
		filter->rgb_int = NULL;
	}
	if (filter->rgb_f) {
		bfree(filter->rgb_f);
		filter->rgb_f = NULL;
	}
	if (filter->output_probability) {
		bfree(filter->output_probability);
		filter->output_probability = NULL;
	}
	destroyScalers(filter);
	if (filter->interpreter) {
		TfLiteInterpreterDelete(filter->interpreter);
		filter->interpreter = NULL;
	}
	if (filter->input_tensor) {
		filter->input_tensor = NULL;
	}

	if (filter->texturedata) {
		bfree(filter->texturedata);
		filter->texturedata = NULL;
	}

	if (filter->texelSize) {
		filter->texelSize = bzalloc(sizeof (struct vec2));
		filter->texelSize = NULL;
	}

	bfree(data);
}

static void *background_mask_create(obs_data_t *settings, obs_source_t *context)
{
	struct background_mask_filter_data *filter =
		bzalloc(sizeof(struct background_mask_filter_data));
	char *effect_path = obs_module_file("background_mask.effect");
	filter->context = context;
	obs_enter_graphics();
	filter->effect = gs_effect_create_from_file(effect_path, NULL);
	filter->mask = gs_effect_get_param_by_name(filter->effect, "mask");
	filter->texelSize_param = gs_effect_get_param_by_name(filter->effect, "u_texelSize");
	filter->step_param = gs_effect_get_param_by_name(filter->effect, "u_step");
	filter->radius_param = gs_effect_get_param_by_name(filter->effect, "u_radius");
	filter->offset_param = gs_effect_get_param_by_name(filter->effect, "u_offset");
	filter->sigmaTexel_param = gs_effect_get_param_by_name(filter->effect, "u_sigmaTexel");
	filter->sigmaColor_param = gs_effect_get_param_by_name(filter->effect, "u_sigmaColor");
	if (!filter->tex) {
		filter->tex = gs_texture_create(TFLITE_WIDTH, TFLITE_HEIGHT, GS_R8, 1, NULL, GS_DYNAMIC );
	}
	gs_effect_set_texture(filter->mask, filter->tex);
	obs_leave_graphics();

	bfree(effect_path);

	filter->mask_value = obs_data_get_double(settings, "SETTING_MASK");
	if (!filter->effect) {
		background_mask_destroy(filter);
		return NULL;
	}
	char *model_path = obs_module_file("tflite/mlkit.tflite");
	TfLiteModel *model =
		TfLiteModelCreateFromFile(model_path);
	TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetNumThreads(options, 2);
	filter->interpreter = TfLiteInterpreterCreate(model, options);
	// The options/model can be deleted immediately after interpreter creation.
	TfLiteInterpreterOptionsDelete(options);
	TfLiteModelDelete(model);
	bfree(model_path);

	TfLiteInterpreterAllocateTensors(filter->interpreter);

	filter->input_tensor = TfLiteInterpreterGetInputTensor(filter->interpreter, 0);
	filter->rgb_linesize = TFLITE_WIDTH * 3;

	return filter;
}

static void background_mask_render(void *data, gs_effect_t *effect)
{
	struct background_mask_filter_data *filter = data;

	if (!obs_source_process_filter_begin(filter->context, GS_RGBA,
					     OBS_ALLOW_DIRECT_RENDERING))
		return;
	gs_texture_set_image(filter->tex,
			     filter->texturedata,
			     TFLITE_WIDTH, false);
	gs_effect_set_texture(filter->mask,  filter->tex);
	gs_effect_set_vec2(filter->texelSize_param, filter->texelSize);
	gs_effect_set_float(filter->step_param, filter->step);
	gs_effect_set_float(filter->radius_param, filter->radius);
	gs_effect_set_float(filter->offset_param, filter->offset);
	gs_effect_set_float(filter->sigmaTexel_param, filter->sigmaTexel);
	gs_effect_set_float(filter->sigmaColor_param, filter->sigmaColor);

	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);

	obs_source_process_filter_end(filter->context, filter->effect, 0, 0);

	gs_blend_state_pop();

	UNUSED_PARAMETER(effect);
}

void calcSmoothParameters(struct background_mask_filter_data *filter, float frameWidth, float frameHeight, float tfWidth, float tfHeight){
	float sigmaSpace = 1.0f;
	float kSparsityFactor = 0.66f; // Higher is more sparse.

	sigmaSpace *= MAX( frameWidth / tfWidth, frameHeight / tfHeight );

	float sparsity = MAX(1., sqrt(sigmaSpace) * kSparsityFactor);

	filter->step = sparsity;
	filter->radius = sigmaSpace;
	filter->offset = filter->step > 1.f ? filter->step * 0.5f : 0.f;

	float texelWidth = 1.f / frameWidth;
	float texelHeight = 1.f / frameHeight;
	filter->texelSize->x = texelWidth;
	filter->texelSize->y = texelHeight;

	filter->sigmaTexel = MAX(texelWidth, texelHeight) * sigmaSpace;
	filter->sigmaColor = 0.1f;
	blog(LOG_ERROR, "step = %f, radius = %f, offset = %f, texelSize-x = %f, texelSize-y = %f, ",
	     filter->step, filter->radius, filter->offset, filter->texelSize->x, filter->texelSize->y);
}

static void mirror_inversion(struct obs_source_frame *frame_data);
static void tflite_get_out(struct background_mask_filter_data * filter);
static struct obs_source_frame *
background_mask_video(void *data, struct obs_source_frame *frame)
{
	struct background_mask_filter_data *filter = data;
	if (filter->frame_width != frame->width || filter->frame_height != frame->height) {
		destroyScalers(filter);
		filter->frame_width = frame->width;
		filter->frame_height = frame->height;
		if (!filter->texelSize) {
			filter->texelSize = bzalloc(sizeof (struct vec2));
		}
		calcSmoothParameters(filter, frame->width, frame->height, TFLITE_WIDTH, TFLITE_HEIGHT);
	}

	if (!filter->texelSize) {
		filter->texelSize = bzalloc(sizeof (struct vec2));
		calcSmoothParameters(filter, frame->width, frame->height, TFLITE_WIDTH, TFLITE_HEIGHT);
	}

	if (!filter->rgb_int) {
		filter->rgb_int = (
			bzalloc(filter->rgb_linesize * TFLITE_HEIGHT *
				sizeof(uint8_t)));
	}
	convertFrameToBGR(frame, filter);

	if (!filter->rgb_f) {
		filter->rgb_f = bzalloc(filter->rgb_linesize * TFLITE_HEIGHT * sizeof(float));
	}
	uint32_t pos;
	uint32_t pos_f;
	for (int i = 0; i < TFLITE_HEIGHT; ++i) {
		for (int j = 0; j <= TFLITE_WIDTH / 2; ++j) {
			pos = filter->rgb_linesize * i + 3 * j;
			pos_f = filter->rgb_linesize * i + 3 * (TFLITE_WIDTH - 1 - j);
			*(filter->rgb_f + pos_f + 2) = *(filter->rgb_int + pos) / 255.0f;
			*(filter->rgb_f + pos_f + 1) = *(filter->rgb_int + pos + 1) / 255.0f;
			*(filter->rgb_f + pos_f) = *(filter->rgb_int + pos + 2) / 255.0f;
		}
	}

	mirror_inversion(frame);
	convertFrameToBGR(frame, filter);

	for (int i = 0; i < TFLITE_HEIGHT; ++i) {
		for (int j = 0; j <= TFLITE_WIDTH / 2; ++j) {
			pos = filter->rgb_linesize * i + 3 * j;
			*(filter->rgb_f + pos + 1) = *(filter->rgb_int + pos + 1) / 255.0f;
			*(filter->rgb_f + pos) = *(filter->rgb_int + pos + 2) / 255.0f;
		}
	}

	tflite_get_out(filter);
	for (int i = 0; i < TFLITE_HEIGHT * TFLITE_WIDTH; ++i) {
		if (filter->output_probability[i] < filter->mask_value) {
			*(filter->texturedata + i) = 0;
		} else {
			*(filter->texturedata + i) = (uint8_t) (255.0f * filter->output_probability[i]);
		}
	}

	return frame;
}
static void tflite_get_out(struct background_mask_filter_data * filter) {
	TfLiteTensorCopyFromBuffer(filter->input_tensor, filter->rgb_f,
				   TFLITE_HEIGHT * filter->rgb_linesize * sizeof(float));

	TfLiteInterpreterInvoke(filter->interpreter);

	const TfLiteTensor *output_tensor =
		TfLiteInterpreterGetOutputTensor(filter->interpreter, 0);
	if (!filter->output_probability) {
		filter->output_probability = (
			bzalloc(TFLITE_HEIGHT * TFLITE_WIDTH * sizeof(float)));
	}
	if (!filter->texturedata) {
		filter->texturedata = bzalloc(TFLITE_HEIGHT * TFLITE_WIDTH * sizeof(uint8_t));
	}
	TfLiteTensorCopyToBuffer(output_tensor, filter->output_probability,
				 TFLITE_HEIGHT * TFLITE_WIDTH * sizeof(float));
}

static void mirror_inversion(struct obs_source_frame *frame)
{
	if (!frame->width) {
		return;
	}
	uint32_t line_size;
	switch (frame->format) {
	case VIDEO_FORMAT_UYVY:
		line_size = frame->linesize[0];
		uint8_t * data = frame->data[0];
		uint32_t *p;
		uint32_t *q;
		uint32_t *frame_data = frame->data[0];
		uint32_t temp_32;
		uint8_t temp_8;
		for (uint32_t i = 0; i < frame->height; i++) {
			for (uint32_t j = 0; j < frame->width / 4; ++j) {
				p = frame_data + frame->width / 2 * i + j;
				q = frame_data + frame->width / 2 * i + frame->width / 2 - 1 - j;
				temp_32 = *p;
				*p = *q;
				*q = temp_32;
				data = (uint8_t *)(p);
				temp_8 = *(data + 1);
				*(data + 1) = *(data + 3);
				*(data + 3) = temp_8;

				if (data != q) {
					data = (uint8_t *)(q);
					temp_8 = *(data + 1);
					*(data + 1) = *(data + 3);
					*(data + 3) = temp_8;
				}
			}
		}
		break;
	default:
		break;
	}
}

static void background_mask_tick(void *data, float seconds)
{
	struct background_mask_filter_data *filter = data;

	obs_enter_graphics();

	obs_leave_graphics();
}

static void chroma_key_update_v2(void *data, obs_data_t *settings)
{
	struct background_mask_filter_data *filter = data;
	filter->mask_value = obs_data_get_double(settings, "SETTING_MASK");
}

static obs_properties_t *chroma_key_properties_v2(void *data)
{
	obs_properties_t *props = obs_properties_create();
	obs_properties_add_float_slider(props, "SETTING_MASK", "MASK_VALUE",
					0.0, 1.0, 0.0001);
	UNUSED_PARAMETER(data);
	return props;
}

static void chroma_key_defaults_v2(obs_data_t *settings)
{
	obs_data_set_default_double(settings, "SETTING_MASK", 0.8);
}

struct obs_source_info background_mask_filter = {
	.id = "background-mask-filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC,
	.get_name = background_mask_name,
	.create = background_mask_create,
	.destroy = background_mask_destroy,
	.video_render = background_mask_render,
	.filter_video = background_mask_video,
	.update = chroma_key_update_v2,
	.get_properties = chroma_key_properties_v2,
	.get_defaults = chroma_key_defaults_v2,
};
