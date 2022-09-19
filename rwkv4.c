#define _Frees_ptr_opt_ // Pointer parameters that are freed by the function, and thus the pointed-to memory should not be used after return.
#define _Outptr_result_buffer_maybenull_(X)

#include <onnxruntime_c_api.h>
#include <stdint.h>
#include <stdio.h>
#include <rwkv4.h>

extern const OrtApi* g_ort;

#include <ort_abort_on_error.h>

void detect_dimensions(OrtSession* session, int64_t* idx_shape, int64_t* state_shape) {
	size_t input_count = 0;
	ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &input_count));

	if (input_count != 6) {
		printf("Not an RWKV model (Wrong input count)\n");
		exit(-1);
	}

	OrtTypeInfo* idx_input_info = NULL;
	OrtTypeInfo* xx_att_input_info = NULL;

	ORT_ABORT_ON_ERROR(g_ort->SessionGetInputTypeInfo(session, 0, &idx_input_info));
	ORT_ABORT_ON_ERROR(g_ort->SessionGetInputTypeInfo(session, 1, &xx_att_input_info));

	const OrtTensorTypeAndShapeInfo* idx_shape_info = NULL;
	const OrtTensorTypeAndShapeInfo* xx_att_shape_info = NULL;

	// Valid until TypeInfo is released
	ORT_ABORT_ON_ERROR(g_ort->CastTypeInfoToTensorInfo(idx_input_info, &idx_shape_info));
	ORT_ABORT_ON_ERROR(g_ort->CastTypeInfoToTensorInfo(xx_att_input_info, &xx_att_shape_info));

	size_t state_dim = 0;

	ORT_ABORT_ON_ERROR(g_ort->GetDimensionsCount(xx_att_shape_info, &state_dim));

	if (state_dim != 2) {
		printf("Not an RWKV model (Wrong input dimensionality)\n");
		exit(-1);
	}

	ORT_ABORT_ON_ERROR(g_ort->GetDimensions(idx_shape_info, idx_shape, 1));
	ORT_ABORT_ON_ERROR(g_ort->GetDimensions(xx_att_shape_info, state_shape, 2));

	g_ort->ReleaseTypeInfo(idx_input_info);
	g_ort->ReleaseTypeInfo(xx_att_input_info);
}
