// gcc.exe -O2 -Wall -Wextra main.c -lonnxruntime -o main.c.exe
// Library and headers from: https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/Microsoft.ML.OnnxRuntime.DirectML.1.12.1.zip

#define _Frees_ptr_opt_ // Pointer parameters that are freed by the function, and thus the pointed-to memory should not be used after return.
#define _Outptr_result_buffer_maybenull_(X)

#include <stdio.h>
#include <wchar.h>
#include <stdint.h>
#include <onnxruntime_c_api.h>
#include <tokenizer.h>
//#include <dml_provider_factory.h>
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_DML, _In_ OrtSessionOptions* options, int device_id);

// https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/fns_candy_style_transfer/fns_candy_style_transfer.c
const OrtApi* g_ort = NULL;

// Does DML report its errors in 4-byte characters? It's either that or it doesn't report anything at all in this build.
#define ORT_ABORT_ON_ERROR(expr)																	\
	do {																							\
		OrtStatus* onnx_status = (expr);															\
		if (onnx_status != NULL) {																	\
			const char* msg = g_ort->GetErrorMessage(onnx_status);									\
			const int code = g_ort->GetErrorCode(onnx_status);										\
			fprintf(stderr, "[%s:%d] Aborting on error %d: %s\n", __FILE__, __LINE__, code, msg);	\
			g_ort->ReleaseStatus(onnx_status);														\
			exit(-1);																				\
		}																							\
	} while (0);

int greedy_sampling(float* x) {
	float max_v = x[0];
	int max_k = 0;

	for (int i = 0; i < 50277; i++) {
		if (x[i] > max_v) {
			max_v = x[i];
			max_k = i;
		}
	}

	return max_k;
}

void usage() {
	printf("Usage: onnxrwkv.exe model_name.onnx [dml | cpu (default)] [verbose]\n");
	exit(-1);
}

int main(int argc, char* argv[]) {
	dict_t* dict = init_dict();

	g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

	if (!g_ort) {
		printf("Failed to init ONNX Runtime engine.\n");
		return -1;
	}

	OrtLoggingLevel loglevel = ORT_LOGGING_LEVEL_WARNING;

	if (argc < 2) {
		usage();
	}

	if (argc > 3) {
		if (strcmp(argv[3], "verbose") == 0) {
			printf("Verbose logging enabled\n");
			loglevel = ORT_LOGGING_LEVEL_VERBOSE;
		}
	}

	OrtEnv* env;
	ORT_ABORT_ON_ERROR(g_ort->CreateEnv(loglevel, "test", &env));

	OrtSessionOptions* session_options;
	ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
	//ORT_ABORT_ON_ERROR(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_DISABLE_ALL));

	if (argc > 2) {
		if (strcmp(argv[2], "dml") == 0) {
			printf("DirectML enabled\n");
			ORT_ABORT_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
		}
	}

	const char* filename = argv[1];
	const int len = strlen(filename) + 1;

	ORTCHAR_T* model_path = malloc(len*4);

	size_t ret = mbsrtowcs(model_path, &filename, len, NULL);

	printf("Filename conversion: %I64u\n", ret);

	OrtSession* session;
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));

	size_t input_count = 0;

	ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &input_count));

	if (input_count != 6) {
		printf("Not an RWKV model (Wrong input count)\n");
		exit(-1);
	}

	OrtMemoryInfo* memory_info;
	ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

	const char* input_names[] = {"idx", "xx_att", "aa_att", "bb_att", "pp_att", "xx_ffn"};
	const char* output_names[] = {"x", "xx_att_r", "aa_att_r", "bb_att_r", "pp_att_r", "xx_ffn_r"};

	OrtTypeInfo* idx_input_info = NULL;
	OrtTypeInfo* xx_att_input_info = NULL;

	ORT_ABORT_ON_ERROR(g_ort->SessionGetInputTypeInfo(session, 0, &idx_input_info));
	ORT_ABORT_ON_ERROR(g_ort->SessionGetInputTypeInfo(session, 1, &xx_att_input_info));

	const OrtTensorTypeAndShapeInfo* idx_shape_info = NULL;
	const OrtTensorTypeAndShapeInfo* xx_att_shape_info = NULL;

	ORT_ABORT_ON_ERROR(g_ort->CastTypeInfoToTensorInfo(idx_input_info, &idx_shape_info));
	ORT_ABORT_ON_ERROR(g_ort->CastTypeInfoToTensorInfo(xx_att_input_info, &xx_att_shape_info));

	size_t state_dim = 0;

	ORT_ABORT_ON_ERROR(g_ort->GetDimensionsCount(xx_att_shape_info, &state_dim));

	if (state_dim != 2) {
		printf("Not an RWKV model (Wrong input dimensionality)\n");
		exit(-1);
	}

	int64_t idx_shape[] = {0};
	int64_t state_shape[] = {0, 0};

	ORT_ABORT_ON_ERROR(g_ort->GetDimensions(idx_shape_info, idx_shape, 1));
	ORT_ABORT_ON_ERROR(g_ort->GetDimensions(xx_att_shape_info, state_shape, 2));

	printf("Autodetected model parameters:\n");
	printf(" ctx_len: %I64u\n", idx_shape[0]);
	printf(" n_layer: %I64u\n", state_shape[0]);
	printf(" n_embd: %I64u\n", state_shape[1]);

	const size_t idx_d_len = idx_shape[0] * sizeof(int32_t);
	const size_t state_d_len = state_shape[0]*state_shape[1] * sizeof(float);

	int32_t* idx_d = malloc(idx_d_len);
	float* xx_att_d = malloc(state_d_len);
	float* aa_att_d = malloc(state_d_len);
	float* bb_att_d = malloc(state_d_len);
	float* pp_att_d = malloc(state_d_len);
	float* xx_ffn_d = malloc(state_d_len);

	for (int i = 0; i < state_shape[0]*state_shape[1]; i++) {
		xx_att_d[i] = 0;
		aa_att_d[i] = 0;
		bb_att_d[i] = 0;
		pp_att_d[i] = -1e30;
		xx_ffn_d[i] = 0;
	}

	for (int i = 0; i < idx_shape[0]; i++)
		idx_d[i] = 0;

	uint16_t prompt_d[1024] = {0};
	uint16_t* prompt = prompt_d;

	tokenize(prompt, 1023, "\nIn a shocking finding", dict);

	OrtValue* idx = NULL;
	OrtValue* xx_att = NULL;
	OrtValue* aa_att = NULL;
	OrtValue* bb_att = NULL;
	OrtValue* pp_att = NULL;
	OrtValue* xx_ffn = NULL;

	ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, idx_d, idx_d_len, idx_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &idx));
	ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, xx_att_d, state_d_len, state_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &xx_att));
	ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, aa_att_d, state_d_len, state_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &aa_att));
	ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, bb_att_d, state_d_len, state_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &bb_att));
	ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, pp_att_d, state_d_len, state_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pp_att));
	ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, xx_ffn_d, state_d_len, state_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &xx_ffn));

	const OrtValue* input_list[] = { idx, xx_att, aa_att, bb_att, pp_att, xx_ffn };

	OrtValue* x = NULL;
	OrtValue* xx_att_r = NULL;
	OrtValue* aa_att_r = NULL;
	OrtValue* bb_att_r = NULL;
	OrtValue* pp_att_r = NULL;
	OrtValue* xx_ffn_r = NULL;

	OrtValue* output_list[] = { x, xx_att_r, aa_att_r, bb_att_r, pp_att_r, xx_ffn_r };


	idx_d[1023] = *prompt;
	prompt++;

	for (int i = 0; i < 1024; i++) {
		ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names, input_list, 6, output_names, 6, output_list));

		input_list[1] = output_list[1];
		input_list[2] = output_list[2];
		input_list[3] = output_list[3];
		input_list[4] = output_list[4];
		input_list[5] = output_list[5];

		float* xx;
		ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_list[0], (void**)&xx));

		int token = greedy_sampling(xx);

		if (*prompt == 0) {
			printf("%s", dict->list_c[dict->list_a[token]]);
			idx_d[1023] = token;
		} else {
			printf("%s", dict->list_c[dict->list_a[*prompt]]);
			idx_d[1023] = *prompt;
			prompt++;
		}
	}

	printf("\n");

	printf("Releasing memory...\n");
	g_ort->ReleaseTypeInfo(idx_input_info);
	g_ort->ReleaseTypeInfo(xx_att_input_info);
	g_ort->ReleaseSessionOptions(session_options);
	g_ort->ReleaseSession(session);
	g_ort->ReleaseEnv(env);
	printf("Done\n");

	return 0;
}

