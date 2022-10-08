#define _Frees_ptr_opt_ // Pointer parameters that are freed by the function, and thus the pointed-to memory should not be used after return.
#define _Outptr_result_buffer_maybenull_(X)

#include <onnxruntime_c_api.h>
#include <stdio.h>
#include <wchar.h>
#include <stdint.h>
#include <performance.h>
#include <tokenizer.h>
#include <rwkv4.h>

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_DML, _In_ OrtSessionOptions* options, int device_id);

// https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/fns_candy_style_transfer/fns_candy_style_transfer.c
const OrtApi* g_ort = NULL;

#include <ort_abort_on_error.h>

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

	#ifndef _WIN32
		// DirectML not available on Linux
	#else
		if (argc > 2) {
			if (strcmp(argv[2], "dml") == 0) {
				printf("DirectML enabled\n");
				ORT_ABORT_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
			}
		}
	#endif

	const char* filename = argv[1];
	ORTCHAR_T* model_path;

	#ifndef _WIN32
		model_path = filename;
	#else
		// Windows wants filename in wide chars
		const int len = strlen(filename) + 1;
		model_path = malloc(len*4);

		size_t ret = mbsrtowcs(model_path, &filename, len, NULL);
		printf("Filename conversion: %I64u\n", ret);
	#endif

	OrtSession* session;
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));

	int64_t idx_shape[] = {0};
	int64_t state_shape[] = {0, 0};

	detect_dimensions(session, idx_shape, state_shape);

	printf("Autodetected model parameters:\n");
	printf(" ctx_len: %I64u\n", idx_shape[0]);
	printf(" n_layer: %I64u\n", state_shape[0]);
	printf(" n_embd: %I64u\n", state_shape[1]);

	const char* input_names[] = {"idx", "xx_att", "aa_att", "bb_att", "pp_att", "xx_ffn"};
	const char* output_names[] = {"x", "xx_att_r", "aa_att_r", "bb_att_r", "pp_att_r", "xx_ffn_r"};

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

	OrtValue* idx = NULL;
	OrtValue* xx_att = NULL;
	OrtValue* aa_att = NULL;
	OrtValue* bb_att = NULL;
	OrtValue* pp_att = NULL;
	OrtValue* xx_ffn = NULL;

	OrtMemoryInfo* memory_info;
	ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

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


	uint16_t prompt_d[1024] = {0};
	uint16_t* prompt = prompt_d;

	tokenize(prompt, 1023, "\nIn a shocking finding", dict);

	idx_d[1023] = *prompt;
	prompt++;

	uint64_t timestamps[1024];

	for (int i = 0; i < 16; i++) {
		uint64_t time_a = microseconds();
		ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names, input_list, 6, output_names, 6, output_list));
		uint64_t time_b = microseconds();

		timestamps[i] = time_b - time_a;

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

		fflush(stdout);
	}

	printf("\n");
	report_performance(timestamps, 16);

	printf("Releasing memory...\n");
	g_ort->ReleaseSessionOptions(session_options);
	g_ort->ReleaseSession(session);
	g_ort->ReleaseEnv(env);
	printf("Done\n");

	return 0;
}

