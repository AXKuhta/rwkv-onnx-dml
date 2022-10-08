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
	FILE* emb_f = open_emb();

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

	OrtThreadingOptions* thread_options;
	ORT_ABORT_ON_ERROR(g_ort->CreateThreadingOptions(&thread_options));

	OrtEnv* env;
	ORT_ABORT_ON_ERROR(g_ort->CreateEnvWithGlobalThreadPools(loglevel, "test", thread_options, &env));

	OrtSessionOptions* session_options;
	ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
	ORT_ABORT_ON_ERROR(g_ort->DisablePerSessionThreads(session_options));
	//ORT_ABORT_ON_ERROR(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_DISABLE_ALL));

	OrtSession* session[24];
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.0.onnx", session_options, &session[0]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.1.onnx", session_options, &session[1]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.2.onnx", session_options, &session[2]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.3.onnx", session_options, &session[3]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.4.onnx", session_options, &session[4]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.5.onnx", session_options, &session[5]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.6.onnx", session_options, &session[6]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.7.onnx", session_options, &session[7]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.8.onnx", session_options, &session[8]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.9.onnx", session_options, &session[9]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.10.onnx", session_options, &session[10]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.11.onnx", session_options, &session[11]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.12.onnx", session_options, &session[12]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.13.onnx", session_options, &session[13]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.14.onnx", session_options, &session[14]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.15.onnx", session_options, &session[15]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.16.onnx", session_options, &session[16]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.17.onnx", session_options, &session[17]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.18.onnx", session_options, &session[18]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.19.onnx", session_options, &session[19]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.20.onnx", session_options, &session[20]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.21.onnx", session_options, &session[21]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.22.onnx", session_options, &session[22]));
	ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "rwkv.23.onnx", session_options, &session[23]));

	int64_t emb_shape[] = {1024};
	int64_t state_shape[] = {1024};

	const char* input_names[] = {"emb", "xx_att", "aa_att", "bb_att", "pp_att", "xx_ffn"};
	const char* output_names[] = {"x", "xx_att_r", "aa_att_r", "bb_att_r", "pp_att_r", "xx_ffn_r"};

	const size_t emb_d_len = emb_shape[0] * sizeof(float);
	const size_t state_d_len = state_shape[0] * sizeof(float);

	float* emb_d = malloc(emb_d_len);
	float* xx_att_d[24];
	float* aa_att_d[24];
	float* bb_att_d[24];
	float* pp_att_d[24];
	float* xx_ffn_d[24];

	for (int i = 0; i < 24; i++) {
                xx_att_d[i] = malloc(state_d_len);
                aa_att_d[i] = malloc(state_d_len);
                bb_att_d[i] = malloc(state_d_len);
                pp_att_d[i] = malloc(state_d_len);
                xx_ffn_d[i] = malloc(state_d_len);

		for (int j = 0; j < state_shape[0]; j++) {
			xx_att_d[i][j] = 0;
			aa_att_d[i][j] = 0;
			bb_att_d[i][j] = 0;
			pp_att_d[i][j] = -1e30;
			xx_ffn_d[i][j] = 0;
		}
	}

	for (int i = 0; i < emb_shape[0]; i++)
		emb_d[i] = 0;

	OrtValue* emb = NULL;
	OrtValue* xx_att[24];
	OrtValue* aa_att[24];
	OrtValue* bb_att[24];
	OrtValue* pp_att[24];
	OrtValue* xx_ffn[24];

	OrtMemoryInfo* memory_info;
	ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
	ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, emb_d, emb_d_len, emb_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &emb));

	for (int i = 0; i < 24; i++) {
		ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, xx_att_d[i], state_d_len, state_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &xx_att[i]));
		ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, aa_att_d[i], state_d_len, state_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &aa_att[i]));
		ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, bb_att_d[i], state_d_len, state_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &bb_att[i]));
		ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, pp_att_d[i], state_d_len, state_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pp_att[i]));
		ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, xx_ffn_d[i], state_d_len, state_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &xx_ffn[i]));
	}

	uint16_t prompt_d[1024] = {0};
	uint16_t* prompt = prompt_d;

	tokenize(prompt, 1023, "\nIn a shocking finding", dict);

	int token = *prompt;
	prompt++;

	OrtValue* x = emb;

	uint64_t timestamps[1024];

	for (int i = 0; i < 16; i++) {
		uint64_t time_a = microseconds();
		read_emb(emb_f, token, emb_d);
		x = emb;

		printf(" [00/24]");

		for (int j = 0; j < 24; j++) {
			const OrtValue* input_list[6] = { x, xx_att[j], aa_att[j], bb_att[j], pp_att[j], xx_ffn[j] };
			OrtValue* output_list[6] = { NULL }; // Make sure output_list is zeroed or else onnxruntime will use its values to do output shape checking

			ORT_ABORT_ON_ERROR(g_ort->Run(session[j], NULL, input_names, input_list, 6, output_names, 6, output_list));

			xx_att[j] = output_list[1];
                        aa_att[j] = output_list[2];
                        bb_att[j] = output_list[3];
                        pp_att[j] = output_list[4];
                        xx_ffn[j] = output_list[5];

			x = output_list[0];

			printf("\x7F\x7F\x7F\x7F\x7F\x7F\x7F\x7F");
			printf(" [%02d/24]", j+1);
			fflush(stdout);
		}

		uint64_t time_b = microseconds();

		timestamps[i] = time_b - time_a;

		float* xx;
		ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(x, (void**)&xx));

		token = greedy_sampling(xx);
		printf("\x7F\x7F\x7F\x7F\x7F\x7F\x7F\x7F");

		if (*prompt == 0) {
			printf("%s", dict->list_c[dict->list_a[token]]);
		} else {
			printf("%s", dict->list_c[dict->list_a[*prompt]]);
			token = *prompt;
			prompt++;
		}
	}

	printf("\n");
	report_performance(timestamps, 16);

	printf("Releasing memory...\n");
	g_ort->ReleaseSessionOptions(session_options);
	for (int j = 0; j < 24; j++) g_ort->ReleaseSession(session[j]);
	g_ort->ReleaseEnv(env);
	printf("Done\n");

	fclose(emb_f);

	return 0;
}

