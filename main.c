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

	int n_layer, n_embd, ctx_len;

	detect_dimensions(&n_layer, &n_embd, &ctx_len);

	dict_t* dict = init_dict();
	FILE* emb_f = open_emb();

	OrtThreadingOptions* thread_options;
	ORT_ABORT_ON_ERROR(g_ort->CreateThreadingOptions(&thread_options));

	OrtEnv* env;
	ORT_ABORT_ON_ERROR(g_ort->CreateEnvWithGlobalThreadPools(loglevel, "test", thread_options, &env));

	OrtSessionOptions* session_options;
	ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
	ORT_ABORT_ON_ERROR(g_ort->DisablePerSessionThreads(session_options));
	//ORT_ABORT_ON_ERROR(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_DISABLE_ALL));

	OrtSession** session = malloc( n_layer * sizeof(OrtSession*) );

	// Load all layers
	for (int i = 0; i < n_layer; i++) {
		char* filename = malloc(128);
		ORTCHAR_T* model_path;

		snprintf(filename, 128, "rwkv.%d.onnx", i);

		#ifndef _WIN32
			model_path = filename;
		#else
			// Windows wants filename in wide chars
			const int len = strlen(filename) + 1;
			model_path = malloc(len*4);

			size_t ret = mbsrtowcs(model_path, &filename, len, NULL);
		#endif

		ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session[i]));
	}

	int64_t state_shape[] = { n_embd };

	const char* input_names[] = {"emb", "xx_att", "aa_att", "bb_att", "pp_att", "xx_ffn"};
	const char* output_names[] = {"x", "xx_att_r", "aa_att_r", "bb_att_r", "pp_att_r", "xx_ffn_r"};

	const size_t state_bsz = n_embd * sizeof(float);

	float* emb_d = malloc(state_bsz);
	float** xx_att_d = malloc( n_layer * sizeof(float*) );
	float** aa_att_d = malloc( n_layer * sizeof(float*) );
	float** bb_att_d = malloc( n_layer * sizeof(float*) );
	float** pp_att_d = malloc( n_layer * sizeof(float*) );
	float** xx_ffn_d = malloc( n_layer * sizeof(float*) );

	for (int i = 0; i < n_layer; i++) {
                xx_att_d[i] = malloc(state_bsz);
                aa_att_d[i] = malloc(state_bsz);
                bb_att_d[i] = malloc(state_bsz);
                pp_att_d[i] = malloc(state_bsz);
                xx_ffn_d[i] = malloc(state_bsz);

		for (int j = 0; j < n_embd; j++) {
			xx_att_d[i][j] = 0;
			aa_att_d[i][j] = 0;
			bb_att_d[i][j] = 0;
			pp_att_d[i][j] = -1e30;
			xx_ffn_d[i][j] = 0;
		}
	}

	for (int i = 0; i < n_embd; i++)
		emb_d[i] = 0;

	OrtValue* emb = NULL;
	OrtValue** xx_att = malloc( n_layer * sizeof(OrtValue*) );
	OrtValue** aa_att = malloc( n_layer * sizeof(OrtValue*) );
	OrtValue** bb_att = malloc( n_layer * sizeof(OrtValue*) );
	OrtValue** pp_att = malloc( n_layer * sizeof(OrtValue*) );
	OrtValue** xx_ffn = malloc( n_layer * sizeof(OrtValue*) );

	OrtMemoryInfo* memory_info;
	ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
	ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, emb_d, state_bsz, state_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &emb));

	for (int i = 0; i < n_layer; i++) {
		ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, xx_att_d[i], state_bsz, state_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &xx_att[i]));
		ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, aa_att_d[i], state_bsz, state_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &aa_att[i]));
		ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, bb_att_d[i], state_bsz, state_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &bb_att[i]));
		ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, pp_att_d[i], state_bsz, state_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pp_att[i]));
		ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, xx_ffn_d[i], state_bsz, state_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &xx_ffn[i]));
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

		fseek(emb_f, state_bsz*token, SEEK_SET);
		fread(emb_d, state_bsz, 1, emb_f);
		x = emb;

		printf(" [00/%d]", n_layer);

		for (int j = 0; j < n_layer; j++) {
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
			printf(" [%02d/%d]", j+1, n_layer);
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
	for (int j = 0; j < n_layer; j++) g_ort->ReleaseSession(session[j]);
	g_ort->ReleaseEnv(env);
	printf("Done\n");

	fclose(emb_f);

	return 0;
}

