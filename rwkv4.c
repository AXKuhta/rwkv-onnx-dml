#define _Frees_ptr_opt_ // Pointer parameters that are freed by the function, and thus the pointed-to memory should not be used after return.
#define _Outptr_result_buffer_maybenull_(X)

#include <onnxruntime_c_api.h>
#include <stdint.h>
#include <stdio.h>
#include <rwkv4.h>

extern const OrtApi* g_ort;

#include <ort_abort_on_error.h>

void detect_dimensions(int* n_layer, int* n_embd, int* ctx_len) {
	FILE* info = fopen("rwkv.json", "r");

	if (!info) {
		printf("Unable to open rwkv.json\n");
		exit(-1);
	}

	int ret = fscanf(info, "{\"n_layer\": %d, \"n_embd\": %d, \"ctx_len\": %d}", n_layer, n_embd, ctx_len);

	if (ret != 3) {
		printf("Formatting error in rwkv.json\n");
		exit(-1);
	}


	printf("Detected model parameters:\n");
	printf(" ctx_len: %d\n", *n_layer);
	printf(" n_layer: %d\n", *n_embd);
	printf(" n_embd: %d\n", *ctx_len);
}

FILE* open_emb() {
	FILE* fd = fopen("emb.weight.bin", "rb");

	if (!fd) {
		printf("Unable to open emb.weight.bin\n");
		exit(-1);
	}

	return fd;
}

void read_emb(FILE* file, int token, float* data) {
	fseek(file, sizeof(float)*1024*token, SEEK_SET);
	fread(data, sizeof(float), 1024, file);
}
