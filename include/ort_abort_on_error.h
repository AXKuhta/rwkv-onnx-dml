
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
