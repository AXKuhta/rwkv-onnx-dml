
void detect_dimensions(OrtSession* session, int64_t* idx_shape, int64_t* state_shape);
void read_emb(FILE* file, int token, float* data);
FILE* open_emb();
