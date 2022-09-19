
typedef struct dict_t {
	uint64_t* idx;
	uint16_t* list_a; // Token ID -> Hashmap ID
	uint16_t* list_b; // Hashmap ID -> Token ID
	uint8_t** list_c; // Strings
	uint16_t count;
} dict_t;

size_t tokenize(uint16_t* buf, size_t elements, const uint8_t* str, dict_t* dict);
dict_t* init_dict();
void free_dict(dict_t* dict);
