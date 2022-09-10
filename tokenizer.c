#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include <tokenizer.h>
#include <wyhash.h>

static	inline	size_t	wyhashmap(uint64_t	*idx,	size_t	idx_size,	const	void *key, size_t	key_size,	uint8_t	insert){
	size_t	i=1;	uint64_t	h2;	uint64_t	sig;
	do{	sig=h2=wyhash(key,key_size,i,_wyp);	i++;	}while(_unlikely_(!sig));
	size_t	i0=wy2u0k(wyhash(key,key_size,0,_wyp),idx_size);
	for(i=i0;	i<idx_size&&idx[i]&&idx[i]!=sig;	i++);
	if(_unlikely_(i==idx_size)){
		for(i=0;	i<i0&&idx[i]&&idx[i]!=sig;  i++);
		if(i==i0)	return	idx_size;
	}
	if(!idx[i]){
		if(insert)	idx[i]=sig;
		else	return	idx_size;
	}
	return	i;
}

void add_token(dict_t* dict, uint8_t* str, size_t len) {
	size_t idx = wyhashmap(dict->idx, 50277, str, len, 1);

	if (idx >= 50277) {
		fprintf(stderr, "Hashmap overflow\n");
		exit(1);
	}

	if (dict->list_c[idx] != NULL) {
		//printf("Hashmap overwrite: token %u overwrites %u\n", dict->count, dict->list_b[idx]);
		free(dict->list_c[idx]);
	}

	dict->list_a[dict->count] = idx;
	dict->list_b[idx] = dict->count++;
	dict->list_c[idx] = malloc(len + 1);
	dict->list_c[idx][len] = 0;

	memcpy(dict->list_c[idx], str, len);
}

// Derived from gpt2tc
// https://bellard.org/libnc/gpt2tc-2021-04-24.tar.gz
// #########################################################################################################

void load_vocab(const char* filename, dict_t* dict) {
	FILE* fd = fopen(filename, "rb");

	if (!fd) {
		fprintf(stderr, "Unable to open %s\n", filename);
		exit(1);
	}

	uint8_t buf[1024];
	size_t len = 0;
	int c;

	fgetc(fd); // [
	fgetc(fd); // \n

	while (1) {
		if (len == 0) {
			c = fgetc(fd);

			if (c == ']') {
				// End of file
				break;
			} else if (c != '"') {
				fprintf(stderr, "Format breakage. Perhaps because of CRLF line endings?\n");
				exit(1);
			}
		}

		c = fgetc(fd);

		if (c < 0)
			break;

		if (c == '"') {
			if (fgetc(fd) != ',') {
				fprintf(stderr, "Format breakage.\n");
				exit(1);
			}

			if (fgetc(fd) != '\n') {
				fprintf(stderr, "Format breakage.\n");
				exit(1);
			}

			if (len > 0) {
				add_token(dict, buf, len);
			}

			len = 0;
		} else {
			if (c == '\\') {
				c = fgetc(fd);

				if (c < 0)
					break;

				if (c == 'n') {
					c = '\n';
				} else if (c == '"') {
					// No op
				} else if (c == 'u') {
					int code = 0;

					fscanf(fd, "%4x", &code);

					if (code > 255) {
						fprintf(stderr, "No proper handling implemented for \\u%04x.\n", code);
						exit(-1);
					}

					c = code & 0xFF;
				} else if (c != '\\') {
					fprintf(stderr, "Invalid escape\n");
					exit(1);
				}
			}

			if (len >= sizeof(buf)) {
				fprintf(stderr, "Word too long %ld\n", ftell(fd));
				exit(1);
			}

			buf[len++] = c;
		}
	}
}

typedef enum {
	CAT_SPACE,
	CAT_LETTER,
	CAT_NUMBER,
	CAT_OTHER,
} CharCatEnum;

static int get_char_cat(int c) {
	if (c == ' ') {
		return CAT_SPACE;
	} else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 128) {
		return CAT_LETTER;
	} else if (c >= '0' && c <= '9') {
		return CAT_NUMBER;
	} else {
		return CAT_OTHER;
	}
}

static int match(size_t *pmatch_len, const uint8_t *buf, size_t buf_len, const char *str) {
	size_t len = strlen(str);

	if (len <= buf_len && !memcmp(buf, str, len)) {
		*pmatch_len = len;
		return 1;
	} else {
		*pmatch_len = 0;
		return 0;
	}
}

static size_t gpt2_get_word(const uint8_t *buf, size_t buf_len) {
	size_t len, p;
	int cat;

	if (buf_len == 0)
		return 0;

	// Use /'\w/g to find cases like that
	if (buf[0] == '\'' &&
		(match(&len, buf, buf_len, "'s") ||
		match(&len, buf, buf_len, "'t") ||
		match(&len, buf, buf_len, "'re") ||
		match(&len, buf, buf_len, "'ve") ||
		match(&len, buf, buf_len, "'m") ||
		match(&len, buf, buf_len, "'ll") ||
		match(&len, buf, buf_len, "'d"))) {
		return len;
	}
	p = 0;

	// Handling of long sequences of spaces
	while (buf[p] == ' ' && buf_len >= (p+2))
		p++;

	if (buf[p] != ' ') {
		cat = get_char_cat(buf[p]);
		len = 1 + p;
		while (len < buf_len && get_char_cat(buf[len]) == cat)
			len++;
		return len;
	} else {
		return 1;
	}
}

size_t tokenize(uint16_t* buf, size_t elements, const uint8_t* str, dict_t* dict) {
	size_t str_len = strlen(str);
	size_t word_len;
	size_t len = 0; // Token length
	size_t idx;
	size_t buf_i = 0;
	size_t offset;

	for(offset = 0; offset < str_len; offset += word_len) {
		word_len = gpt2_get_word(str + offset, str_len - offset);

		// find the longest word(s)
		for(size_t i = 0; i < word_len; i += len) {
			for(len = word_len - i; len >= 1; len--) {
				idx = wyhashmap(dict->idx, 50277, str + offset + i, len, 0);

				if (idx < 50277 && dict->list_c[idx])
					break;
			}

			assert(len >= 1);

			buf[buf_i] = dict->list_b[idx];
			buf_i++;

			if (buf_i == elements) {
				return offset;
			}
		}
	}

	return offset;
}

// #########################################################################################################

dict_t* init_dict() {
	dict_t* dict = malloc(sizeof(dict_t));

	dict->idx = malloc(8*50277);
	dict->list_a = malloc(2*50277);
	dict->list_b = malloc(2*50277);
	dict->list_c = malloc(8*50277);
	dict->count = 0;

	memset(dict->list_a, 0, 2*50277);
	memset(dict->list_b, 0, 2*50277);
	memset(dict->list_c, 0, 8*50277);

	load_vocab("20B_edited.json", dict);

	return dict;
}

void free_dict(dict_t* dict) {
	free(dict->idx);
	free(dict->list_a);
	free(dict->list_b);
	free(dict->list_c);
	free(dict);
}

void test() {
	dict_t* dict = init_dict();

	printf("Elements: %u\n", dict->count);

	const char* prompt = ""
	"static void word_list_end(WordList *s)\n"
	"{\n"
	"    int i;\n"
	"    Word *p;\n"
	"    \n"
	"    for(i = 0; i < s->word_count; i++) {\n"
	"        p = &s->words[i];\n"
	"        free(p->buf);\n"
	"    }\n"
	"    free(s->words);\n"
	"    free(s->hash_table);\n"
	"    free(s);\n"
	"}";

	uint16_t buf[256] = {0};

	tokenize(buf, 256, prompt, dict);

	for (int i = 0; buf[i] != 0; i++)
		printf("%u ", buf[i]);

	free_dict(dict);
}
