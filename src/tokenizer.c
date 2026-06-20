#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#include "../include/hashmap.h"
#include "../include/tokenizer.h"
#include "../include/allocator.h"
#include "../include/header.h"

/* Byte-to-Unicode lookup tables used by the tokenizer's BPE preprocessing. */
static const char *bytes_to_unicode_table[256];
static char bytes_to_unicode_buf[256][TOKENIZER_MAX_UTF8_BYTES];
static size_t bytes_to_unicode_len[256];

/* Append a Unicode code point as UTF-8 into dst. */
static void append_utf8(char *dst, size_t *pos, uint32_t cp) {
    if (cp <= 0x7F) {
        dst[(*pos)++] = (char)cp;
    } else if (cp <= 0x7FF) {
        dst[(*pos)++] = (char)(0xC0 | (cp >> 6));
        dst[(*pos)++] = (char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        dst[(*pos)++] = (char)(0xE0 | (cp >> 12));
        dst[(*pos)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        dst[(*pos)++] = (char)(0x80 | (cp & 0x3F));
    } else {
        dst[(*pos)++] = (char)(0xF0 | (cp >> 18));
        dst[(*pos)++] = (char)(0x80 | ((cp >> 12) & 0x3F));
        dst[(*pos)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        dst[(*pos)++] = (char)(0x80 | (cp & 0x3F));
    }
}

/* Build the byte-to-Unicode mapping used by the tokenizer. */
static void build_bytes_to_unicode(void) {
    int bs[256];
    int cs[256];
    int used[256] = {0};
    int n = 0;

    for (int i = 33;  i <= 126; i++) bs[n++] = i;
    for (int i = 161; i <= 172; i++) bs[n++] = i;
    for (int i = 174; i <= 255; i++) bs[n++] = i;

    for (int i = 0; i < n; i++)
        used[bs[i]] = 1;

    int cur = 256;
    for (int b = 0; b < 256; b++)
        cs[b] = used[b] ? b : cur++;

    for (int b = 0; b < 256; b++) {
        size_t pos = 0;
        memset(bytes_to_unicode_buf[b], 0, sizeof(bytes_to_unicode_buf[b]));
        append_utf8(bytes_to_unicode_buf[b], &pos, (uint32_t)cs[b]);
        bytes_to_unicode_buf[b][pos] = '\0';
        bytes_to_unicode_len[b] = pos;
        bytes_to_unicode_table[b] = bytes_to_unicode_buf[b];
    }
}

/* Return true if c is an ASCII whitespace character. */
static int is_space_byte(unsigned char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f';
}

/* Concatenate two strings into a newly allocated buffer. */
static char *concat2(const char *a, const char *b) {
    size_t na = strlen(a);
    size_t nb = strlen(b);
    char *out = (char *)malloc(na + nb + 1);
    if (!out) return NULL;
    memcpy(out, a, na);
    memcpy(out + na, b, nb);
    out[na + nb] = '\0';
    return out;
}

/* Look up a token in the vocabulary and return its ID. */
static int vocab_lookup(const Tokenizer *t, const char *token, uint32_t *out_id) {
    uint32_t v;
    if (!strmap_get(&t->token_to_id, token, &v)) return 0;
    if (out_id) *out_id = v;
    return 1;
}

/* Look up the rank of a merge pair: "left right". */
static int merge_rank_lookup(const Tokenizer *t, const char *left, const char *right, uint32_t *out_rank) {
    char key[512];
    size_t nl = strlen(left);
    size_t nr = strlen(right);
    if (nl + nr + 2 > sizeof(key)) return 0;
    memcpy(key, left, nl);
    key[nl] = ' ';
    memcpy(key + nl + 1, right, nr);
    key[nl + nr + 1] = '\0';
    return strmap_get(&t->merge_rank, key, out_rank);
}

/* Add one special token to the tokenizer's special-token list. */
static int tokenizer_add_special(Tokenizer *t, const char *token, uint32_t id) {
    if (t->special_count >= TOKENIZER_MAX_SPECIALS) return 0;
    t->special_tokens[t->special_count] = token;
    t->special_ids[t->special_count] = id;
    t->special_count++;
    return 1;
}

/* Sort special tokens by decreasing length so longest matches are tested first. */
static int special_cmp_desc_len(const void *a, const void *b) {
    const char *sa = *(const char *const *)a;
    const char *sb = *(const char *const *)b;
    size_t la = strlen(sa);
    size_t lb = strlen(sb);
    if (la < lb) return 1;
    if (la > lb) return -1;
    return strcmp(sa, sb);
}

/*
 * Read a string array from the file.
 * If is_tokens is true, the function fills token-related structures.
 * Otherwise it fills the merge-rank map.
 */
static int read_array(Arena *arena, FILE *f, DeferredStringArray ref_array, HashMap_t *map,
                Tokenizer *t, const TokenTypeRLE *rle, bool is_tokens)
{
    fseek(f, ref_array.file_offset, SEEK_SET);

    uint32_t run_i = 0;
    uint64_t run_pos = 0;

    for (uint64_t i = 0; i < ref_array.count; i++) {
        uint64_t slen;

        /* Each entry is stored as: [u64 length][raw bytes]. */
        if (fread(&slen, sizeof(slen), 1, f) != 1) return 0;

        char *buf = arena_alloc(arena, slen + 1);
        if (!buf) return 0;

        if (fread(buf, 1, slen, f) != slen) return 0;
        buf[slen] = '\0';

        if (!strmap_put(map, buf, (uint32_t)i)) return 0;

        if (is_tokens) {
            /* Keep the reverse mapping for token -> string lookup. */
            t->id_to_token[(uint32_t)i] = buf;

            /* Recover token type from the run-length encoded metadata. */
            int32_t type = 1;
            if (rle && run_i < rle->run_count) {
                type = rle->runs[run_i].value;
                run_pos++;
                if (run_pos >= rle->runs[run_i].count) {
                    run_i++;
                    run_pos = 0;
                }
            }

            /* Token types 3 and 4 are treated as special tokens. */
            if (type == 3 || type == 4) {
                tokenizer_add_special(t, buf, (uint32_t)i);
            }
        }
    }

    return 1;
}

/*
 * Initialize the tokenizer from the GGUF header and the model file.
 * This loads the vocabulary, merge table, and special tokens.
 */
int tokenizer_init(Tokenizer *t, Arena *arena, const gguf_header_t *header, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return 0;

    memset(t, 0, sizeof(*t));

    uint64_t token_count = header->tokens_ref.count;
    uint64_t merge_count = header->merges_ref.count;

    t->token_count = token_count;
    t->merge_count = merge_count;

    /* Model-specific special IDs. */
    t->bos_id = 151643;
    t->eos_id = 151645;
    t->pad_id = 151643;

    build_bytes_to_unicode();

    t->id_to_token = arena_alloc(arena, token_count * sizeof(char *));
    if (!t->id_to_token) return 0;

    if (!strmap_init(&t->token_to_id, arena, token_count)) return 0;
    if (!strmap_init(&t->merge_rank, arena, merge_count)) return 0;

    if (!read_array(arena, f, header->tokens_ref, &t->token_to_id, t, &header->token_types, true)) {
        fclose(f);
        return 0;
    }

    if (!read_array(arena, f, header->merges_ref, &t->merge_rank, t, NULL, false)) {
        fclose(f);
        return 0;
    }

    fclose(f);

    if (t->special_count > 1) {
        //qsort(t->special_tokens, t->special_count, sizeof(t->special_tokens[0]), special_cmp_desc_len);
        for (size_t i = 0; i < t->special_count - 1; i++) {
            size_t max_idx = i;
            for (size_t j = i + 1; j < t->special_count; j++) {
                if (special_cmp_desc_len(&t->special_tokens[j], &t->special_tokens[max_idx]) < 0) {
                    max_idx = j;
                }
            }
            if (max_idx != i) {
                const char *tmp_tok = t->special_tokens[i];
                t->special_tokens[i] = t->special_tokens[max_idx];
                t->special_tokens[max_idx] = tmp_tok;

                uint32_t tmp_id = t->special_ids[i];
                t->special_ids[i] = t->special_ids[max_idx];
                t->special_ids[max_idx] = tmp_id;
            }
        }
    }

    return 1;
}

/* Check whether a special token matches at the current position in the input string. */
static const char *match_special_at(const Tokenizer *t, const char *s, size_t *out_len, uint32_t *out_id) {
    for (size_t i = 0; i < t->special_count; i++) {
        const char *sp = t->special_tokens[i];
        size_t n = strlen(sp);
        if (strncmp(s, sp, n) == 0) {
            if (out_len) *out_len = n;
            if (out_id) *out_id = t->special_ids[i];
            return sp;
        }
    }
    return NULL;
}

/* Ensure the output buffer has enough capacity for at least 'need' elements. */
static int ensure_capacity(uint32_t **buf, size_t *cap, size_t need) {
    if (need <= *cap) return 1;

    size_t new_cap = (*cap == 0) ? 32 : *cap;
    while (new_cap < need) new_cap *= 2;

    uint32_t *tmp = (uint32_t *)realloc(*buf, new_cap * sizeof(uint32_t));
    if (!tmp) return 0;

    *buf = tmp;
    *cap = new_cap;
    return 1;
}

/* Duplicate a string using malloc. */
static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = (char *)malloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n + 1);
    return p;
}

/*
 * Encode a single text piece with BPE.
 * The piece is first converted byte-by-byte through the byte-to-Unicode table,
 * then merged greedily using the lowest-ranked available pair.
 */
static int bpe_encode_piece(const Tokenizer *t, const char *piece, uint32_t **out_ids, size_t *out_n, size_t *out_cap) {
    
    size_t plen = strlen(piece);
    if (plen == 0) return 1;

    char **syms = (char **)malloc(plen * sizeof(char *));
    if (!syms) return 0;
    size_t n = 0;

    for (size_t i = 0; i < plen; i++) {
        unsigned char b = (unsigned char)piece[i];
        syms[n] = xstrdup(bytes_to_unicode_table[b]);
        if (!syms[n]) {
            for (size_t j = 0; j < n; j++) free(syms[j]);
            free(syms);
            return 0;
        }
        n++;
    }

    while (n >= 2) {
        uint32_t best_rank = INT_MAX;
        size_t best_i = (size_t)-1;

        for (size_t i = 0; i + 1 < n; i++) {
            uint32_t rank;
            if (merge_rank_lookup(t, syms[i], syms[i + 1], &rank)) {
                if (rank < best_rank) {
                    best_rank = rank;
                    best_i = i;
                }
            }
        }

        if (best_i == (size_t)-1) break;

        char *merged = concat2(syms[best_i], syms[best_i + 1]);
        if (!merged) {
            for (size_t j = 0; j < n; j++) free(syms[j]);
            free(syms);
            return 0;
        }

        free(syms[best_i]);
        free(syms[best_i + 1]);
        syms[best_i] = merged;

        memmove(&syms[best_i + 1], &syms[best_i + 2], (n - best_i - 2) * sizeof(char *));
        n--;
    }

    if (!ensure_capacity(out_ids, out_cap, *out_n + n)) {
        for (size_t j = 0; j < n; j++) free(syms[j]);
        free(syms);
        return 0;
    }

    for (size_t i = 0; i < n; i++) {
        uint32_t id;
        if (!vocab_lookup(t, syms[i], &id)) {
            if (t->unk_id != UINT32_MAX) {
                id = t->unk_id;
            } else {
                fprintf(stderr, "Tokenizer: missing vocab entry for piece '%s'\n", syms[i]);
                for (size_t j = 0; j < n; j++) free(syms[j]);
                free(syms);
                return 0;
            }
        }
        (*out_ids)[*out_n + i] = id;
    }
    *out_n += n;

    for (size_t j = 0; j < n; j++) free(syms[j]);
    free(syms);

    return 1;
}

int tokenizer_encode(Tokenizer *t,
                     const char *text,
                     uint32_t **out_tokens,
                     size_t *out_count,
                     int add_bos)
{
    if (!t || !text || !out_tokens || !out_count) return 0;

    *out_tokens = NULL;
    *out_count = 0;

    size_t cap = 32;
    uint32_t *ids = (uint32_t *)malloc(cap * sizeof(*ids));
    if (!ids) return 0;

    size_t n = 0;
    if (add_bos && t->bos_id != UINT32_MAX) {
        ids[n++] = t->bos_id;
    }

    size_t text_len = strlen(text);
    char *scratch = (char *)malloc(text_len + 1);
    if (!scratch) {
        free(ids);
        return 0;
    }

    size_t i = 0;
    size_t pending_ws_start = 0;
    size_t pending_ws_len = 0;
    int has_pending_ws = 0;

    while (i < text_len) {
        uint32_t sp_id = 0;
        size_t sp_len = 0;

        if (match_special_at(t, text + i, &sp_len, &sp_id)) {
            if (has_pending_ws) {
                memcpy(scratch, text + pending_ws_start, pending_ws_len);
                scratch[pending_ws_len] = '\0';

                if (!bpe_encode_piece(t, scratch, &ids, &n, &cap)) {
                    free(scratch);
                    free(ids);
                    return 0;
                }

                has_pending_ws = 0;
                pending_ws_len = 0;
            }

            if (!ensure_capacity(&ids, &cap, n + 1)) {
                free(scratch);
                free(ids);
                return 0;
            }

            ids[n++] = sp_id;
            i += sp_len;
            continue;
        }

        if (is_space_byte((unsigned char)text[i])) {
            if (!has_pending_ws) {
                pending_ws_start = i;
                has_pending_ws = 1;
                pending_ws_len = 0;
            }
            ++pending_ws_len;
            ++i;
            continue;
        }

        size_t start = i;
        while (i < text_len) {
            uint32_t dummy_id;
            size_t dummy_len;

            if (match_special_at(t, text + i, &dummy_len, &dummy_id)) break;
            if (is_space_byte((unsigned char)text[i])) break;
            ++i;
        }

        size_t pos = 0;
        if (has_pending_ws) {
            memcpy(scratch + pos, text + pending_ws_start, pending_ws_len);
            pos += pending_ws_len;
            has_pending_ws = 0;
            pending_ws_len = 0;
        }

        memcpy(scratch + pos, text + start, i - start);
        pos += i - start;
        scratch[pos] = '\0';

        if (!bpe_encode_piece(t, scratch, &ids, &n, &cap)) {
            free(scratch);
            free(ids);
            return 0;
        }
    }

    if (has_pending_ws) {
        memcpy(scratch, text + pending_ws_start, pending_ws_len);
        scratch[pending_ws_len] = '\0';

        if (!bpe_encode_piece(t, scratch, &ids, &n, &cap)) {
            free(scratch);
            free(ids);
            return 0;
        }
    }

    free(scratch);

    *out_tokens = ids;
    *out_count = n;
    return 1;
}

static void build_unicode_to_bytes(HashMap_t *map) {
    for (int b = 0; b < 256; b++) {
        strmap_put(map, bytes_to_unicode_table[b], (uint32_t)b);
    }
}

int tokenizer_decode(Tokenizer *t,
                      Arena *arena,
                      const uint32_t *tokens,
                      size_t count,
                      char **out_text)
{
    if (!t || !tokens || !out_text) return 0;

    HashMap_t unicode_to_byte;
    if (!strmap_init(&unicode_to_byte, arena, 512)) return 0;
    build_unicode_to_bytes(&unicode_to_byte);

    size_t cap = 256;
    size_t len = 0;
    char *out = (char *)malloc(cap);
    if (!out) return 0;

    for (size_t k = 0; k < count; k++) {
        uint32_t id = tokens[k];

        if (id >= t->token_count || !t->id_to_token[id]) {
            free(out);
            return 0;
        }

        const char *tok = t->id_to_token[id];
        size_t tok_len = strlen(tok);

        size_t i = 0;
        while (i < tok_len) {
            unsigned char c = (unsigned char)tok[i];
            size_t cp_len;

            if ((c & 0x80) == 0)      cp_len = 1;
            else if ((c & 0xE0) == 0xC0) cp_len = 2;
            else if ((c & 0xF0) == 0xE0) cp_len = 3;
            else if ((c & 0xF8) == 0xF0) cp_len = 4;
            else                          cp_len = 1;

            if (i + cp_len > tok_len) cp_len = tok_len - i;

            char cp_buf[TOKENIZER_MAX_UTF8_BYTES];
            memcpy(cp_buf, tok + i, cp_len);
            cp_buf[cp_len] = '\0';

            uint32_t byte_val;
            if (strmap_get(&unicode_to_byte, cp_buf, &byte_val)) {
                if (len + 1 > cap) {
                    size_t new_cap = cap * 2;
                    char *tmp = (char *)realloc(out, new_cap);
                    if (!tmp) { free(out); return 0; }
                    out = tmp;
                    cap = new_cap;
                }
                out[len++] = (char)byte_val;
            } else {
                if (len + cp_len > cap) {
                    size_t new_cap = cap;
                    while (new_cap < len + cp_len) new_cap *= 2;
                    char *tmp = (char *)realloc(out, new_cap);
                    if (!tmp) { free(out); return 0; }
                    out = tmp;
                    cap = new_cap;
                }
                memcpy(out + len, cp_buf, cp_len);
                len += cp_len;
            }

            i += cp_len;
        }
    }

    if (len + 1 > cap) {
        char *tmp = (char *)realloc(out, len + 1);
        if (!tmp) { free(out); return 0; }
        out = tmp;
    }
    out[len] = '\0';

    *out_text = out;
    return 1;
}

/* Print a compact debug view of the tokenizer state. */
void tokenizer_dump(const Tokenizer *t) {
    if (!t) return;

    printf("=== Tokenizer ===\n");
    printf("tokens     : %zu\n", t->token_count);
    printf("merges     : %zu\n", t->merge_count);
    printf("bos_id     : %u\n",  t->bos_id);
    printf("eos_id     : %u\n",  t->eos_id);
    printf("pad_id     : %u\n",  t->pad_id);
    printf("unk_id     : %u\n",  t->unk_id);

    printf("\n--- special tokens (%zu) ---\n", t->special_count);
    for (size_t i = 0; i < t->special_count; i++) {
        printf("  [%zu] id=%-6u  %s\n", i, t->special_ids[i], t->special_tokens[i]);
    }

    printf("\n--- token_to_id (first 20) ---\n");
    size_t shown = 0;
    for (size_t i = 0; i < t->token_to_id.cap && shown < 20; i++) {
        if (!t->token_to_id.entries[i].key) continue;
        printf("  %-30s → %d\n",
               t->token_to_id.entries[i].key,
               t->token_to_id.entries[i].value);
        shown++;
    }

    printf("\n--- merge_rank (first 20) ---\n");
    shown = 0;
    for (size_t i = 0; i < t->merge_rank.cap && shown < 20; i++) {
        if (!t->merge_rank.entries[i].key) continue;
        printf("  rank %-6d  %s\n",
               t->merge_rank.entries[i].value,
               t->merge_rank.entries[i].key);
        shown++;
    }
}
