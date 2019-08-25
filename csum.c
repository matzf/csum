#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <immintrin.h>

static void _panic(int error, const char *file, const char *func, int line)
{
	fprintf(stderr, "%s:%s:%i: errno: %d/\"%s\"\n", file, func, line, error, strerror(error));
	exit(EXIT_FAILURE);
}

#define panic(error) _panic(error, __FILE__, __func__, __LINE__)

struct slice
{
  const char *ptr;
  size_t len;
};

struct slice open_r_mmap(const char *filename)
{
  int f = open(filename, O_RDONLY);
  if(f < 0) {
    panic(errno);
  }
  struct stat statbuf;
  int err = fstat(f, &statbuf);
  if(err) {
    panic(errno);
  }
  const size_t len = statbuf.st_size;
  const char *ptr = mmap(NULL, len, PROT_READ, MAP_SHARED|MAP_POPULATE, f, 0);
  if(ptr == MAP_FAILED) {
    panic(errno);
  }
  return (struct slice){ .ptr = ptr, .len = len};
}

__attribute__((noinline)) 
static uint16_t csum_simple(const char *ptr, size_t len)
{
  uint32_t sum = 0;
  if (len == 0) {
    return sum;
  }
  const int len2 = len/2;
  const uint16_t *p16 = (uint16_t *)ptr;
  for (int j = 0; j < len2; j++) {
    sum += p16[j];
  }
  // Add left-over byte, if any
  if (len % 2 != 0) {
    sum += ptr[len-1];
  }
  // Fold 32-bit sum to 16 bits
  while (sum > 0xFFFF) {
    sum = (sum & 0xFFFF) + (sum >> 16);
  }
  return ~sum;
}

static uint16_t fold_csum32(uint32_t sum)
{
  // Fold 32-bit sum to 16 bits
  while (sum > 0xFFFF) {
    sum = (sum & 0xFFFF) + (sum >> 16);
  }
  return ~sum;
}

__attribute__ ((unused))
static void print_pi32(const char *s, __m128i a)
{
  printf("%s: ", s);
  printf("0x%x, ", _mm_extract_epi32(a, 0));
  printf("0x%x, ", _mm_extract_epi32(a, 1));
  printf("0x%x, ", _mm_extract_epi32(a, 2));
  printf("0x%x, ", _mm_extract_epi32(a, 3));
  printf("\n");
}

__attribute__ ((unused))
static void print_i16(const char *s, __m256i a)
{
  printf("%s: ", s);
  printf("0x%x, ", _mm256_extract_epi16(a, 0));
  printf("0x%x, ", _mm256_extract_epi16(a, 1));
  printf("0x%x, ", _mm256_extract_epi16(a, 2));
  printf("0x%x, ", _mm256_extract_epi16(a, 3));
  printf("0x%x, ", _mm256_extract_epi16(a, 4));
  printf("0x%x, ", _mm256_extract_epi16(a, 5));
  printf("0x%x, ", _mm256_extract_epi16(a, 6));
  printf("0x%x, ", _mm256_extract_epi16(a, 7));
  printf("0x%x, ", _mm256_extract_epi16(a, 8));
  printf("0x%x, ", _mm256_extract_epi16(a, 9));
  printf("0x%x, ", _mm256_extract_epi16(a, 10));
  printf("0x%x, ", _mm256_extract_epi16(a, 11));
  printf("0x%x, ", _mm256_extract_epi16(a, 12));
  printf("0x%x, ", _mm256_extract_epi16(a, 13));
  printf("0x%x, ", _mm256_extract_epi16(a, 14));
  printf("0x%x, ", _mm256_extract_epi16(a, 15));
  printf("\n");
}

#define _shuffle_select(a,b,c,d) (a<<6|b<<4|c<<2|d)

static uint32_t hsum_u32(__m256i a)
{
  // a:     a0,a1,a2,a3, a4,a5,a6,a7
  //
  // alo:   a0,a1,a2,a3
  // ahi: + a4,a5,a6,a7
  // 
  // b = alo + ahi
  // b:     b0,b1,b2,b3
  //
  // blo:   b0,b1,<b2,b3>
  // bhi: + b2,b3,<b2,b3>
  //              
  // c:     c0,c1,<.....>
  // return c0 + c1
  
  __m128i alo = _mm256_castsi256_si128(a);
  __m128i ahi = _mm256_extracti128_si256(a, 1); // high 128
  __m128i b = _mm_add_epi32(alo, ahi);     // reduce down to 128

  __m128i blo = b;
  __m128i bhi = _mm_shuffle_epi32(b, _shuffle_select(2,3,2,3));
  __m128i c = _mm_add_epi32(blo, bhi);
  
  uint32_t c0 = _mm_cvtsi128_si32(c);
  uint32_t c1 = _mm_extract_epi32(c, 1);

  return c0 + c1;
}

static uint32_t hsum_u16_u32(__m256i a)
{
  __m128i alo = _mm256_castsi256_si128(a);
  __m128i ahi = _mm256_extracti128_si256(a, 1);
  __m256i alo32 = _mm256_cvtepu16_epi32(alo);
  __m256i ahi32 = _mm256_cvtepu16_epi32(ahi);
  return hsum_u32(alo32) + hsum_u32(ahi32); // XXX can be optimized
}

__attribute__((noinline)) 
static uint16_t csum_avx2_32(const char *ptr, size_t len)
{
  __m128i* d = (__m128i*)ptr;
  __m256i sum_v = _mm256_setzero_si256();
  for (size_t i = 0; i < len/sizeof(*d); ++i) {
    __m256i p = _mm256_cvtepu16_epi32(d[i]);
    sum_v = _mm256_add_epi32(sum_v, p);
  }

  uint32_t sum = hsum_u32(sum_v);
  return fold_csum32(sum);
}

__attribute__((noinline)) 
static uint16_t csum_avx2_16(const char *ptr, size_t len)
{
  __m256i v_sum = _mm256_setzero_si256();
  __m256i carry = _mm256_setzero_si256();
  __m256i one = _mm256_set1_epi16(1);

  __m256i* d = (__m256i*)ptr;
  for (size_t i = 0; i < len/sizeof(*d); ++i) {
    __m256i p = _mm256_load_si256(d + i);
#if 1
    __m256i tmp = _mm256_adds_epu16(p, v_sum);
    v_sum = _mm256_add_epi16(p, v_sum);
    __m256i inc = _mm256_andnot_si256(_mm256_cmpeq_epi16(tmp, v_sum), one);
#else
    __m256i gt = _mm256_cmpgt_epi16(v_sum, p);
    __m256i eq = _mm256_cmpeq_epi16(v_sum, p);
    __m256i not_less = _mm256_or_si256(gt, eq);
    __m256i inc = _mm256_andnot_si256(not_less, one);
#endif
    carry = _mm256_add_epi16(carry, inc);
#if 0
    if(i<4) {
      printf("--\n");
      print_i16("p  ", p);
      print_i16("tmp", tmp);
      print_i16("inc", inc);
      print_i16("sum", v_sum);
      print_i16("car", carry);
    }
#endif
  }

  v_sum = _mm256_add_epi16(v_sum, carry);
  uint32_t sum = hsum_u16_u32(v_sum);
  return fold_csum32(sum);
}

#define benchmark(fun) \
  __attribute__((noinline, unused)) \
  static uint16_t benchmark_##fun(struct slice s) { \
    volatile uint16_t sum; \
    for(int i = 0; i < 16*1024*1024; ++i) { \
      sum = fun(s.ptr, s.len); \
    } \
    printf("%s: 0x%x\n", #fun, sum); \
    return sum; \
  }

benchmark(csum_simple)
benchmark(csum_avx2_16)
benchmark(csum_avx2_32)


int main(int argc, const char **argv)
{
  if(argc < 2) exit(2);
  struct slice s = open_r_mmap(argv[1]);
  s.len = 1024;
  /*
  size_t len = 4096;
  char *ptr = aligned_alloc(32, len);
  memset(ptr, 0, len);
  uint16_t *p16 = (uint16_t*)ptr;
  p16[0] = 0xffff;
  p16[16] = 0x2;
  struct slice s = { .ptr = ptr, .len = len };
  */

  printf("%p, %zu\n", s.ptr, s.len);
  printf("simple: 0x%x\n", csum_simple(s.ptr, s.len));
  printf("avx2_16: 0x%x\n", csum_avx2_16(s.ptr, s.len));
  printf("avx2_32: 0x%x\n", csum_avx2_32(s.ptr, s.len));

  for(int i = 0; i < 4; ++i) {
    benchmark_csum_simple(s);
    benchmark_csum_avx2_16(s);
    benchmark_csum_avx2_32(s);
  }
}
