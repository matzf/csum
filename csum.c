#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>

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

static struct slice open_r_mmap(const char *filename)
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

int main(int argc, const char **argv)
{
  if(argc < 2) exit(2);
  struct slice s = open_r_mmap(argv[1]);
  printf("%p, %zu\n", s.ptr, s.len);
  volatile uint16_t sum;
  for(int i = 0; i < 128; ++i) {
    sum = csum_simple(s.ptr, s.len);
  }
  printf("%x\n", sum);
}
