
CFLAGS := -O3 -Wall -Wextra -Werror -Wno-unused-function -g -mtune=broadwell -march=broadwell

csum: csum.c
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)
