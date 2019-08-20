
CFLAGS := -O3 -Wall -Wextra -Werror -g -mtune=native -march=native

csum: csum.c
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)
