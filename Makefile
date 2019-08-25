
CFLAGS := -O3 -Wall -Wextra -Werror -g -mtune=broadwell -march=broadwell

csum: csum.c
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)
