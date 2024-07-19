Explore various SIMD implementations of 16 bit one's complement checksum as
used in IP, UDP and TCP.
Some algorithms are currently broken on unpadded inputs.

```
make
./csum -b      # benchmark; run 16M times on random input 1KiB input
./csum <file>  # compute different checksums on file
```
