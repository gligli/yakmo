#!/bin/sh
flags="-flto -fuse-linker-plugin -msse3 -O4"
./configure CFLAGS="$flags" CXXFLAGS="$flags" LDFLAGS="$flags" --disable-timer && \
  make clean && make && make install-strip && \
  cp /mingw64/bin/yakmo ../encoder
