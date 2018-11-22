# AM_CXXFLAGS = -Wall -Wextra -Wformat=2 -Wcast-qual -Wcast-align -Wwrite-strings -Wconversion -Wpointer-arith -Wshadow -pedantic
AM_CXXFLAGS = -Wall
bin_PROGRAMS = yakmo

yakmo_SOURCES = yakmo.h yakmo.cc
EXTRA_yakmo_SOURCES = timer.cc timer.h

yakmo_DEPENDENCIES = @YAKMO_OBJ@
yakmo_LDADD = @YAKMO_OBJ@
