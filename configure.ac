#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([yakmo], [2015-01-21], [ynaga@tkl.iis.u-tokyo.ac.jp])
AM_INIT_AUTOMAKE([-Wall foreign])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADERS([config.h])
AC_LANG([C++])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([stdint.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_INT64_T
AC_TYPE_MODE_T
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT64_T

# Checks for library functions.
AC_FUNC_STRTOD
AC_CHECK_FUNCS([floor sqrt strtol strtoull])

AC_ARG_ENABLE(64bit,
        [AS_HELP_STRING([--enable-64bit],
            [build 64bit version])],
        [],
        [enable_64bit="yes"])

if [test "x$enable_64bit" = "xyes"]; then
  orig_cflags=$CXXFLAGS
  CXXFLAGS=-m64
  AC_RUN_IFELSE([AC_LANG_PROGRAM([],
     [return sizeof (void*) == 8 ? 0 : 1;])],
     [CXXFLAGS="-m64 $orig_cflags"],
     [AC_MSG_ERROR([Don't know how to build a 64-bit object; try --disable-64bit])
    ])
fi

YAKMO_OBJ=
YAKMO_SRC=
# RTDSC timer
AC_ARG_ENABLE([timer],
        [AS_HELP_STRING([--enable-timer],
            [use RDTSC timer (default=yes)])],
        [],
        [enable_timer="yes"])
if [test "x$enable_timer" != "xno"]; then
  AC_DEFINE([USE_TIMER], [1], [timer support])
  YAKMO_OBJ+=" timer.${OBJEXT}"
fi
AC_SUBST(YAKMO_OBJ)

# C++11 unordered_set
AC_ARG_ENABLE([hash],
        [AS_HELP_STRING([--enable-hash],
            [use C++11 unordered_set (default=yes)])],
        [],
        [enable_hash="yes"])
AS_IF([test "x$enable_hash" != "xno" || test "$TRIE_IMPL" = "hash"],
  [AC_CHECK_HEADERS([unordered_set],
    [AC_DEFINE([USE_HASH], [1], [hash support])],
    [AC_CHECK_HEADERS([tr1/unordered_set],
      [AC_DEFINE([USE_TR1_HASH], [1], [hash support (TR1)])],
      [AC_MSG_WARN(cannot find unordered_set; disabled)])])])

# C++11 mt19937
AC_ARG_ENABLE([mt19937],
        [AS_HELP_STRING([--enable-mt19937],
            [use C++11 mt19937 to shuffle examples (default=no)])],
        [],
        [enable_mt19937="no"])
AS_IF([test "x$enable_mt19937" != "xno"],
  [AC_CHECK_HEADERS([random],
    [AC_DEFINE([USE_MT19937], [1], [mt19937 support])],
    [AC_CHECK_HEADERS([tr1/random],
      [AC_DEFINE([USE_TR1_MT19937], [1], [mt19937 support])],
      [AC_MSG_ERROR(cannot find random; try --disable-mt19937)])])])

# use float as floating-point numbers
AC_ARG_ENABLE([float],
        [AS_HELP_STRING([--enable-float],
            [use float as floating-point numbers (default=no)])],
        [],
        [enable_float="no"])
AS_IF([test "x$enable_float" != "xno"],
  [AC_DEFINE([USE_FLOAT], [1], [float as floating-point numbers])])

# OPENMP
AC_ARG_ENABLE([openmp],
        [AS_HELP_STRING([--enable-openmp],
            [use OPENMP for parallel training (default=no)])],
        [],
        [enable_openmp="no"])
if [test "x${enable_openmp}" != "xno"]; then
  orig_cflags=$CXXFLAGS
  CXXFLAGS="-fopenmp $orig_cflags -O0"
  AC_RUN_IFELSE([AC_LANG_PROGRAM([#include <omp.h>],
                                 [return 0;])],
     [CXXFLAGS="-fopenmp $orig_cflags"],
     [AC_MSG_ERROR([compiler doesn's support OpenMP; try --disable-openmp])])
fi

AC_CONFIG_FILES([Makefile
                 src/Makefile])
AC_OUTPUT
