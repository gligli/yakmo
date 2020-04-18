#pragma once

#ifndef __GNUC__
  #define HAVE_CONFIG_H
#endif // __GNUC__

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define __need_clock_t
#include <sys/timeb.h>
#include <sys/types.h>
#include <time.h>

#ifndef __GNUC__
  struct timeval {
    int64_t tv_sec, tv_usec;
  };
#endif

inline static void verrx(int eval, const char *fmt, va_list ap)
{
    //putprog();
    if (fmt != NULL)
        (void)vfprintf(stderr, fmt, ap);
    (void)fputc('\n', stderr);
    exit(eval);
}
 
inline static void errx(int eval, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    verrx(eval, fmt, ap);
    va_end(ap);
}

/* Structure describing CPU time used by a process and its children.  */
struct tms
{
	clock_t tms_utime;          /* User CPU time.  */
	clock_t tms_stime;          /* System CPU time.  */

	clock_t tms_cutime;         /* User CPU time of dead children.  */
	clock_t tms_cstime;         /* System CPU time of dead children.  */
};

typedef long long suseconds_t;

inline static int gettimeofday(struct timeval* t, void* timezone)
{
	struct _timeb timebuffer;
	_ftime64_s(&timebuffer);
	t->tv_sec = timebuffer.time;
	t->tv_usec = 1000 * timebuffer.millitm;
	return 0;
}

inline static clock_t times(struct tms *__buffer) {

	__buffer->tms_utime = clock();
	__buffer->tms_stime = 0;
	__buffer->tms_cstime = 0;
	__buffer->tms_cutime = 0;
	return __buffer->tms_utime;
}
    
#ifdef __cplusplus
}
#endif
