#ifndef _getline_h_
#define _getline_h_ 1

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdint.h>

#define __PROTO(args) args

#define GETLINE_NO_LIMIT -1

int64_t
  getline __PROTO ((char **_lineptr, int64_t *_n, FILE *_stream));
int64_t
  getline_safe __PROTO ((char **_lineptr, int64_t *_n, FILE *_stream,
                         int limit));
int64_t
  getstr __PROTO ((char **_lineptr, int64_t *_n, FILE *_stream,
		   int _terminator, int64_t _offset, int limit));

#ifdef __cplusplus
}
#endif

#endif /* _getline_h_ */
