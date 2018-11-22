// pecco -- please enjoy classification with conjunctive features
//  $Id: timer.cc 1852 2014-06-20 15:02:49Z ynaga $
// Copyright (c) 2008-2014 Naoki Yoshinaga <ynaga@tkl.iis.u-tokyo.ac.jp>
#include "timer.h"

namespace ny {
  long double Timer::clock () {
    static long double clock = 0;
    if (std::fpclassify (clock) == FP_ZERO) {
      register uint64_t start, end;
      timeval s_t, e_t;
      register int min_interval = 5000;
      uint64_t interval = 0;
      gettimeofday (&s_t, NULL);
      start = rdtsc ();
      gettimeofday (&e_t, NULL);
      while ((e_t.tv_sec - s_t.tv_sec) * 1000000 + e_t.tv_usec - s_t.tv_usec < min_interval)
        gettimeofday (&e_t, NULL);
      end = rdtsc ();
      interval = static_cast <uint64_t> ((e_t.tv_sec - s_t.tv_sec) * 1000000 + e_t.tv_usec - s_t.tv_usec);
      clock = (long double) (end - start) / interval;
    }
    return clock;
  }
  void Timer::printElapsed () const {
    if (_trial == 0) return;
    if (_trial == 1)
      std::fprintf (stderr, "%-10s: %.4Lf ms.\n",
                    _label.c_str (), ( _elapsed / clock ()) / 1000);
    else
      std::fprintf (stderr, "%-10s: %.4Lf ms./%s (%.8Lf/%lu)\n",
                    _label.c_str (), _elapsed / (clock () * 1000) / _trial,
                    _unit.c_str (),  _elapsed / (clock () * 1000), _trial);
  }
}
