// yakmo -- yet another k-means via orthogonalization
//  $Id: yakmo.cc 1866 2015-01-21 10:25:43Z ynaga $
// Copyright (c) 2012-2015 Naoki Yoshinaga <ynaga@tkl.iis.u-tokyo.ac.jp>
#include <cstdio>
#include "yakmo.h"
#include "timer.h"

int main (int argc, char** argv) {
  yakmo::option opt (argc, argv);
  yakmo::orthogonal_kmeans* m = new yakmo::orthogonal_kmeans (opt);

#ifdef USE_TIMER
  ny::TimerPool timer_pool;
  ny::Timer* train_t = timer_pool.push ("train");
  ny::Timer* test_t  = timer_pool.push ("test");
#endif
  
  bool instant = std::strcmp (opt.model, "-") == 0;
  if (opt.mode == yakmo::option::TRAIN || opt.mode == yakmo::option::BOTH) {
    TIMER (train_t->startTimer ());
    m->train_from_file (opt.train, opt.m, opt.output, opt.mode != yakmo::option::TRAIN, instant);
    if (! instant) m->save (opt.model);
    TIMER (train_t->stopTimer ());
  }
  if (opt.mode == yakmo::option::TEST || opt.mode == yakmo::option::BOTH) {
    TIMER (test_t->startTimer ());
    if (opt.mode == yakmo::option::BOTH && ! instant)
      { delete m; m = new yakmo::orthogonal_kmeans (opt); }
    if (! instant) m->load (opt.model);
    m->test_on_file (opt.test, opt.output);
    TIMER (test_t->stopTimer ());
  }
  delete m;
  TIMER (timer_pool.print ());
  return 0;
}
