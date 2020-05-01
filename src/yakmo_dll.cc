// yakmo -- yet another k-means via orthogonalization
//  $Id: yakmo.cc 1866 2015-01-21 10:25:43Z ynaga $
// Copyright (c) 2012-2015 Naoki Yoshinaga <ynaga@tkl.iis.u-tokyo.ac.jp>

// DLL version by GliGli

#include <cstdio>
#include "yakmo.h"

#define DLL_API __declspec(dllexport)

#define API_FP_PRE() \
  unsigned int _cFP; \
  _controlfp_s(&_cFP, 0, 0); \
  _set_controlfp(0x1f, 0x1f);

#define API_FP_POST() \
  _set_controlfp(_cFP, 0x1f); \
  _clearfp();


extern "C"
{
  __declspec(dllexport) void* __stdcall yakmo_create(uint32_t  k, uint32_t  restartCount, int32_t maxIter, int32_t initType, int32_t initSeed, int32_t doNormalize, int32_t isVerbose)
	{
    API_FP_PRE();

    yakmo::option opt(0, NULL);
    opt.k = k;
    opt.m = restartCount;
    opt.iter = maxIter;
    opt.init = static_cast<yakmo::init_t>(initType);
    opt.random = initSeed != 0;
    opt.normalize = doNormalize != 0;
    opt.quiet = isVerbose == 0;
    return new yakmo::orthogonal_kmeans(opt);

    API_FP_POST();
  }

  __declspec(dllexport) void __stdcall yakmo_destroy(void* ay)
	{
    API_FP_PRE();

    delete (yakmo::orthogonal_kmeans*)ay;

    API_FP_POST();
  }

  __declspec(dllexport) void __stdcall yakmo_load_train_data(void* ay, uint32_t rowCount, uint32_t colCount, const yakmo::fl_t** dataset)
  {
    API_FP_PRE();

    yakmo::orthogonal_kmeans* yakmo = (yakmo::orthogonal_kmeans*) ay;
    yakmo->load_train_data(rowCount, colCount, dataset);

    API_FP_POST();
  }

  __declspec(dllexport) void __stdcall yakmo_train_on_data(void* ay, int32_t* pointToCluster)
  {
    API_FP_PRE();

    yakmo::orthogonal_kmeans* yakmo = (yakmo::orthogonal_kmeans*) ay;
    yakmo->train_on_data(pointToCluster);

    API_FP_POST();
  }
}
