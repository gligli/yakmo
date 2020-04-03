// yakmo -- yet another k-means via orthogonalization
//  $Id: yakmo.h 1866 2015-01-21 10:25:43Z ynaga $
// Copyright (c) 2012-2015 Naoki Yoshinaga <ynaga@tkl.iis.u-tokyo.ac.jp>

#ifdef __GNUC__
#include <getopt.h>
#else
#include "getopt.h"
#endif

#include "pmmintrin.h"
#include "getline.h"
#include "vs_support.h"
#include <stdint.h>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

#ifdef HAVE_CONFIG_H
#include "../config.h"
#endif

#ifdef USE_MT19937
#include <random>
#endif

#ifdef USE_TR1_MT19937
#include <tr1/random>
namespace std { using namespace tr1; }
#define USE_MT19937
#endif

#ifdef USE_HASH
#include <unordered_set>
#endif

#ifdef USE_TR1_HASH
#include <tr1/unordered_set>
namespace std { using namespace tr1; }
#define USE_HASH
#endif

#ifndef USE_HASH
#include <set>
#endif

#define YAKMO_COPYRIGHT  "yakmo - yet another k-means via orthogonalization\n\
Copyright (c) 2012-2013 Naoki Yoshinaga, All rights reserved.\n\
\n\
Usage: %s [options] train model test\n\
\n\
train     train file        set '-' to skip clustering\n\
model     model file        set '-' to train/test w/o saving centroids\n\
test      test  file        set '-' to output clustering results of train\n\
\n"

#define YAKMO_OPT  "Optional parameters in training and testing:\n\
  -t, --dist-type=TYPE      select type of distance function\n\
                            * 0 - Euclidean\n\
  -c, --init-centroid=TYPE  select method of choosing initial centroids\n\
                              0 - random\n\
                            * 1 - k-means++\n\
  -k, --num-cluster=NUM     k in k-means (3)\n\
  -m, --num-result=NUM      number of alternative results (1)\n\
  -i, --iteration=NUM       maximum number of iterations per clustering (0)\n\
  -r, --init-random-seed    initialize random seed in initializing centroids\n\
  -n, --normalize           normalize L2-norm of data points\n\
  -O, --output=TYPE         select output type of testing\n\
                            * 0 - no output\n\
                              1 - report assignment per cluster\n\
                              2 - report assignment per item\n\
  -v, --verbose             verbosity level (1)\n\
  -h, --help                show this help and exit\n"

static const  char*  yakmo_short_options = "t:c:k:m:i:rnO:v:h";

static struct option yakmo_long_options[] = {
  {"dist-type",        required_argument, NULL, 't'},
  {"init-centroid",    required_argument, NULL, 'c'},
  {"num-cluster",      required_argument, NULL, 'k'},
  {"num-result",       required_argument, NULL, 'm'},
  {"iteration",        required_argument, NULL, 'i'},
  {"normalize",        no_argument,       NULL, 'n'},
  {"init-random-seed", no_argument,       NULL, 'r'},
  {"output",           required_argument, NULL, 'O'},
  {"verbose",          no_argument,       NULL, 'v'},
  {"help",             no_argument,       NULL, 'h'},
  {NULL, 0, NULL, 0}
};

namespace yakmo
{
  typedef unsigned int uint;
#ifdef USE_FLOAT
  typedef float  fl_t;
#else
  typedef double fl_t;
#endif
  
	__declspec(noinline) static const float euclidean_baseline_float(const int n, const float* x, const float* y){
    float result = 0.f;
    for(int i = 0; i < n; ++i){
      const float num = x[i] - y[i];
      result += num * num;
    }
    return result;
  }

	__declspec(noinline) static const double euclidean_baseline_double(const int n, const double* x, const double* y) {
		double result = 0.f;
		for (int i = 0; i < n; ++i) {
			const double num = x[i] - y[i];
			result += num * num;
		}
		return result;
	}

	static const float euclidean_intrinsic_float(int n, const float* x, const float* y) {
    __m128 euclidean0 = _mm_setzero_ps();
		__m128 euclidean1 = _mm_setzero_ps();

		for (; n > 7; n -= 8) {
			const __m128 a0 = _mm_loadu_ps(x);
			x += 4;
			const __m128 a1 = _mm_loadu_ps(x);
			x += 4;
			const __m128 b0 = _mm_loadu_ps(y);
			y += 4;
			const __m128 b1 = _mm_loadu_ps(y);
			y += 4;

			const __m128 a0_minus_b0 = _mm_sub_ps(a0, b0);
			const __m128 a1_minus_b1 = _mm_sub_ps(a1, b1);

			const __m128 a0_minus_b0_sq = _mm_mul_ps(a0_minus_b0, a0_minus_b0);
			const __m128 a1_minus_b1_sq = _mm_mul_ps(a1_minus_b1, a1_minus_b1);

			euclidean0 = _mm_add_ps(euclidean0, a0_minus_b0_sq);
			euclidean1 = _mm_add_ps(euclidean1, a1_minus_b1_sq);
		}
    
		const __m128 euclidean = _mm_add_ps(euclidean, euclidean);
		
		const __m128 sum = _mm_hadd_ps(euclidean, euclidean);

		float result = sum.m128_f32[0];
    
    if (n)
      result += euclidean_baseline_float(n, x, y);	// remaining 1-7 entries
    
    return result;
  }  
    
	static const double euclidean_intrinsic_double(int n, const double* x, const double* y) {
		__m128d euclidean0 = _mm_setzero_pd();
		__m128d euclidean1 = _mm_setzero_pd();

		for (; n > 3; n -= 4) {
			const __m128d a0 = _mm_loadu_pd(x);
			x += 2;
			const __m128d a1 = _mm_loadu_pd(x);
			x += 2;

			const __m128d b0 = _mm_loadu_pd(y);
			y += 2;
			const __m128d b1 = _mm_loadu_pd(y);
			y += 2;

			const __m128d a0_minus_b0 = _mm_sub_pd(a0, b0);
			const __m128d a1_minus_b1 = _mm_sub_pd(a1, b1);

			const __m128d a0_minus_b0_sq = _mm_mul_pd(a0_minus_b0, a0_minus_b0);
			const __m128d a1_minus_b1_sq = _mm_mul_pd(a1_minus_b1, a1_minus_b1);

			euclidean0 = _mm_add_pd(euclidean0, a0_minus_b0_sq);
			euclidean1 = _mm_add_pd(euclidean1, a1_minus_b1_sq);
		}

		const __m128d euclidean = _mm_add_pd(euclidean0, euclidean1);

		const __m128d sum = _mm_hadd_pd(euclidean, euclidean);

		double result = sum.m128d_f64[0];

		if (n)
			result += euclidean_baseline_double(n, x, y);	// remaining 1-3 entries

		return result;
  }

  static inline bool getLine (FILE*& fp, char*& line, int64_t& read) {
#ifdef __APPLE__
    if ((line = fgetln (fp, &read)) == NULL) return false;
#else
    static int64_t read_ = 0; static int64_t size = 0; // static helps inlining
    if ((read_ = getline (&line, &size, fp)) == -1) return false;
    read = read_;
#endif
    *(line + read - 1) = '\0';
    return true;
  }
  static bool isspace (const char p) { return p == ' ' || p == '\t'; }
  template <typename T> T strton (const char* s, char** error) {
    const int64_t  ret  = static_cast <int64_t>  (std::strtoll  (s, error, 10));
    const uint64_t retu = static_cast <uint64_t> (std::strtoull (s, error, 10));
    if (std::numeric_limits <T>::is_specialized &&
        (ret  < static_cast <int64_t>  (std::numeric_limits <T>::min ()) ||
         retu > static_cast <uint64_t> (std::numeric_limits <T>::max ())))
      errx (1, "overflow: %s", s);
    return static_cast <T> (ret);
  }
  enum dist_t { EUCLIDEAN };
  enum init_t { RANDOM, KMEANSPP };
  struct option { // option handler
    enum mode_t { BOTH, TRAIN, TEST };
    const char* com, *train, *model, *test;
    //
    dist_t   dist;  // dist-type
    init_t   init;  //
    mutable uint  k;
    mutable uint  m;
    uint     iter;
    bool     random;
    bool     normalize;
    uint16_t output;
    uint     verbosity;
    mode_t   mode;
    option (int argc, char** argv) : com (argc ? argv[0] : "--"), train ("-"), model ("-"), test ("-"), dist (EUCLIDEAN), init (KMEANSPP), k (3), m (1), iter (0), random (false), normalize (false), output (0), verbosity (1), mode (BOTH)
    { set (argc, argv); }
    void set (int argc, char** argv) { // getOpt
      if (argc == 0) return;
      optind = 1;
      while (1) {
        int opt = getopt_long (argc, argv,
                               yakmo_short_options, yakmo_long_options, NULL);
        if (opt == -1) break;
        char* err = NULL;
        switch (opt) {
          case 't': dist      = strton <dist_t> (optarg, &err); break;
          case 'c': init      = strton <init_t> (optarg, &err); break;
          case 'k': k         = strton <uint> (optarg, &err); break;
          case 'm': m         = strton <uint> (optarg, &err); break;
          case 'i': iter      = strton <uint> (optarg, &err); break;
          case 'r': random    = true; break;
          case 'n': normalize = true; break;
          case 'O': output    = strton <uint16_t> (optarg, &err); break;
          case 'v': verbosity = strton <uint> (optarg, &err); break;
            // misc
          case 'h': printCredit (); printHelp (); std::exit (0);
          default:  printCredit (); std::exit (0);
        }
        if (err && *err)
          errx (1, "unrecognized option value: %s", optarg);
      }
      if (dist != EUCLIDEAN)
        errx (1, "only euclidean distance is supported.");
      if (init != RANDOM && init != KMEANSPP)
        errx (1, "unsupported centroid initialization.");
      if (argc < optind + 3) {
        printCredit ();
        errx (1, "Type `%s --help' for option details.", com);
      }
      train = argv[optind];
      model = argv[++optind];
      test  = argv[++optind];
      setMode (); // induce appropriate mode
    }
    void setMode () {
      if (std::strcmp (train, "-") == 0 && std::strcmp (test, "-") == 0)
        errx (1, "specify at least training or test file.");
      else if (std::strcmp (test,  "-") == 0) mode = TRAIN;
      else if (std::strcmp (train, "-") == 0) mode = TEST;
      else                                    mode = BOTH;
      if (std::strcmp (model, "-") == 0 && mode == TEST)
        errx (1, "instant mode needs training files.");
      const char* mode0 [] = {"BOTH", "TRAIN", "TEST"};
      std::fprintf (stderr, "mode: %s\n", mode0[mode]);
    }
    void printCredit () { std::fprintf (stderr, YAKMO_COPYRIGHT, com); }
    void printHelp   () { std::fprintf (stderr, YAKMO_OPT); }
  };
  // implementation of space-efficient k-means using triangle inequality:
  //   G. Hamerly. Making k-means even faster (SDM 2010)
  class kmeans {
  public:
#pragma pack(1)
    struct node_t {
      uint idx;
      fl_t val;
      node_t () : idx (0), val (0) {}
      node_t (uint idx_, fl_t val_) : idx (idx_), val (val_) {}
      bool operator< (const node_t &n) const { return idx < n.idx; }
    };
#pragma pack()
    class centroid_t;
    class point_t {
    public:
      point_t (const node_t* n, const uint size, const fl_t norm)
        : up_d (0), lo_d (0), id (), _size (size), _body (new node_t[_size]), _norm (norm)
      { std::copy (n, n + size, body ()); }
      point_t& operator= (const point_t& p) {
        up_d = p.up_d; lo_d = p.lo_d; id = p.id; _size = p._size; _body = p._body; _norm = p._norm;
        return *this;
      }
      fl_t calc_ip (const centroid_t& c) const {
        // return inner product between this point and the given centroid
        fl_t ret = 0;
        for (const node_t* n = begin (); n != end (); ++n)
          ret += n->val * c[n->idx];
        return ret;
      }
      fl_t calc_dist (const centroid_t& c, const dist_t dist) const {
        // return distance from this point to the given centroid
        fl_t ret = (_norm + c.norm()) * 0.5f;
        fl_t * cp = c._dv;
        for (const node_t* n = begin(); n != end(); ++n)
          ret -= n->val * *++cp;
        return  2.0f * ret;
      }
      void set_closest (const std::vector <centroid_t> &cs, const dist_t dist) {
        uint i   = id == 0 ? 1 : 0; // second closest (cand)
        uint id0 = id;
        fl_t d0 (calc_dist (cs[id0], dist)), d1 (calc_dist (cs[i], dist));
        if (d1 < d0) { id = i; std::swap (d0, d1); }
        for (++i; i < cs.size (); ++i) { // for all other centers
          if (i == id0) continue;
          const fl_t di = calc_dist (cs[i], dist);
          if      (di < d0) { d1 = d0; d0 = di; id = i; }
          else if (di < d1) { d1 = di; }
        }
        up_d = std::sqrt (d0);
        lo_d = std::sqrt (d1);
      }
      void shrink (const uint nf)
      { while (! empty () && back ().idx > nf) --_size; }
      void project (const centroid_t& c) {
        const fl_t norm_ip = calc_ip (c) / c.norm ();
        up_d = lo_d = id = 0; _norm = 0; // reset
        for (uint i = 0; i < _size; ++i) {
          fl_t v = c[_body[i].idx] * norm_ip;
          _norm += v * v;
          _body[i].val = v;
        }
      }
      const node_t* begin () const { return _body; }
      const node_t* end   () const { return _body + _size; }
      fl_t    norm  () const { return _norm; }
      uint    size  () const { return _size; }
      bool    empty () const { return _size == 0; }
      node_t& back  () const { return _body[_size - 1]; }
      node_t* body  () const { return _body; }
      void    clear () const { if (_body) delete [] _body; }
      fl_t    up_d;  // distance to the closest centroid
      fl_t    lo_d;  // distance to the second closest centroid
      uint    id;    // cluster id
    private:
      uint    _size;
      node_t* _body;
      fl_t    _norm;
    };
    class centroid_t {
    public:
      centroid_t (point_t& p, const uint nf, const bool delegate = false) :
        delta (0), next_d (0), _norm (p.norm ()), _dv (0), _sum (0), _body (0), _nelm (0), _nf (nf), _size (0) {
        if (delegate) {
          _size = p.size ();
          _body = p.body (); // delegate
        } else { // workaround for a bug in value initialization in gcc 4.0
          _dv  = new fl_t[_nf + 1]; std::fill_n (_dv,  _nf + 1, 0);
          _sum = new fl_t[_nf + 1]; std::fill_n (_sum, _nf + 1, 0);
          for (const node_t* n = p.begin (); n != p.end (); ++n)
            _dv[n->idx] = n->val;
        }
      }
      fl_t operator[] (const uint i) const { return _dv[i]; }
      void pop (const point_t& p) {
        for (const node_t* n = p.begin (); n != p.end (); ++n)
          _sum[n->idx] -= n->val;
        --_nelm;
      }
      void push (const point_t& p) {
        for (const node_t* n = p.begin (); n != p.end (); ++n)
          _sum[n->idx] += n->val;
        ++_nelm;
      }
      fl_t calc_dist (const centroid_t& c, const dist_t dist, const bool skip = true) const {
        // return distance from this centroid to the given centroid
#ifdef USE_FLOAT
		return euclidean_intrinsic_float(_nf, _dv, c._dv);
#else
		return euclidean_intrinsic_double(_nf, _dv, c._dv);
#endif
      }
      void set_closest (const std::vector <centroid_t>& centroid, const dist_t dist) {
        uint i = (this == &centroid[0]) ? 1 : 0;
        next_d = calc_dist (centroid[i], dist, false);
        for (++i; i < centroid.size (); ++i) {
          if (this == &centroid[i]) continue;
          const fl_t di = calc_dist (centroid[i], dist);
          if (di < next_d) next_d = di;
        }
        next_d = std::sqrt (next_d);
      }
      void reset (const dist_t dist) { // move center
        delta = _norm = 0;
        //switch (dist) {
        //  case EUCLIDEAN:
            for (uint i = 0; i <= _nf; ++i) {
              const fl_t v = _sum[i] / static_cast <fl_t> (_nelm);
              delta += (v - _dv[i]) * (v - _dv[i]);
              _norm += v * v;
              _dv[i] = v;
            }
        //}
        delta = std::sqrt (delta);
      }
      void compress () {
        _size = 0;
        for (uint i = 0; i <= _nf; ++i)
          if (std::fpclassify (_dv[i]) != FP_ZERO)
            ++_size;
        _body = new node_t[_size];
        for (uint i (0), j (0); i <= _nf; ++i)
          if (std::fpclassify (_dv[i]) != FP_ZERO)
            _body[j].idx = i, _body[j].val = _dv[i], ++j;
        delete [] _dv;  _dv  = 0;
        delete [] _sum; _sum = 0;
      }
      void decompress () {
        _dv = new fl_t[_nf + 1]; std::fill_n (_dv, _nf + 1, 0);
        for (uint i = 0; i < _size; ++i)
          _dv[_body[i].idx] = _body[i].val;
        delete [] _body; _body = 0;
      }
      void print (FILE* fp, const uint j) const {
        std::fprintf (fp, "%d", j);
        for (uint i = 0; i < _size; ++i)
          std::fprintf (fp, " %d:%.16g", _body[i].idx, _body[i].val);
        std::fprintf (fp, "\n");
      }
      fl_t norm  () const { return _norm; }
      void clear () {
        if (_dv)   delete [] _dv;
        if (_sum)  delete [] _sum;
        if (_body) delete [] _body;
      }
      //
      fl_t     delta;  // moved distance
      fl_t     next_d; // distance to neighbouring centroind
    private:
      fl_t     _norm;  // norm
      fl_t*    _dv;
      fl_t*    _sum;
      node_t*  _body;
      uint     _nelm;  // # elements belonging to the cluster
      uint     _nf;    // # features
      uint     _size;  // # nozero features
      friend point_t;
    };
    kmeans (const option &opt) : _opt (opt), _point (), _centroid (), _body (), _nf (0) { _centroid.reserve (_opt.k); }
    ~kmeans () {
      clear_point ();
      clear_centroid ();
    }
    void clear_point () {
      for (std::vector <point_t>::iterator it = _point.begin ();
           it != _point.end (); ++it)
        it->clear ();
      std::vector <point_t> ().swap (_point);
    }
    void clear_centroid () {
      for (std::vector <centroid_t>::iterator it = _centroid.begin ();
           it != _centroid.end (); ++it)
        it->clear ();
      std::vector <centroid_t> ().swap (_centroid);
    }
    std::vector <point_t>&    point    () { return _point; }
    std::vector <centroid_t>& centroid () { return _centroid; }
    static point_t read_point (char* const ex, const char* const ex_end, std::vector <node_t>& tmp, const bool normalize = false) {
      tmp.clear ();
      fl_t norm = 0;
      char* p = ex;
      while (p != ex_end) {
        int64_t fi = 0;
        for (; *p >= '0' && *p <= '9'; ++p) {
          fi *= 10, fi += *p, fi -= '0';
          if (fi > std::numeric_limits <uint>::max ())
            errx (1, "overflow: %s", ex);
        }
        if (*p != ':') errx (1, "illegal feature index: %s", ex);
        ++p;
        const fl_t v = static_cast <fl_t> (std::strtod (p, &p));
        tmp.push_back (node_t (static_cast <uint> (fi), v));
        norm += v * v;
        while (isspace (*p)) ++p;
      }
      std::sort (tmp.begin (), tmp.end ());
      if (normalize) { // normalize
        norm = std::sqrt (norm);
        for (std::vector <node_t>::iterator it = tmp.begin ();
             it != tmp.end (); ++it)
          it->val /= norm;
        norm = 1.0;
      }
      return point_t (&tmp[0], static_cast <uint> (tmp.size ()), norm); // expect RVO
    }
    void set_point (char* ex, char* ex_end, const bool normalize) {
      _point.push_back (read_point (ex, ex_end, _body, normalize));
      if (! _point.back ().empty ())
        _nf = std::max (_point.back ().back(). idx, _nf);
    }
    void delegate (kmeans* km) {
      std::swap (_point, km->point ());
      km->nf () = _nf;
    }
    void compress () {
      for (std::vector <centroid_t>::iterator it = _centroid.begin ();
           it != _centroid.end (); ++it)
        it->compress ();
    }
    void decompress () {
      for (std::vector <centroid_t>::iterator it = _centroid.begin ();
           it != _centroid.end (); ++it)
        it->decompress ();
    }
    void push_centroid (point_t& p, const bool delegate = false)
    { _centroid.push_back (centroid_t (p, _nf, delegate)); }
    // implementation of fast k-means:
    //   D. Arthur and S. Vassilvitskii. k-means++: the advantages of careful seeding. SODA (2007)
    void init () {
      struct rng_t {
#ifdef USE_TR1_MT19937
        std::variate_generator <std::mt19937,
                                     std::uniform_real <> > gen;
        rng_t (const bool r) : gen (r ? std::mt19937 (std::random_device () ()) : std::mt19937 (), std::uniform_real <> (0, 1)) {}
        fl_t operator () () { return gen (); }
#elif defined (USE_MT19937)
        std::uniform_real_distribution <> dist;
        std::mt19937 mt;
        rng_t (const bool r) : dist (), mt (r ? std::mt19937 (std::random_device () ()) : std::mt19937 ()) {}
        fl_t operator () () { return dist (mt); }
#else
        size_t x, y, z, w;
        static size_t init () {
          static size_t seed (static_cast <size_t> (std::time (0))), offset (0);
          // multiplier taken from Knuth TAOCP Vol2. 3rd Ed. P.106.
          return seed = 1812433253UL * (seed ^ (seed >> 30)) + ++offset;
        }
        rng_t (const bool r) : x (r ? init () : 123456789), y (r ? init () : 362436069), z (r ? init () : 521288629), w (r ? init () : 88675123) {}
        size_t gen () { // Xorshift RNG; http://www.jstatsoft.org/v08/i14/paper
          size_t t = (x ^ (x << 11)); x = y; y = z; z = w;
          return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
        }
        fl_t operator () () {
          return static_cast <fl_t> (gen () / static_cast <long double> (std::numeric_limits <size_t>::max ()));
        }
#endif
      } rng (_opt.random);
      //
#ifdef USE_HASH
      std::unordered_set <uint> chosen;
#else
      std::set <uint> chosen;
#endif
      std::vector <fl_t> r;
      fl_t obj = 0;
      if (_opt.init == KMEANSPP) r.resize (_point.size (), 0);
      for (uint i = 0; i < _opt.k; ++i) {
        uint c = 0;
        do {
          switch (_opt.init) {
            case RANDOM:
              c = static_cast <uint> (std::floor (rng () * _point.size ()));
              break;
            case KMEANSPP:
              c = static_cast <uint>
                  (i == 0 ?
                   std::floor (rng () * _point.size ()) :
                   std::distance (r.begin (),
                                  std::lower_bound (r.begin (), r.end (), obj * rng ())));
              break;
          }
          // skip chosen centroids; fix a bug reported by Gleb
          while (chosen.find (c) != chosen.end ())
            c = c < _point.size () - 1 ? c + 1 : 0;
        } while (chosen.find (c) != chosen.end ());
        push_centroid (_point[c]);
        obj = 0;
        chosen.insert (c);
        for (uint j = 0; j < _point.size (); ++j) {
          point_t& p = _point[j];
          const fl_t di = p.calc_dist (_centroid[i], _opt.dist);
          if (i == 0 || di < p.up_d)      // closest
            { p.lo_d = p.up_d; p.up_d = di; p.id = i; }
          else if (i == 1 || di < p.lo_d) // second closest
            { p.lo_d = di; }
          if (i < _opt.k - 1) {
            if (_opt.init == KMEANSPP) {
              obj += p.up_d;
              r[j] = obj;
            }
          } else { // i == _k - 1
            p.up_d = std::sqrt (p.up_d);
            p.lo_d = std::sqrt (p.lo_d);
            _centroid[p.id].push (p);
          }
        }
      }
      if (_opt.verbosity > 1) std::fprintf (stderr, "\n");
    }
    void update_bounds () {
      uint id0 (0), id1 (1);
      if (_centroid[id1].delta > _centroid[id0].delta) { std::swap (id0, id1); }
      for (uint j = 2; j < _opt.k; ++j)
        if (_centroid[j].delta > _centroid[id1].delta) {
          id1 = j;
          if (_centroid[j].delta > _centroid[id0].delta) std::swap (id0, id1);
        }
      for (std::vector <point_t>::iterator it = _point.begin ();
           it != _point.end (); ++it) {
        it->up_d += _centroid[it->id].delta;
        it->lo_d -= _centroid[it->id == id0 ? id1 : id0].delta;
      }
    }
    uint& nf () { return _nf; }
    fl_t getObj () const { // const
      fl_t obj = 0;
      for (std::vector <point_t>::const_iterator it = _point.begin ();
           it != _point.end (); ++it)
        obj += it->calc_dist (_centroid[it->id], _opt.dist);
      return obj;
    }
    void run () {
      init ();
      uint moved = static_cast <uint> (_point.size ());
      uint iter_lim = _opt.iter <= 0 ? UINT_MAX : _opt.iter;
      for (uint i = 0; i <= iter_lim; ++i) { // find neighbour center
        if (moved) {
          for (uint j = 0; j < _opt.k; ++j) // move center
            _centroid[j].reset (_opt.dist);
          update_bounds ();
        }
        if (i > 0) {
          if (_opt.verbosity > 1)
            std::fprintf (stderr, "  %3d: obj = %e; #moved = %6d\n", i, getObj (), moved);
          else
            std::fprintf (stderr, ".");
        }
        if (! moved) break;
        for (uint j = 0; j < _opt.k; ++j)
          _centroid[j].set_closest (_centroid, _opt.dist);
        moved = 0;
        for (uint j = 0; j < _point.size (); ++j) { // for all points
          point_t&   p  = _point[j];
          const uint id0 = p.id;
          const fl_t m   = std::max (_centroid[id0].next_d / 2, p.lo_d);
          if (p.up_d > m) {
            p.up_d = std::sqrt (p.calc_dist (_centroid[id0], _opt.dist));
            if (p.up_d > m) {
              p.set_closest (_centroid, _opt.dist);
              if (p.id != id0) {
                ++moved;
                _centroid[id0].pop (p);
                _centroid[p.id].push (p);
              }
            }
          }
        }
      }
      std::fprintf (stderr, "%s", moved ? "break" : "done");
      if (_opt.verbosity == 1)
        std::fprintf (stderr, "; obj = %g.\n", getObj ());
      else
        std::fprintf (stderr, ".\n");
    }
  private:
    const option _opt;
    std::vector <point_t>     _point;
    std::vector <centroid_t>  _centroid;
    std::vector <node_t>      _body;
    uint  _nf;
  };
  // implementation of orthogonal k-means:
  //   Y. Cui et al. Non-redundant multi-view clustering via orthogonalization (ICDM 2007)
  class orthogonal_kmeans {
  public:
    orthogonal_kmeans (const option &opt) : _opt (opt), _kms () {}
    ~orthogonal_kmeans () {
      for (std::vector <kmeans*>::iterator it = _kms.begin ();
           it != _kms.end (); ++it)
        delete *it;
    }
    void print (const uint i,
                const std::vector <const char*>& label,
                const std::vector <std::vector <uint> >& c2p) {
      for (uint j = 0; j < c2p.size (); ++j) {
        if (_opt.m == 1)
          std::fprintf (stdout, "c%d", j);
        else
          std::fprintf (stdout, "c%d_%d", i, j);
        for (std::vector <uint>::const_iterator it = c2p[j].begin ();
             it != c2p[j].end (); ++it)
          std::fprintf (stdout, " %s", label[*it]);
        std::fprintf (stdout, "\n");
      }
    }
    void print (const std::vector <const char*>& label,
                const std::vector <std::vector <uint> >& p2c) {
      for (uint j = 0; j < label.size (); ++j) {
        std::fprintf (stdout, "%s", label[j]);
        for (std::vector <uint>::const_iterator it = p2c[j].begin ();
             it != p2c[j].end (); ++it)
          std::fprintf (stdout, " %d", *it);
        std::fprintf (stdout, "\n");
      }
    }
    void train_from_file (const char* train, const uint iter, const uint output = 0, const bool test_on_other_data = false, const bool instant = false) {
      std::vector <const char*> label;
      std::vector <std::vector <uint> > p2c; // point id to cluster id
      std::vector <std::vector <uint> > c2p (_opt.k); // cluster id to point id
      kmeans* km = new kmeans (_opt);
      FILE* fp = std::fopen (train, "r");
      if (! fp)
        errx (1, "no such file: %s", train);
      char*  line = 0;
      int64_t read = 0;
      while (getLine (fp, line, read)) {
        char* ex (line), *ex_end (line + read - 1);
        while (ex != ex_end && ! isspace (*ex)) ++ex;
        if (! test_on_other_data) {
          char* copy = new char[ex - line + 1];
          std::memcpy (copy, line, static_cast <size_t> (ex - line));
          copy[ex - line] = '\0';
          label.push_back (copy);
          if (output == 2) p2c.push_back (std::vector <uint> ());
        }
        while (isspace (*ex)) ++ex;
        km->set_point (ex, ex_end, _opt.normalize);
      }
      if (km->point ().size () <= _opt.k)
        errx (1, "# points (=%ld) <= k (=%d); done.",
              km->point ().size (), _opt.k);
      _kms.push_back (km);
      std::fclose (fp);
      for (uint i = 1; i <= iter; ++i) {
        std::fprintf (stderr, "iter=%d k-means (k=%d): ", i, _opt.k);
        if (i >= 2) {
          kmeans* km_ = _kms.back (); // last of mohikans
          // project
          std::vector <kmeans::point_t>& point_ = km_->point ();
          for (std::vector <kmeans::point_t>::iterator it = point_.begin ();
               it != point_.end (); ++it)
            it->project (km_->centroid ()[it->id]);
          if (test_on_other_data || ! instant) {
            km = new kmeans (_opt);
            km_->delegate (km);
            km_->compress ();
            _kms.push_back (km);
          } else {
            km_->clear_centroid ();
          }
        }
        km->run ();
        if (! test_on_other_data) {
          if (output == 1)
            for (uint j = 0; j < _opt.k; ++j) c2p[j].clear ();
          std::vector <kmeans::point_t> &point = km->point ();
          for (uint j = 0; j < point.size (); ++j) {
            if (output == 1) c2p[point[j].id].push_back (j);
            if (output == 2) p2c[j].push_back (point[j].id);
          }
          if (output == 1) print (i, label, c2p);
        }
      }
      if (! test_on_other_data)
        if (output == 2) print (label, p2c);
      if (test_on_other_data || ! instant) {
        _kms.back ()->compress ();
        _kms.back ()->clear_point ();
      }
      for (std::vector <const char*>::iterator it = label.begin (); it != label.end (); ++it)
        delete [] *it;
    }
    void save (const char* model) {
      FILE* fp = std::fopen (model, "w");
      std::fprintf (fp, "%d # m\n", _opt.m);
      std::fprintf (fp, "%d # k\n", _opt.k);
      std::fprintf (fp, "%d # number of features\n", _kms.back ()->nf ());
      for (uint i = 0; i < _kms.size (); ++i) {
        const std::vector <kmeans::centroid_t>& centroid = _kms[i]->centroid ();
        for (uint j = 0; j < centroid.size (); ++j) {
          if (_opt.m == 1)
            std::fprintf (fp, "c");
          else
            std::fprintf (fp, "c%d_", i);
          centroid[j].print (fp, j);
        }
      }
      std::fclose (fp);
    }
    void load (const char* model) {
      FILE* fp = std::fopen (model, "r");
      if (! fp)
        errx (1, "no such file: %s", model);
      char*  line = 0;
      int64_t read = 0;
      if (! getLine (fp, line, read)) errx (1, "premature model (0): %s", model);
      _opt.m  = static_cast <uint> (std::strtol (line, NULL, 10));
      if (! getLine (fp, line, read)) errx (1, "premature model (1): %s", model);
      _opt.k  = static_cast <uint> (std::strtol (line, NULL, 10));
      if (! getLine (fp, line, read)) errx (1, "premature model (2): %s", model);
      const uint nf = static_cast <uint> (std::strtol (line, NULL, 10));
      std::vector <kmeans::node_t> body;
      for (uint i = 0; i < _opt.m; ++i) {
        kmeans* km = new kmeans (_opt);
        km->nf () = nf;
        for (uint j = 0; j < _opt.k; ++j) {
          if (! getLine (fp, line, read))
            errx (1, "premature model (+): %s", model);
          char* ex (line), *ex_end (line + read - 1);
          while (ex != ex_end && ! isspace (*ex)) ++ex;
          while (isspace (*ex)) ++ex;
          kmeans::point_t p = kmeans::read_point (ex, ex_end, body, _opt.normalize);
          km->push_centroid (p, true); // delegated
        }
        _kms.push_back (km);
      }
      std::fclose (fp);
    }
    void test_on_file (const char* test, const uint output = 0) {
      std::vector <kmeans::point_t>     point;
      std::vector <const char*>         label;
      std::vector <std::vector <uint> > p2c; // point id to cluster id
      std::vector <std::vector <uint> > c2p (_opt.k); // cluster id to point id
      std::vector <kmeans::node_t> body;
      FILE* fp = std::fopen (test, "r");
      if (! fp)
        errx (1, "no such file: %s", test);
      char*  line = 0;
      int64_t read = 0;
      while (getLine (fp, line, read)) {
        char* ex (line), *ex_end (line + read - 1);
        while (ex != ex_end && ! isspace (*ex)) ++ex;
        char* copy = new char[ex - line + 1];
        std::memcpy (copy, line, static_cast <size_t> (ex - line));
        copy[ex - line] = '\0';
        label.push_back (copy);
        while (isspace (*ex)) ++ex;
        point.push_back (kmeans::read_point (ex, ex_end, body, _opt.normalize));
        if (output == 2) p2c.push_back (std::vector <uint> ());
      }
      std::fclose (fp);
      for (uint i = 0; i < _kms.size (); ++i) {
        if (output == 1)
          for (uint j = 0; j < _opt.k; ++j) c2p[j].clear ();
        kmeans* km =_kms[i];
        km->decompress ();
        for (uint j = 0; j < point.size (); ++j) {
          kmeans::point_t& p = point[j];
          p.shrink (km->nf ());
          p.set_closest (km->centroid (), _opt.dist);
          if (output == 1) c2p[p.id].push_back (j);
          if (output == 2) p2c[j].push_back (p.id);
          p.project (km->centroid ()[p.id]);
        }
        if (output == 1) print (i + 1, label, c2p);
      }
      if (output == 2) print (label, p2c);
      for (std::vector <kmeans::point_t>::iterator it = point.begin (); it != point.end (); ++it)
        it->clear ();
      for (std::vector <const char*>::iterator it = label.begin (); it != label.end (); ++it)
        delete [] *it;
    }
  private:
    const option          _opt;
    std::vector <kmeans*> _kms;
  };
}
