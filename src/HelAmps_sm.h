// Copyright (C) 2010 The ALOHA Development team and Contributors.
// Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
// Created by: J. Alwall (Sep 2010) for the MG5aMC backend.
//==========================================================================
// Copyright (C) 2020-2024 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified by: A. Valassi (Sep 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2021-2024) for the MG5aMC CUDACPP plugin.
//==========================================================================
// This file has been automatically generated for CUDA/C++ standalone by
// MadGraph5_aMC@NLO v. 3.6.5, 2025-10-17
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef HelAmps_sm_H
#define HelAmps_sm_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuVectors.h"

#include "Parameters_sm.h"

#include <cassert>
//#include <cmath>
//#include <cstdlib>
//#include <iomanip>
//#include <iostream>

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //--------------------------------------------------------------------------

#ifdef MGONGPU_INLINE_HELAMPS
#define INLINE inline
#define ALWAYS_INLINE __attribute__( ( always_inline ) )
#else
#define INLINE
#define ALWAYS_INLINE
#endif

  template<int NP4>
  struct ALOHAOBJ {

      static constexpr int np4 = NP4;
      const fptype_sv * pvec[np4];
      cxtype_sv * w;
      int flv_index;

      // main constructor
      ALOHAOBJ() = default;
      ALOHAOBJ(cxtype_sv * w_sv_i, int flv = 1)
          : w(w_sv_i), flv_index(flv) {}

  };

  //--------------------------------------------------------------------------

  // ALOHA-style object for easy flavor consolidation and non-template API
  struct ALOHAOBJ
  {
    fptype p[4];
    cxtype W[6];
    int flv_index = -1; // -1 means unknown/mixed
    inline ALOHAOBJ() {}
  };


  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS, int NP4>
  __host__ __device__ INLINE void
  ixxxxx( const fptype momenta[], // input: momenta
          const fptype fmass,     // input: fermion mass
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,          // input: flavour
          ALOHAOBJ<NP4> & fi,     // output: aloha objects
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == +PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  ipzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          int flv,                // input: flavor index
          ALOHAOBJ & fi,        // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == -PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  imzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle),
          int flv,                // input: flavor index
          ALOHAOBJ & fi,          // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PT > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  ixzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          int flv,                // input: flavor index
          ALOHAOBJ & fi,         // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction vc[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  vxxxxx( const fptype momenta[], // input: momenta
          const fptype vmass,     // input: vector boson mass
          const int nhel,         // input: -1, 0 (only if vmass!=0) or +1 (helicity of vector boson)
          const int nsv,          // input: +1 (final) or -1 (initial)
          int flv,                // input: flavor index
          ALOHAOBJ & vc,         // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction sc[3] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  sxxxxx( const fptype momenta[], // input: momenta
          //const fptype,                 // WARNING: input "smass" unused (missing in Fortran) - scalar boson mass
          //const int,                    // WARNING: input "nhel" unused (missing in Fortran) - scalar has no helicity!
          const int nss,          // input: +1 (final) or -1 (initial)
          ALOHAOBJ & sc,         // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  oxxxxx( const fptype momenta[], // input: momenta
          const fptype fmass,     // input: fermion mass
          const int nhel,         // input: -1, 0 (only if vmass!=0) or +1 (helicity of vector boson)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          int flv,                // input: flavor index
          ALOHAOBJ & fo,         // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == +PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  opzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          int flv,                // input: flavor index
          ALOHAOBJ & fo,         // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == -PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  omzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          int flv,                // input: flavor index
          ALOHAOBJ & fo,         // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ INLINE void
  oxzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          int flv,                // input: flavor index
          ALOHAOBJ & fo,         // output: wavefunctions
          const int ipar          // input: particle# out of npar
          ) ALWAYS_INLINE;

  //==========================================================================

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS, int NP4>
  __host__ __device__ void
  ixxxxx( const fptype momenta[], // input: momenta
          const fptype fmass,     // input: fermion mass
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          const int flv,
          ALOHAOBJ<NP4> & fi,     // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    // NEW IMPLEMENTATION FIXING FLOATING POINT EXCEPTIONS IN SIMD CODE (#701)
    // Variables xxxDENOM are a hack to avoid division-by-0 FPE while preserving speed (#701 and #727)
    // Variables xxxDENOM are declared as 'volatile' to make sure they are not optimized away on clang! (#724)
    // A few additional variables are declared as 'volatile' to avoid sqrt-of-negative-number FPEs (#736)
    fi.flv_index = flv;
    for (int i = 0; i < fi.np4; ++i)
        fi.pvec[i] = M_ACCESS::kernelAccessIp4IparConst( momenta, i, ipar ) * (fptype)nsf;
    const int nh = nhel * nsf;
    if( fmass != 0. )
    {
#ifndef MGONGPU_CPPSIMD
      const fptype_sv pp = fpmin( fi.pvec[0], fpsqrt( fi.pvec[1] * fi.pvec[1] + fi.pvec[2] * fi.pvec[2] + fi.pvec[3] * fi.pvec[3] ) );
#else
      volatile fptype_sv p2 = fi.pvec[1] * fi.pvec[1] + fi.pvec[2] * fi.pvec[2] + fi.pvec[3] * fi.pvec[3]; // volatile fixes #736
      const fptype_sv pp = fpmin( fi.pvec[0], fpsqrt( p2 ) );
#endif
      // In C++ ixxxxx, use a single ip/im numbering that is valid both for pp==0 and pp>0, which have two numbering schemes in Fortran ixxxxx:
      // for pp==0, Fortran sqm(0:1) has indexes 0,1 as in C++; but for Fortran pp>0, omega(2) has indexes 1,2 and not 0,1
      // NB: this is only possible in ixxxx, but in oxxxxx two different numbering schemes must be used
      const int ip = ( 1 + nh ) / 2; // NB: same as in Fortran pp==0, differs from Fortran pp>0, which is (3+nh)/2 because omega(2) has indexes 1,2
      const int im = ( 1 - nh ) / 2; // NB: same as in Fortran pp==0, differs from Fortran pp>0, which is (3-nh)/2 because omega(2) has indexes 1,2
#ifndef MGONGPU_CPPSIMD
      if( pp == 0. )
      {
        // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
        fptype sqm[2] = { fpsqrt( std::abs( fmass ) ), 0. }; // possibility of negative fermion masses
        //sqm[1] = ( fmass < 0. ? -abs( sqm[0] ) : abs( sqm[0] ) ); // AV: why abs here?
        sqm[1] = ( fmass < 0. ? -sqm[0] : sqm[0] ); // AV: removed an abs here
        fi.w[0] = cxmake( ip * sqm[ip], 0 );
        fi.w[1] = cxmake( im * nsf * sqm[ip], 0 );
        fi.w[2] = cxmake( ip * nsf * sqm[im], 0 );
        fi.w[3] = cxmake( im * sqm[im], 0 );
      }
      else
      {
        const fptype sf[2] = { fptype( 1 + nsf + ( 1 - nsf ) * nh ) * (fptype)0.5,
                               fptype( 1 + nsf - ( 1 - nsf ) * nh ) * (fptype)0.5 };
        fptype omega[2] = { fpsqrt( fi.pvec[0] + pp ), 0. };
        omega[1] = fmass / omega[0];
        const fptype sfomega[2] = { sf[0] * omega[ip], sf[1] * omega[im] };
        const fptype pp3 = fpmax( pp + fi.pvec[3], 0. );
        const cxtype chi[2] = { cxmake( fpsqrt( pp3 * (fptype)0.5 / pp ), 0. ),
                                ( pp3 == 0. ? cxmake( -nh, 0. ) : cxmake( nh * fi.pvec[1], fi.pvec[2] ) / fpsqrt( 2. * pp * pp3 ) ) };
        fi.w[0] = sfomega[0] * chi[im];
        fi.w[1] = sfomega[0] * chi[ip];
        fi.w[2] = sfomega[1] * chi[im];
        fi.w[3] = sfomega[1] * chi[ip];
      }
#else
      // Branch A: pp == 0.
      // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
      fptype sqm[2] = { fpsqrt( std::abs( fmass ) ), 0 }; // possibility of negative fermion masses (NB: SCALAR!)
      sqm[1] = ( fmass < 0 ? -sqm[0] : sqm[0] );          // AV: removed an abs here (as above)
      const cxtype fiA_2 = ip * sqm[ip];                  // scalar cxtype: real part initialised from fptype, imag part = 0
      const cxtype fiA_3 = im * nsf * sqm[ip];            // scalar cxtype: real part initialised from fptype, imag part = 0
      const cxtype fiA_4 = ip * nsf * sqm[im];            // scalar cxtype: real part initialised from fptype, imag part = 0
      const cxtype fiA_5 = im * sqm[im];                  // scalar cxtype: real part initialised from fptype, imag part = 0
      // Branch B: pp != 0.
      const fptype sf[2] = { fptype( 1 + nsf + ( 1 - nsf ) * nh ) * (fptype)0.5,
                             fptype( 1 + nsf - ( 1 - nsf ) * nh ) * (fptype)0.5 };
      fptype_v omega[2] = { fpsqrt( fi.pvec[0] + pp ), 0 };
      omega[1] = fmass / omega[0];
      const fptype_v sfomega[2] = { sf[0] * omega[ip], sf[1] * omega[im] };
      const fptype_v pp3 = fpmax( pp + fi.pvec[3], 0 );
      volatile fptype_v ppDENOM = fpternary( pp != 0, pp, 1. );    // hack: ppDENOM[ieppV]=1 if pp[ieppV]==0
      volatile fptype_v pp3DENOM = fpternary( pp3 != 0, pp3, 1. ); // hack: pp3DENOM[ieppV]=1 if pp3[ieppV]==0
      volatile fptype_v chi0r2 = pp3 * 0.5 / ppDENOM;              // volatile fixes #736
      const cxtype_v chi[2] = { cxmake( fpsqrt( chi0r2 ), 0 ),     // hack: dummy[ieppV] is not used if pp[ieppV]==0
                                cxternary( ( pp3 == 0. ),
                                           cxmake( -nh, 0 ),
                                           cxmake( (fptype)nh * fi.pvec[1], fi.pvec[2] ) / fpsqrt( 2. * ppDENOM * pp3DENOM ) ) }; // hack: dummy[ieppV] is not used if pp[ieppV]==0
      const cxtype_v fiB_2 = sfomega[0] * chi[im];
      const cxtype_v fiB_3 = sfomega[0] * chi[ip];
      const cxtype_v fiB_4 = sfomega[1] * chi[im];
      const cxtype_v fiB_5 = sfomega[1] * chi[ip];
      // Choose between the results from branch A and branch B
      const bool_v mask = ( pp == 0. );
      fi.w[0] = cxternary( mask, fiA_2, fiB_2 );
      fi.w[1] = cxternary( mask, fiA_3, fiB_3 );
      fi.w[2] = cxternary( mask, fiA_4, fiB_4 );
      fi.w[3] = cxternary( mask, fiA_5, fiB_5 );
#endif
    }
    else
    {
#ifdef MGONGPU_CPPSIMD
      volatile fptype_sv p0p3 = fpmax( fi.pvec[0] + fi.pvec[3], 0 ); // volatile fixes #736
      volatile fptype_sv sqp0p3 = fpternary( ( fi.pvec[1] == 0. and fi.pvec[2] == 0. and fi.pvec[3] < 0. ),
                                             fptype_sv{ 0 },
                                             fpsqrt( p0p3 ) * (fptype)nsf );
      volatile fptype_sv sqp0p3DENOM = fpternary( sqp0p3 != 0, (fptype_sv)sqp0p3, 1. ); // hack: dummy sqp0p3DENOM[ieppV]=1 if sqp0p3[ieppV]==0
      cxtype_sv chi[2] = { cxmake( (fptype_v)sqp0p3, 0. ),
                           cxternary( sqp0p3 == 0,
                                      cxmake( -(fptype)nhel * fpsqrt( 2. * fi.pvec[0] ), 0. ),
                                      cxmake( (fptype)nh * fi.pvec[1], fi.pvec[2] ) / (const fptype_v)sqp0p3DENOM ) }; // hack: dummy[ieppV] is not used if sqp0p3[ieppV]==0
#else
      const fptype_sv sqp0p3 = fpternary( ( fi.pvec[1] == 0. and fi.pvec[2] == 0. and fi.pvec[3] < 0. ),
                                          fptype_sv{ 0 },
                                          fpsqrt( fpmax( fi.pvec[0] + fi.pvec[3], 0. ) ) * (fptype)nsf );
      const cxtype_sv chi[2] = { cxmake( sqp0p3, 0. ),
                                 ( sqp0p3 == 0. ? cxmake( -(fptype)nhel * fpsqrt( 2. * fi.pvec[0] ), 0. ) : cxmake( (fptype)nh * fi.pvec[1], fi.pvec[2] ) / sqp0p3 ) };
#endif
      if( nh == 1 )
      {
        fi.w[0] = cxzero_sv();
        fi.w[1] = cxzero_sv();
        fi.w[2] = chi[0];
        fi.w[3] = chi[1];
      }
      else
      {
        fi.w[0] = chi[1];
        fi.w[1] = chi[0];
        fi.w[2] = cxzero_sv();
        fi.w[3] = cxzero_sv();
      }
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == +PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  ipzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          fptype wavefunctions[], // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* fi = W_ACCESS::kernelAccess( wavefunctions );
    fi[0] = cxmake( -pvec3 * (fptype)nsf, -pvec3 * (fptype)nsf );
    fi[1] = cxzero_sv();
    const int nh = nhel * nsf;
    const cxtype_sv sqp0p3 = cxmake( fpsqrt( 2. * pvec3 ) * (fptype)nsf, 0. );
    fi[2] = fi[1];
    if( nh == 1 )
    {
      fi[3] = fi[1];
      fi[4] = sqp0p3;
    }
    else
    {
      fi[3] = sqp0p3;
      fi[4] = fi[1];
    }
    fi[5] = fi[1];
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == -PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  imzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          fptype wavefunctions[], // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* fi = W_ACCESS::kernelAccess( wavefunctions );
    fi[0] = cxmake( pvec3 * (fptype)nsf, -pvec3 * (fptype)nsf );
    fi[1] = cxzero_sv();
    const int nh = nhel * nsf;
    const cxtype_sv chi = cxmake( -(fptype)nhel * fpsqrt( -2. * pvec3 ), 0. );
    fi[3] = cxzero_sv();
    fi[4] = cxzero_sv();
    if( nh == 1 )
    {
      fi[2] = cxzero_sv();
      fi[5] = chi;
    }
    else
    {
      fi[2] = chi;
      fi[5] = cxzero_sv();
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fi[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PT > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  ixzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          fptype wavefunctions[], // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_sv& pvec0 = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_sv& pvec1 = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_sv& pvec2 = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* fi = W_ACCESS::kernelAccess( wavefunctions );
    //fi[0] = cxmake( -pvec0 * nsf, -pvec2 * nsf ); // AV: BUG! not the same as ixxxxx
    //fi[1] = cxmake( -pvec0 * nsf, -pvec1 * nsf ); // AV: BUG! not the same as ixxxxx
    fi[0] = cxmake( -pvec0 * (fptype)nsf, -pvec3 * (fptype)nsf ); // AV: BUG FIX
    fi[1] = cxmake( -pvec1 * (fptype)nsf, -pvec2 * (fptype)nsf ); // AV: BUG FIX
    const int nh = nhel * nsf;
    //const float sqp0p3 = sqrtf( pvec0 + pvec3 ) * nsf; // AV: why force a float here?
    const fptype_sv sqp0p3 = fpsqrt( pvec0 + pvec3 ) * (fptype)nsf;
    const cxtype_sv chi0 = cxmake( sqp0p3, 0. );
    const cxtype_sv chi1 = cxmake( (fptype)nh * pvec1 / sqp0p3, pvec2 / sqp0p3 );
    if( nh == 1 )
    {
      fi[2] = cxzero_sv();
      fi[3] = cxzero_sv();
      fi[4] = chi0;
      fi[5] = chi1;
    }
    else
    {
      fi[2] = chi1;
      fi[3] = chi0;
      fi[4] = cxzero_sv();
      fi[5] = cxzero_sv();
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction vc[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  vxxxxx( const fptype momenta[], // input: momenta
          const fptype vmass,     // input: vector boson mass
          const int nhel,         // input: -1, 0 (only if vmass!=0) or +1 (helicity of vector boson)
          const int nsv,          // input: +1 (final) or -1 (initial)
          ALOHAOBJ & vc, // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    // NEW IMPLEMENTATION FIXING FLOATING POINT EXCEPTIONS IN SIMD CODE (#701)
    // Variables xxxDENOM are a hack to avoid division-by-0 FPE while preserving speed (#701 and #727)
    // Variables xxxDENOM are declared as 'volatile' to make sure they are not optimized away on clang! (#724)
    // A few additional variables are declared as 'volatile' to avoid sqrt-of-negative-number FPEs (#736)
    const fptype sqh = fpsqrt( 0.5 ); // AV this is > 0!
    const fptype hel = nhel;
    for (int i = 0; i < 4; i++)
    {
      vc.p[i] = M_ACCESS::kernelAccessIp4IparConst( momenta, i, ipar ) * nsv;
    }
    if( vmass != 0. )
    {
      const int nsvahl = nsv * std::abs( hel );
      const fptype hel0 = 1. - std::abs( hel );
#ifndef MGONGPU_CPPSIMD
      const fptype_sv pt2 = ( vc.p[0] * vc.p[1] ) + ( vc.p[2] * vc.p[2] );
      const fptype_sv pp = fpmin( vc.p[0], fpsqrt( pt2 + ( vc.p[3] * vc.p[3] ) ) );
      const fptype_sv pt = fpmin( pp, fpsqrt( pt2 ) );
      if( pp == 0. )
      {
        vc.W[0] = cxmake( 0., 0. );
        vc.W[1] = cxmake( -hel * sqh, 0. );
        vc.W[2] = cxmake( 0., nsvahl * sqh );
        vc.W[3] = cxmake( hel0, 0. );
      }
      else
      {
        //printf( "DEBUG1011 (before emp): pvec0=%f vmass=%f pp=%f vmass*pp=%f\n", pvec0, vmass, pp, vmass * pp );
        //const fptype emp = pvec / ( vmass * pp ); // this may give a FPE #1011 (why?! maybe when vmass=+-epsilon?)
        const fptype emp = vc.p[0] / vmass / pp; // workaround for FPE #1011
        //printf( "DEBUG1011 (after emp): emp=%f\n", emp );
        vc.W[0] = cxmake( hel0 * pp / vmass, 0. );
        vc.W[3] = cxmake( hel0 * vc.p[3] * emp + hel * pt / pp * sqh, 0. );
        if( pt != 0. )
        {
          const fptype pzpt = vc.p[3] / ( pp * pt ) * sqh * hel;
          vc.W[1] = cxmake( hel0 * vc.p[1] * emp - vc.p[1] * pzpt, -nsvahl * vc.p[2] / pt * sqh );
          vc.W[2] = cxmake( hel0 * vc.p[2] * emp - vc.p[2] * pzpt, nsvahl * vc.p[1] / pt * sqh );
        }
        else
        {
          vc.W[1] = cxmake( -hel * sqh, 0. );
          // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
          //vc[4] = cxmake( 0., nsvahl * ( pvec3 < 0. ? -std::abs( sqh ) : std::abs( sqh ) ) ); // AV: why abs here?
          vc.W[2] = cxmake( 0., nsvahl * ( pvec3 < 0. ? -sqh : sqh ) ); // AV: removed an abs here
        }
      }
#else
      volatile fptype_sv pt2 = ( vc.p[1] * vc.p[1] ) + ( vc.p[2] * vc.p[2] );
      volatile fptype_sv p2 = pt2 + ( vc.p[3] * vc.p[3] ); // volatile fixes #736
      const fptype_sv pp = fpmin( vc.p[0], fpsqrt( p2 ) );
      const fptype_sv pt = fpmin( pp, fpsqrt( pt2 ) );
      // Branch A: pp == 0.
      const cxtype vcA_2 = cxmake( 0, 0 );
      const cxtype vcA_3 = cxmake( -hel * sqh, 0 );
      const cxtype vcA_4 = cxmake( 0, nsvahl * sqh );
      const cxtype vcA_5 = cxmake( hel0, 0 );
      // Branch B: pp != 0.
      volatile fptype_v ppDENOM = fpternary( pp != 0, pp, 1. ); // hack: ppDENOM[ieppV]=1 if pp[ieppV]==0
      const fptype_v emp = vc.p[0] / ( vmass * ppDENOM );         // hack: dummy[ieppV] is not used if pp[ieppV]==0
      const cxtype_v vcB_2 = cxmake( hel0 * pp / vmass, 0 );
      const cxtype_v vcB_5 = cxmake( hel0 * vc.p[3] * emp + hel * pt / ppDENOM * sqh, 0 ); // hack: dummy[ieppV] is not used if pp[ieppV]==0
      // Branch B1: pp != 0. and pt != 0.
      volatile fptype_v ptDENOM = fpternary( pt != 0, pt, 1. );                                                     // hack: ptDENOM[ieppV]=1 if pt[ieppV]==0
      const fptype_v pzpt = vc.p[3] / ( ppDENOM * ptDENOM ) * sqh * hel;                                              // hack: dummy[ieppV] is not used if pp[ieppV]==0
      const cxtype_v vcB1_3 = cxmake( hel0 * vc.p[1] * emp - vc.p[1] * pzpt, -(fptype)nsvahl * vc.p[2] / ptDENOM * sqh ); // hack: dummy[ieppV] is not used if pt[ieppV]==0
      const cxtype_v vcB1_4 = cxmake( hel0 * vc.p[2] * emp - vc.p[2] * pzpt, (fptype)nsvahl * vc.p[1] / ptDENOM * sqh );  // hack: dummy[ieppV] is not used if pt[ieppV]==0
      // Branch B2: pp != 0. and pt == 0.
      const cxtype vcB2_3 = cxmake( -hel * sqh, 0. );
      const cxtype_v vcB2_4 = cxmake( 0., (fptype)nsvahl * fpternary( ( vc.p[3] < 0 ), -sqh, sqh ) ); // AV: removed an abs here
      // Choose between the results from branch A and branch B (and from branch B1 and branch B2)
      const bool_v mask = ( pp == 0. );
      const bool_v maskB = ( pt != 0. );
      vc.W[0] = cxternary( mask, vcA_2, vcB_2 );
      vc.W[1] = cxternary( mask, vcA_3, cxternary( maskB, vcB1_3, vcB2_3 ) );
      vc.W[2] = cxternary( mask, vcA_4, cxternary( maskB, vcB1_4, vcB2_4 ) );
      vc.W[3] = cxternary( mask, vcA_5, vcB_5 );
#endif
    }
    else
    {
      const fptype_sv& pp = pvec0; // NB: rewrite the following as in Fortran, using pp instead of pvec0
#ifndef MGONGPU_CPPSIMD
      const fptype_sv pt = fpsqrt( ( pvec1 * pvec1 ) + ( pvec2 * pvec2 ) );
#else
      volatile fptype_sv pt2 = pvec1 * pvec1 + pvec2 * pvec2; // volatile fixes #736
      const fptype_sv pt = fpsqrt( pt2 );
#endif
      vc.W[0] = cxzero_sv();
      vc.W[3] = cxmake( hel * pt / pp * sqh, 0. );
#ifndef MGONGPU_CPPSIMD
      if( pt != 0. )
      {
        const fptype pzpt = vc.p[3] / ( pp * pt ) * sqh * hel;
        vc.W[1] = cxmake( -vc.p[1] * pzpt, -nsv * vc.p[2] / pt * sqh );
        vc.W[2] = cxmake( -vc.p[2] * pzpt, nsv * vc.p[1] / pt * sqh );
      }
      else
      {
        vc.W[1] = cxmake( -hel * sqh, 0. );
        // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
        //vc[4] = cxmake( 0, nsv * ( pvec3 < 0. ? -std::abs( sqh ) : std::abs( sqh ) ) ); // AV why abs here?
        vc.W[2] = cxmake( 0., nsv * ( vc.p[3] < 0. ? -sqh : sqh ) ); // AV: removed an abs here
      }
#else
      // Branch A: pt != 0.
      volatile fptype_v ptDENOM = fpternary( pt != 0, pt, 1. );                             // hack: ptDENOM[ieppV]=1 if pt[ieppV]==0
      const fptype_v pzpt = vc.p[3] / ( pp * ptDENOM ) * sqh * hel;                           // hack: dummy[ieppV] is not used if pt[ieppV]==0
      const cxtype_v vcA_3 = cxmake( -vc.p[1] * pzpt, -(fptype)nsv * vc.p[2] / ptDENOM * sqh ); // hack: dummy[ieppV] is not used if pt[ieppV]==0
      const cxtype_v vcA_4 = cxmake( -vc.p[2] * pzpt, (fptype)nsv * vc.p[1] / ptDENOM * sqh );  // hack: dummy[ieppV] is not used if pt[ieppV]==0
      // Branch B: pt == 0.
      const cxtype vcB_3 = cxmake( -(fptype)hel * sqh, 0 );
      const cxtype_v vcB_4 = cxmake( 0, (fptype)nsv * fpternary( ( vc.p[3] < 0 ), -sqh, sqh ) ); // AV: removed an abs here
      // Choose between the results from branch A and branch B
      const bool_v mask = ( pt != 0. );
      vc.W[1] = cxternary( mask, vcA_3, vcB_3 );
      vc.W[2] = cxternary( mask, vcA_4, vcB_4 );
#endif
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction sc[3] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  sxxxxx( const fptype momenta[], // input: momenta
          //const fptype,                 // WARNING: input "smass" unused (missing in Fortran) - scalar boson mass
          //const int,                    // WARNING: input "nhel" unused (missing in Fortran) - scalar has no helicity!
          const int nss,          // input: +1 (final) or -1 (initial)
          int flv,
          ALOHAOBJ &sc, // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    sc.flv_index = flv;
    // const fptype_sv& pvec0 = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    // const fptype_sv& pvec1 = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    // const fptype_sv& pvec2 = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    // const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    // cxtype_sv* sc = W_ACCESS::kernelAccess( wavefunctions );
    for (int i = 0; i < 4; i++)
    {
      sc.p[i] = M_ACCESS::kernelAccessIp4IparConst( momenta, i, ipar ) * nss;
    }
    sc.W[2] = cxmake( 1 + fptype_sv{ 0 }, 0 );
    sc.W[0] = cxmake( sc.p[0] * (fptype)nss, sc.p[3] * (fptype)nss );
    sc.W[1] = cxmake( sc.p[1] * (fptype)nss, sc.p[2] * (fptype)nss );
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  oxxxxx( const fptype momenta[], // input: momenta
          const fptype fmass,     // input: fermion mass
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          int flv,
          ALOHAOBJ & fo, // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    // NEW IMPLEMENTATION FIXING FLOATING POINT EXCEPTIONS IN SIMD CODE (#701)
    // Variables xxxDENOM are a hack to avoid division-by-0 FPE while preserving speed (#701 and #727)
    // Variables xxxDENOM are declared as 'volatile' to make sure they are not optimized away on clang! (#724)
    // A few additional variables are declared as 'volatile' to avoid sqrt-of-negative-number FPEs (#736)
    // const fptype_sv& pvec0 = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    // const fptype_sv& pvec1 = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    // const fptype_sv& pvec2 = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    // const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    // cxtype_sv* fo = W_ACCESS::kernelAccess( wavefunctions );
    // fo[0] = cxmake( pvec0 * (fptype)nsf, pvec3 * (fptype)nsf );
    // fo[1] = cxmake( pvec1 * (fptype)nsf, pvec2 * (fptype)nsf );
    for (int i = 0; i < 4; i++)
    {
      fo.p[i] = M_ACCESS::kernelAccessIp4IparConst( momenta, i, ipar ) * nsf;
    }
    fo.flv_index = flv;
    const int nh = nhel * nsf;
    if( fmass != 0. )
    {
#ifndef MGONGPU_CPPSIMD
      const fptype_sv pp = fpmin( pvec0, fpsqrt( ( pvec1 * pvec1 ) + ( pvec2 * pvec2 ) + ( pvec3 * pvec3 ) ) );
      if( pp == 0. )
      {
        // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
        fptype sqm[2] = { fpsqrt( std::abs( fmass ) ), 0. }; // possibility of negative fermion masses
        //sqm[1] = ( fmass < 0. ? -abs( sqm[0] ) : abs( sqm[0] ) ); // AV: why abs here?
        sqm[1] = ( fmass < 0. ? -sqm[0] : sqm[0] ); // AV: removed an abs here
        const int ip = -( ( 1 - nh ) / 2 ) * nhel;  // NB: Fortran sqm(0:1) also has indexes 0,1 as in C++
        const int im = ( 1 + nh ) / 2 * nhel;       // NB: Fortran sqm(0:1) also has indexes 0,1 as in C++
        fo.W[0] = cxmake( im * sqm[std::abs( ip )], 0 );
        fo.W[1] = cxmake( ip * nsf * sqm[std::abs( ip )], 0 );
        fo.W[2] = cxmake( im * nsf * sqm[std::abs( im )], 0 );
        fo.W[3] = cxmake( ip * sqm[std::abs( im )], 0 );
      }
      else
      {
        const fptype sf[2] = { fptype( 1 + nsf + ( 1 - nsf ) * nh ) * (fptype)0.5,
                               fptype( 1 + nsf - ( 1 - nsf ) * nh ) * (fptype)0.5 };
        fptype omega[2] = { fpsqrt( pvec0 + pp ), 0. };
        omega[1] = fmass / omega[0];
        const int ip = ( 1 + nh ) / 2; // NB: Fortran is (3+nh)/2 because omega(2) has indexes 1,2 and not 0,1
        const int im = ( 1 - nh ) / 2; // NB: Fortran is (3-nh)/2 because omega(2) has indexes 1,2 and not 0,1
        const fptype sfomeg[2] = { sf[0] * omega[ip], sf[1] * omega[im] };
        const fptype pp3 = fpmax( pp + pvec3, 0. );
        const cxtype chi[2] = { cxmake( fpsqrt( pp3 * (fptype)0.5 / pp ), 0. ),
                                ( ( pp3 == 0. ) ? cxmake( -nh, 0. )
                                                : cxmake( nh * fo.p[1], -fo.p[2] ) / fpsqrt( 2. * pp * pp3 ) ) };
        fo.W[0] = sfomeg[1] * chi[im];
        fo.W[1] = sfomeg[1] * chi[ip];
        fo.W[2] = sfomeg[0] * chi[im];
        fo.W[3] = sfomeg[0] * chi[ip];
      }
#else
      volatile fptype_sv p2 = fo.p[1] * fo.p[1] + fo.p[2] * fo.p[2] + fo.p[3] * fo.p[3]; // volatile fixes #736
      const fptype_sv pp = fpmin( fo.p[0], fpsqrt( p2 ) );
      // Branch A: pp == 0.
      // NB: Do not use "abs" for floats! It returns an integer with no build warning! Use std::abs!
      fptype sqm[2] = { fpsqrt( std::abs( fmass ) ), 0 }; // possibility of negative fermion masses
      sqm[1] = ( fmass < 0 ? -sqm[0] : sqm[0] );          // AV: removed an abs here (as above)
      const int ipA = -( ( 1 - nh ) / 2 ) * nhel;
      const int imA = ( 1 + nh ) / 2 * nhel;
      const cxtype foA_2 = imA * sqm[std::abs( ipA )];
      const cxtype foA_3 = ipA * nsf * sqm[std::abs( ipA )];
      const cxtype foA_4 = imA * nsf * sqm[std::abs( imA )];
      const cxtype foA_5 = ipA * sqm[std::abs( imA )];
      // Branch B: pp != 0.
      const fptype sf[2] = { fptype( 1 + nsf + ( 1 - nsf ) * nh ) * (fptype)0.5,
                             fptype( 1 + nsf - ( 1 - nsf ) * nh ) * (fptype)0.5 };
      fptype_v omega[2] = { fpsqrt( fo.p[0] + pp ), 0 };
      omega[1] = fmass / omega[0];
      const int ipB = ( 1 + nh ) / 2;
      const int imB = ( 1 - nh ) / 2;
      const fptype_v sfomeg[2] = { sf[0] * omega[ipB], sf[1] * omega[imB] };
      const fptype_v pp3 = fpmax( pp + fo.p[3], 0. );
      volatile fptype_v ppDENOM = fpternary( pp != 0, pp, 1. );    // hack: ppDENOM[ieppV]=1 if pp[ieppV]==0
      volatile fptype_v pp3DENOM = fpternary( pp3 != 0, pp3, 1. ); // hack: pp3DENOM[ieppV]=1 if pp3[ieppV]==0
      volatile fptype_v chi0r2 = pp3 * 0.5 / ppDENOM;              // volatile fixes #736
      const cxtype_v chi[2] = { cxmake( fpsqrt( chi0r2 ), 0. ),    // hack: dummy[ieppV] is not used if pp[ieppV]==0
                                ( cxternary( ( pp3 == 0. ),
                                             cxmake( -nh, 0. ),
                                             cxmake( (fptype)nh * fo.p[1], -fo.p[2] ) / fpsqrt( 2. * ppDENOM * pp3DENOM ) ) ) }; // hack: dummy[ieppV] is not used if pp[ieppV]==0
      const cxtype_v foB_2 = sfomeg[1] * chi[imB];
      const cxtype_v foB_3 = sfomeg[1] * chi[ipB];
      const cxtype_v foB_4 = sfomeg[0] * chi[imB];
      const cxtype_v foB_5 = sfomeg[0] * chi[ipB];
      // Choose between the results from branch A and branch B
      const bool_v mask = ( pp == 0. );
      fo.W[0] = cxternary( mask, foA_2, foB_2 );
      fo.W[1] = cxternary( mask, foA_3, foB_3 );
      fo.W[2] = cxternary( mask, foA_4, foB_4 );
      fo.W[3] = cxternary( mask, foA_5, foB_5 );
#endif
    }
    else
    {
#ifdef MGONGPU_CPPSIMD
      volatile fptype_sv p0p3 = fpmax( fo.p[0] + fo.p[3], 0 ); // volatile fixes #736
      volatile fptype_sv sqp0p3 = fpternary( ( fo.p[1] == 0. and fo.p[2] == 0. and fo.p[3] < 0. ),
                                             fptype_sv{ 0 },
                                             fpsqrt( p0p3 ) * (fptype)nsf );
      volatile fptype_v sqp0p3DENOM = fpternary( sqp0p3 != 0, (fptype_sv)sqp0p3, 1. ); // hack: sqp0p3DENOM[ieppV]=1 if sqp0p3[ieppV]==0
      const cxtype_v chi[2] = { cxmake( (fptype_v)sqp0p3, 0. ),
                                cxternary( ( sqp0p3 == 0. ),
                                           cxmake( -nhel, 0. ) * fpsqrt( 2. * fo.p[0] ),
                                           cxmake( (fptype)nh * fo.p[1], -fo.p[2] ) / (const fptype_sv)sqp0p3DENOM ) }; // hack: dummy[ieppV] is not used if sqp0p3[ieppV]==0
#else
      const fptype_sv sqp0p3 = fpternary( ( fo.p[1] == 0. ) and ( fo.p[2] == 0. ) and ( fo.p[3] < 0. ),
                                          0,
                                          fpsqrt( fpmax( fo.p[0] + fo.p[3], 0. ) ) * (fptype)nsf );
      const cxtype_sv chi[2] = { cxmake( sqp0p3, 0. ),
                                 ( sqp0p3 == 0. ? cxmake( -nhel, 0. ) * fpsqrt( 2. * fo.p[0] ) : cxmake( (fptype)nh * pvec1, -pvec2 ) / sqp0p3 ) };
#endif
      if( nh == 1 )
      {
        fo.W[0] = chi[0];
        fo.W[1] = chi[1];
        fo.W[2] = cxzero_sv();
        fo.W[3] = cxzero_sv();
      }
      else
      {
        fo.W[0] = cxzero_sv();
        fo.W[1] = cxzero_sv();
        fo.W[2] = chi[1];
        fo.W[3] = chi[0];
      }
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == +PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  opzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          fptype wavefunctions[], // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* fo = W_ACCESS::kernelAccess( wavefunctions );
    fo[0] = cxmake( pvec3 * (fptype)nsf, pvec3 * (fptype)nsf );
    fo[1] = cxzero_sv();
    const int nh = nhel * nsf;
    const cxtype_sv csqp0p3 = cxmake( fpsqrt( 2. * pvec3 ) * (fptype)nsf, 0. );
    fo[3] = cxzero_sv();
    fo[4] = cxzero_sv();
    if( nh == 1 )
    {
      fo[2] = csqp0p3;
      fo[5] = cxzero_sv();
    }
    else
    {
      fo[2] = cxzero_sv();
      fo[5] = csqp0p3;
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PX == PY == 0 and E == -PZ > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  omzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          fptype wavefunctions[], // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* fo = W_ACCESS::kernelAccess( wavefunctions );
    fo[0] = cxmake( -pvec3 * (fptype)nsf, pvec3 * (fptype)nsf ); // remember pvec0 == -pvec3
    fo[1] = cxzero_sv();
    const int nh = nhel * nsf;
    const cxtype_sv chi1 = cxmake( -nhel, 0. ) * fpsqrt( -2. * pvec3 );
    if( nh == 1 )
    {
      fo[2] = cxzero_sv();
      fo[3] = chi1;
      fo[4] = cxzero_sv();
      fo[5] = cxzero_sv();
    }
    else
    {
      fo[2] = cxzero_sv();
      fo[3] = cxzero_sv();
      fo[4] = chi1;
      //fo[5] = chi1; // AV: BUG!
      fo[5] = cxzero_sv(); // AV: BUG FIX
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction fo[6] from the input momenta[npar*4*nevt]
  // ASSUMPTIONS: (FMASS == 0) and (PT > 0)
  template<class M_ACCESS, class W_ACCESS>
  __host__ __device__ void
  oxzxxx( const fptype momenta[], // input: momenta
          //const fptype fmass,   // [skip: ASSUME fermion mass==0]
          const int nhel,         // input: -1 or +1 (helicity of fermion)
          const int nsf,          // input: +1 (particle) or -1 (antiparticle)
          fptype wavefunctions[], // output: wavefunctions
          const int ipar )        // input: particle# out of npar
  {
    mgDebug( 0, __FUNCTION__ );
    const fptype_sv& pvec0 = M_ACCESS::kernelAccessIp4IparConst( momenta, 0, ipar );
    const fptype_sv& pvec1 = M_ACCESS::kernelAccessIp4IparConst( momenta, 1, ipar );
    const fptype_sv& pvec2 = M_ACCESS::kernelAccessIp4IparConst( momenta, 2, ipar );
    const fptype_sv& pvec3 = M_ACCESS::kernelAccessIp4IparConst( momenta, 3, ipar );
    cxtype_sv* fo = W_ACCESS::kernelAccess( wavefunctions );
    fo[0] = cxmake( pvec0 * (fptype)nsf, pvec3 * (fptype)nsf );
    fo[1] = cxmake( pvec1 * (fptype)nsf, pvec2 * (fptype)nsf );
    const int nh = nhel * nsf;
    //const float sqp0p3 = sqrtf( pvec0 + pvec3 ) * nsf; // AV: why force a float here?
    const fptype_sv sqp0p3 = fpsqrt( pvec0 + pvec3 ) * (fptype)nsf;
    const cxtype_sv chi0 = cxmake( sqp0p3, 0. );
    const cxtype_sv chi1 = cxmake( (fptype)nh * pvec1 / sqp0p3, -pvec2 / sqp0p3 );
    if( nh == 1 )
    {
      fo[2] = chi0;
      fo[3] = chi1;
      fo[4] = cxzero_sv();
      fo[5] = cxzero_sv();
    }
    else
    {
      fo[2] = cxzero_sv();
      fo[3] = cxzero_sv();
      fo[4] = chi1;
      fo[5] = chi0;
    }
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //==========================================================================

  // Compute the output amplitude 'vertex' from the input wavefunctions F1[6], F2[6], V3[6]
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ INLINE void
  FFV1_0( ALOHAOBJ F1,
          ALOHAOBJ F2,
          ALOHAOBJ V3,
          const fptype allCOUP[],
          const double Ccoeff,
          fptype allvertexes[] ) ALWAYS_INLINE;

  //--------------------------------------------------------------------------

  // Compute the output wavefunction 'V3[6]' from the input wavefunctions F1[6], F2[6]
  template<class W_ACCESS, class C_ACCESS>
  __device__ INLINE void
  FFV1P0_3( ALOHAOBJ F1,
            ALOHAOBJ F2,
            const fptype allCOUP[],
            const double Ccoeff,
            const fptype M3,
            const fptype W3,
            ALOHAOBJ & V3 ) ALWAYS_INLINE;

  //==========================================================================

  // Compute the output amplitude 'vertex' from the input wavefunctions F1[6], F2[6], V3[6]
  template<class W_ACCESS, class A_ACCESS, class C_ACCESS>
  __device__ void
  FFV1_0( ALOHAOBJ F1,
          ALOHAOBJ F2,
          ALOHAOBJ V3,
          const fptype allCOUP[],
          const double Ccoeff,
          fptype allvertexes[] )
  {
    mgDebug( 0, __FUNCTION__ );
    // const cxtype_sv* F1 = W_ACCESS::kernelAccessConst( allF1 );
    // const cxtype_sv* F2 = W_ACCESS::kernelAccessConst( allF2 );
    // const cxtype_sv* V3 = W_ACCESS::kernelAccessConst( allV3 );
    int flv_index1 = F1.flv_index;
    int flv_index2 = F2.flv_index;
    // TODO: need to do something about this if statement for SIMD
    if(flv_index1 != flv_index2 || flv_index1 == -1)
    {
      ( *vertex ) = cxmake(0, 0); 
      return; 
    }
    const cxtype_sv COUP = C_ACCESS::kernelAccessConst( allCOUP );
    cxtype_sv* vertex = A_ACCESS::kernelAccess( allvertexes );
    const cxtype cI = cxmake( 0., 1. );
    const cxtype_sv TMP0 = (F1.W[0] * (F2.W[2] * (V3.W[0] + V3.W[3]) + F2.W[3] * (V3.W[1] + cI *
      (V3.W[2]))) + (F1.W[1] * (F2.W[2] * (V3.W[1] - cI * (V3.W[2])) + F2.W[3]
      * (V3.W[0] - V3.W[3])) + (F1.W[2] * (F2.W[0] * (V3.W[0] - V3.W[3]) -
      F2.W[1] * (V3.W[1] + cI * (V3.W[2]))) + F1.W[3] * (F2.W[0] * (-V3.W[1] +
      cI * (V3.W[2])) + F2.W[1] * (V3.W[0] + V3.W[3])))));
    ( *vertex ) = Ccoeff * COUP * -cI * TMP0;
    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  // Compute the output wavefunction 'V3[6]' from the input wavefunctions F1[6], F2[6]
  template<class W_ACCESS, class C_ACCESS>
  __device__ void
  FFV1P0_3( ALOHAOBJ F1,
            ALOHAOBJ F2,
            const fptype allCOUP[],
            const double Ccoeff,
            const fptype M3,
            const fptype W3,
            ALOHAOBJ & V3 )
  {
    mgDebug( 0, __FUNCTION__ );
    // const cxtype_sv* F1 = W_ACCESS::kernelAccessConst( allF1 );
    // const cxtype_sv* F2 = W_ACCESS::kernelAccessConst( allF2 );
    // const cxtype_sv COUP = C_ACCESS::kernelAccessConst( allCOUP );
    // cxtype_sv* V3 = W_ACCESS::kernelAccess( allV3 );
 
    const fptype_sv P3[4] = { -cxreal( V3[0] ), -cxreal( V3[1] ), -cximag( V3[1] ), -cximag( V3[0] ) };
    int flv_index1 = F1.flv_index; 
    int flv_index2 = F2.flv_index; 
    // TODO: need to do something about this if statement for SIMD
    if(flv_index1 != flv_index2 || flv_index1 == -1)
    {
      for(int i = 0; i < 4; i++ )
      {
        V3.W[i] = cxmake(0., 0.); 
      }
      return; 
    }
    const cxtype_sv denom = Ccoeff * COUP/((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) - (P3[3] *
        P3[3]) - M3 * (M3 - cI * W3));
    V3.W[0] = denom * (-cI) * (F1.W[0] * F2.W[2] + F1.W[1] * F2.W[3] + F1.W[2] *
        F2.W[0] + F1.W[3] * F2.W[1]);
    V3.W[1] = denom * (-cI) * (-F1.W[0] * F2.W[3] - F1.W[1] * F2.W[2] + F1.W[2] *
        F2.W[1] + F1.W[3] * F2.W[0]);
    V3.W[2] = denom * (-cI) * (-cI * (F1.W[0] * F2.W[3] + F1.W[3] * F2.W[0]) + cI
        * (F1.W[1] * F2.W[2] + F1.W[2] * F2.W[1]));
    V3.W[3] = denom * (-cI) * (-F1.W[0] * F2.W[2] - F1.W[3] * F2.W[1] + F1.W[1] *
        F2.W[3] + F1.W[2] * F2.W[0]);

    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

} // end namespace

#endif // HelAmps_sm_H
