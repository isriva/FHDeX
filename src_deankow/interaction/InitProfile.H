#ifndef INIT_PROFILE_H_
#define INIT_PROFILE_H_

#include <AMReX_REAL.H>

// Optional piecewise-constant initial profile along x, read from init_profile_x
// (n-1 breakpoints) and init_profile_val (n values). Only the shape matters:
// phi is renormalized to integrate to 1 after init_phi.
struct InitProfile
{
    static constexpr int max_seg = 16;
    int         nseg = 0;   // 0 = use the default initial condition
    amrex::Real x[max_seg-1] = {};
    amrex::Real val[max_seg] = {};
};

#endif
