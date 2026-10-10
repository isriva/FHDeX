#ifndef POTENTIALS_H_
#define POTENTIALS_H_

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Algorithm.H>
#include <AMReX_REAL.H>

// Parameters for the interaction potential (prefix ip_) and the external
// potential (prefix ep_). All are read from the "amr." ParmParse namespace and
// shared by the SPDE (mykernel.H) and the particle (StochasticPC) discretizations.

// Functional form of the interaction potential. To add a new form: add an enum
// value here, a name in read_potential_params, a case in ip_U and
// ip_dUdr_over_r, and a branch in PrintContinuumStability (AmrCoreAdv.cpp).
enum IntPotType : int {
    GEMA = 0,  // generalized exponential model: U = ip_eps*exp(-(r/ip_R)^ip_alpha)
<<<<<<< HEAD
    HK   = 1,  // U = ip_eps*ip_R*(1-(r/ip_R)^2) for r < ip_R, 0 otherwise
    MORSE = 2, // generalized Morse: U = -ip_eps_att*exp(-(r-ip_re)/ip_R_att)
               //                        + ip_eps_rep*exp(-(r-ip_re)/ip_R_rep)
    PP   = 3   // piecewise parabolic (Gerber et al., arXiv:2510.17629):
               // U = -ip_eps*ip_R*w(r/ip_R) with, for a = ip_pp_a,
               //   w(s) = ip_pp_alpha*(s^2-a^2)/2 + ip_pp_beta*(a^2-1)/2  s <= a
               //   w(s) = ip_pp_beta*(s^2-1)/2                            a < s <= 1
               //   w(s) = 0                                               s > 1
               // ip_pp_alpha = ip_pp_beta = 1 is hk
    WCA  = 4,  // U = 4*ip_eps*((ip_R/r)^12 - (ip_R/r)^6) + ip_eps for r < 2^(1/6)*ip_R,
               // 0 otherwise. The mesh kernel holds U(r) at U(ip_wca_rcap*ip_R) for
               // smaller r; the particle force is uncapped unless ip_wca_cap_part = 1
    GAUSS = 5  // Gaussian: U = ip_eps*exp(-r^2/ip_R^2)
};

// Below this separation the Morse pair force is set to zero: dU/dr does not
// vanish at r = 0, so (dU/dr)/r would be dominated by round-off.
constexpr amrex::Real ip_morse_rmin = 1.e-6;

// WCA cutoff (the minimum of the Lennard-Jones potential) in units of ip_R
constexpr amrex::Real ip_wca_rc = 1.122462048309373; // 2^(1/6)

struct PotentialParams
{
    // interaction potential
    int         use_int_pot = 1;
    int         ip_type     = IntPotType::GEMA;
    amrex::Real ip_eps      = 666.0;
    amrex::Real ip_R        = 0.1;
    amrex::Real ip_alpha    = 3.0;
    amrex::Real ip_eps_att  = 1.0;   // morse only
    amrex::Real ip_eps_rep  = 0.5;   // morse only
    amrex::Real ip_R_att    = 0.1;   // morse only
    amrex::Real ip_R_rep    = 0.05;  // morse only
    amrex::Real ip_re       = 0.0;   // morse only
<<<<<<< HEAD
    amrex::Real ip_wca_rcap = 0.8;   // wca only: core cap radius in units of ip_R
    int         ip_wca_cap_part = 0; // wca only: 1 = also cap the particle force
=======
    amrex::Real ip_pp_alpha = 1.0;   // pp only: core stiffness
    amrex::Real ip_pp_beta  = 1.0;   // pp only: tail stiffness
    amrex::Real ip_pp_a     = 0.5;   // pp only: core radius / ip_R
>>>>>>> origin
    amrex::Real ip_range    = 0.0;   // particle pair cutoff radius; <= 0 means no cutoff
    int         ip_mesh_avg = 1;     // SPDE mesh kernel: 1 = cell-pair average of U, 0 = U at cell centres

    // ensemble directions (top-level is_ensemble_dir): no transport along them,
    // and the SPDE mesh kernel couples only cells in the same realization
    amrex::GpuArray<int,3> ens_dir = {0, 0, 0};

    // external potential
    int         use_ext_pot = 0;
    amrex::Real ep_alpha    = 0.25;
    amrex::Real ep_beta     = 0.75;
    amrex::Real ep_gamma    = 1.1e-3;
};

// Interaction potential U(r), not truncated by ip_range (the SPDE uses the
// full kernel; ip_range is the particle pair cutoff). This is the SPDE mesh
// kernel, so the WCA core is always capped here.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ip_U (amrex::Real r, PotentialParams const& pot) noexcept
{
    switch (pot.ip_type) {
    case IntPotType::MORSE:
        return -pot.ip_eps_att * std::exp(-(r - pot.ip_re)/pot.ip_R_att)
               + pot.ip_eps_rep * std::exp(-(r - pot.ip_re)/pot.ip_R_rep);
    case IntPotType::WCA:
    {
        if (r >= ip_wca_rc*pot.ip_R) { return 0.0; }
        const amrex::Real rr = amrex::max(r, pot.ip_wca_rcap*pot.ip_R);
        const amrex::Real s2 = (pot.ip_R*pot.ip_R)/(rr*rr);
        const amrex::Real s6 = s2*s2*s2;
        return 4.0 * pot.ip_eps * (s6*s6 - s6) + pot.ip_eps;
    }
    case IntPotType::GAUSS:
    {
        const amrex::Real s = r/pot.ip_R;
        return pot.ip_eps * std::exp(-s*s);
    }
    case IntPotType::HK:
    {
        if (r >= pot.ip_R) { return 0.0; }
        const amrex::Real s = r/pot.ip_R;
        return 0.5 * pot.ip_eps * pot.ip_R * (1.0 - s*s);
    }
    case IntPotType::PP:
    {
        if (r >= pot.ip_R) { return 0.0; }
        const amrex::Real s = r/pot.ip_R;
        const amrex::Real a = pot.ip_pp_a;
        const amrex::Real w = (s <= a)
            ? 0.5*pot.ip_pp_alpha*(s*s - a*a) + 0.5*pot.ip_pp_beta*(a*a - 1.0)
            : 0.5*pot.ip_pp_beta*(s*s - 1.0);
        return -pot.ip_eps * pot.ip_R * w;
    }
    case IntPotType::GEMA:
    default:
        return pot.ip_eps * std::exp(-std::pow(r/pot.ip_R, pot.ip_alpha));
    }
}

// Length scale that the mesh-kernel quadrature has to resolve
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ip_Rmin (PotentialParams const& pot) noexcept
{
    if (pot.ip_type == IntPotType::MORSE) { return amrex::min(pot.ip_R_att, pot.ip_R_rep); }
    if (pot.ip_type == IntPotType::WCA) { return (ip_wca_rc - pot.ip_wca_rcap)*pot.ip_R; }
    return pot.ip_R;
}

// exp(-s^2/R^2) averaged over a pair of 1D cells of size h whose centres are
// s apart: (1/h^2) int int exp(-(s+u-v)^2/R^2) du dv over [-h/2,h/2]^2
//   = (R/h)^2 [G(z+a) - 2G(z) + G(z-a)],  z = s/R, a = h/R,
// G(z) = (sqrt(pi)/2) z erf(z) + exp(-z^2)/2, the second antiderivative of
// exp(-z^2). Summed over all cell separations it gives sqrt(pi) R / h exactly.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ip_gauss_cellavg_1d (amrex::Real s, amrex::Real h, amrex::Real R) noexcept
{
    const amrex::Real z = s/R;
    const amrex::Real a = h/R;
    // the second difference cancels badly for h << R; the cell average is then
    // the centre value up to O(a^2)
    if (a < 1.e-3) { return std::exp(-z*z); }
    constexpr amrex::Real sqrtpi_half = 0.886226925452758014;
    auto G = [=] (amrex::Real x) noexcept {
        return sqrtpi_half*x*std::erf(x) + 0.5*std::exp(-x*x);
    };
    return (G(z+a) - 2.0*G(z) + G(z-a)) / (a*a);
}

// SPDE mesh kernel: U averaged over a pair of cells (sizes dx) whose centres
// are separated by sep, along the directions with avg_dir = 1; the others use
// the separation itself (ensemble directions, where only sep = 0 is used).
// This is the finite-volume convolution of U with a piecewise-constant
// density, so sum_cells U*cellvol equals the integral of U at any dx, also
// for ip_R << dx where the centre values miss the kernel almost entirely.
// The Gaussian is done exactly; the other kernels by the triangle-weighted
// form (1/h) int_{-h}^{h} (1 - |u|/h) U(s + u) du per direction, with a
// midpoint rule fine enough to resolve ip_Rmin (at most nmax points per
// direction).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ip_U_cellavg (amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& sep,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                          amrex::GpuArray<int,AMREX_SPACEDIM> const& avg_dir,
                          PotentialParams const& pot) noexcept
{
    if (pot.ip_type == IntPotType::GAUSS) {
        amrex::Real u = pot.ip_eps;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            const amrex::Real s = sep[d]/pot.ip_R;
            u *= avg_dir[d] ? ip_gauss_cellavg_1d(sep[d], dx[d], pot.ip_R) : std::exp(-s*s);
        }
        return u;
    }

    constexpr int nmax = 256;
    const amrex::Real rq = ip_Rmin(pot);
    int n[3] = {1, 1, 1};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (avg_dir[d]) {
            // even, so that the kink of the weight at u = 0 is on a cell edge
            const int nd = 2*static_cast<int>(std::ceil(8.0*dx[d]/rq));
            n[d] = amrex::max(2, amrex::min(nmax, nd));
        }
    }
    amrex::Real sum = 0.0;
    for (int c = 0; c < n[2]; ++c) {
    for (int b = 0; b < n[1]; ++b) {
    for (int a = 0; a < n[0]; ++a) {
        const int idx[3] = {a, b, c};
        amrex::Real r2 = 0.0;
        amrex::Real w = 1.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            amrex::Real x = sep[d];
            if (n[d] > 1) {
                // midpoint of the idx-th of n[d] intervals of [-dx, dx]
                const amrex::Real du = 2.0*dx[d]/n[d];
                const amrex::Real u = -dx[d] + (idx[d] + 0.5)*du;
                x += u;
                w *= (1.0 - std::abs(u)/dx[d]) * du/dx[d];
            }
            r2 += x*x;
        }
        sum += w*ip_U(std::sqrt(r2), pot);
    }}}
    return sum;
}

// (dU/dr)/r, so that the force on particle i from particle j is
// (dU/dr)/r * (x_j - x_i). Used only by the particles.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ip_dUdr_over_r (amrex::Real r, PotentialParams const& pot) noexcept
{
    if (r <= 0.0) { return 0.0; }
    if (pot.ip_range > 0.0 && r >= pot.ip_range) { return 0.0; }
    switch (pot.ip_type) {
    case IntPotType::MORSE:
    {
        if (r < ip_morse_rmin) { return 0.0; }
        const amrex::Real dUdr =
              pot.ip_eps_att/pot.ip_R_att * std::exp(-(r - pot.ip_re)/pot.ip_R_att)
            - pot.ip_eps_rep/pot.ip_R_rep * std::exp(-(r - pot.ip_re)/pot.ip_R_rep);
        return dUdr / r;
    }
    case IntPotType::WCA:
    {
        if (r >= ip_wca_rc*pot.ip_R) { return 0.0; }
        // uncapped by default: resolving the r^-13 core is left to the choice of dt
        if (pot.ip_wca_cap_part && r < pot.ip_wca_rcap*pot.ip_R) { return 0.0; }
        const amrex::Real s2 = (pot.ip_R*pot.ip_R)/(r*r);
        const amrex::Real s6 = s2*s2*s2;
        return -24.0 * pot.ip_eps * (2.0*s6*s6 - s6) / (r*r);
    }
    case IntPotType::GAUSS:
    {
        // dU/dr = -2 r/ip_R^2 U, so (dU/dr)/r is finite at r = 0
        const amrex::Real R2 = pot.ip_R*pot.ip_R;
        return -2.0 * pot.ip_eps * std::exp(-r*r/R2) / R2;
    }
    case IntPotType::HK:
        return (r < pot.ip_R) ? -pot.ip_eps / pot.ip_R : 0.0;
    case IntPotType::PP:
        if (r >= pot.ip_R) { return 0.0; }
        return -pot.ip_eps / pot.ip_R
               * ((r <= pot.ip_pp_a*pot.ip_R) ? pot.ip_pp_alpha : pot.ip_pp_beta);
    case IntPotType::GEMA:
    default:
    {
        const amrex::Real s = std::pow(r/pot.ip_R, pot.ip_alpha);
        return -pot.ip_alpha * pot.ip_eps * std::exp(-s) * s / (r*r);
    }
    }
}

// Magnitude scale of U: max |U| sampled on [0, rmax]. This is |U(0)| for the
<<<<<<< HEAD
// monotone kernels (gema, hk, wca, gauss); for morse it also sees the well.
=======
// monotone kernels (gema, hk, pp); for morse it also sees the well.
>>>>>>> origin
inline amrex::Real ip_Uscale (PotentialParams const& pot, amrex::Real rmax)
{
    constexpr int nsamp = 4096;
    amrex::Real umax = std::abs(ip_U(0.0, pot));
    for (int n = 1; n <= nsamp; ++n) {
        umax = std::max(umax, std::abs(ip_U(rmax*n/nsamp, pot)));
    }
    return umax;
}

inline void read_potential_params (PotentialParams& pot)
{
    amrex::ParmParse pp("amr");

    // Old names are rejected rather than silently ignored
    const char* old_names[3][2] = {{"interaction_strength", "ip_eps"},
                                   {"interaction_scale",    "ip_R"},
                                   {"interaction_range",    "ip_range"}};
    for (auto const& nm : old_names) {
        if (pp.contains(nm[0])) {
            amrex::Abort(std::string("amr.") + nm[0] + " is no longer supported; use amr." + nm[1]);
        }
    }

    pp.query("use_int_pot", pot.use_int_pot);
    std::string ip_type_name = "gema";
    pp.query("ip_type", ip_type_name);
    std::transform(ip_type_name.begin(), ip_type_name.end(), ip_type_name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ip_type_name == "gema") {
        pot.ip_type = IntPotType::GEMA;
    } else if (ip_type_name == "hk") {
        pot.ip_type = IntPotType::HK;
    } else if (ip_type_name == "morse") {
        pot.ip_type = IntPotType::MORSE;
<<<<<<< HEAD
    } else if (ip_type_name == "wca") {
        pot.ip_type = IntPotType::WCA;
    } else if (ip_type_name == "gauss") {
        pot.ip_type = IntPotType::GAUSS;
    } else {
        amrex::Abort("Unknown amr.ip_type = " + ip_type_name + "; valid types are: gema, hk, morse, wca, gauss");
=======
    } else if (ip_type_name == "pp") {
        pot.ip_type = IntPotType::PP;
    } else {
        amrex::Abort("Unknown amr.ip_type = " + ip_type_name + "; valid types are: gema, hk, morse, pp");
>>>>>>> origin
    }
    pp.query("ip_eps", pot.ip_eps);
    pp.query("ip_R", pot.ip_R);
    pp.query("ip_alpha", pot.ip_alpha);
    pp.query("ip_eps_att", pot.ip_eps_att);
    pp.query("ip_eps_rep", pot.ip_eps_rep);
    pp.query("ip_R_att", pot.ip_R_att);
    pp.query("ip_R_rep", pot.ip_R_rep);
    pp.query("ip_re", pot.ip_re);
<<<<<<< HEAD
    pp.query("ip_wca_rcap", pot.ip_wca_rcap);
    pp.query("ip_wca_cap_part", pot.ip_wca_cap_part);
=======
    pp.query("ip_pp_alpha", pot.ip_pp_alpha);
    pp.query("ip_pp_beta", pot.ip_pp_beta);
    pp.query("ip_pp_a", pot.ip_pp_a);
>>>>>>> origin
    pp.query("ip_range", pot.ip_range);

    std::string mesh_kernel = "avg";
    pp.query("ip_mesh_kernel", mesh_kernel);
    if (mesh_kernel == "avg") {
        pot.ip_mesh_avg = 1;
    } else if (mesh_kernel == "point") {
        pot.ip_mesh_avg = 0;
    } else {
        amrex::Abort("Unknown amr.ip_mesh_kernel = " + mesh_kernel + "; valid values are: avg, point");
    }

    {
        amrex::ParmParse pp_top;
        amrex::Vector<int> ens(AMREX_SPACEDIM, 0);
        pp_top.queryarr("is_ensemble_dir", ens, 0, AMREX_SPACEDIM);
        int n_ens = 0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            pot.ens_dir[d] = (ens[d] != 0);
            n_ens += pot.ens_dir[d];
        }
        if (n_ens == AMREX_SPACEDIM) {
            amrex::Abort("is_ensemble_dir cannot be set in every direction");
        }
    }

    pp.query("use_ext_pot", pot.use_ext_pot);
    pp.query("ep_alpha", pot.ep_alpha);
    pp.query("ep_beta", pot.ep_beta);
    pp.query("ep_gamma", pot.ep_gamma);

    if (pot.use_int_pot) {
        if (pot.ip_type == IntPotType::MORSE) {
            if (pot.ip_R_rep <= 0.0) { amrex::Abort("amr.ip_R_rep must be positive"); }
            if (pot.ip_R_att <= pot.ip_R_rep) { amrex::Abort("amr.ip_R_att must be greater than amr.ip_R_rep"); }
            if (pot.ip_eps_rep < 0.0) { amrex::Abort("amr.ip_eps_rep must be non-negative"); }
            if (pot.ip_eps_att <= pot.ip_eps_rep) { amrex::Abort("amr.ip_eps_att must be greater than amr.ip_eps_rep"); }
            if (pot.ip_re < 0.0) { amrex::Abort("amr.ip_re must be non-negative"); }
        } else if (pot.ip_R <= 0.0) {
            amrex::Abort("amr.ip_R must be positive");
        }
        if (pot.ip_type == IntPotType::GEMA && pot.ip_alpha <= 0.0) { amrex::Abort("amr.ip_alpha must be positive"); }
<<<<<<< HEAD
        if (pot.ip_type == IntPotType::WCA) {
            if (pot.ip_eps < 0.0) { amrex::Abort("amr.ip_eps must be non-negative for wca"); }
            // the level-0 mesh kernel samples r = 0, so the core must be capped
            if (pot.ip_wca_rcap <= 0.0 || pot.ip_wca_rcap >= ip_wca_rc) {
                amrex::Abort("amr.ip_wca_rcap must be in (0, 2^(1/6))");
            }
=======
        if (pot.ip_type == IntPotType::PP) {
            if (pot.ip_pp_alpha <= 0.0) { amrex::Abort("amr.ip_pp_alpha must be positive"); }
            if (pot.ip_pp_beta <= 0.0) { amrex::Abort("amr.ip_pp_beta must be positive"); }
            if (pot.ip_pp_a <= 0.0 || pot.ip_pp_a >= 1.0) { amrex::Abort("amr.ip_pp_a must be in (0,1)"); }
>>>>>>> origin
        }
    }
    // HK and PP have compact support, so their exact particle cutoff is ip_R
    if (pot.use_int_pot && (pot.ip_type == IntPotType::HK || pot.ip_type == IntPotType::PP)
        && pot.ip_range <= 0.0) {
        pot.ip_range = pot.ip_R;
    }
    // likewise WCA, with cutoff 2^(1/6)*ip_R
    if (pot.use_int_pot && pot.ip_type == IntPotType::WCA && pot.ip_range <= 0.0) {
        pot.ip_range = ip_wca_rc*pot.ip_R;
    }
    // The Gaussian has no compact support; by default cut it where U has
    // dropped to 1e-5 of |U(0)|, a factor of 10 inside the truncation check below
    if (pot.use_int_pot && pot.ip_type == IntPotType::GAUSS && pot.ip_range <= 0.0) {
        pot.ip_range = pot.ip_R*std::sqrt(std::log(1.e5));
    }
    if (pot.use_ext_pot && pot.ep_gamma <= 0.0) {
        amrex::Abort("amr.ep_gamma must be positive");
    }

    amrex::Print() << "Interaction potential: use_int_pot = " << pot.use_int_pot
                   << " ip_type = " << ip_type_name;
    if (pot.ip_type == IntPotType::MORSE) {
        amrex::Print() << " ip_eps_att = " << pot.ip_eps_att
                       << " ip_eps_rep = " << pot.ip_eps_rep
                       << " ip_R_att = " << pot.ip_R_att
                       << " ip_R_rep = " << pot.ip_R_rep
                       << " ip_re = " << pot.ip_re;
    } else {
        amrex::Print() << " ip_eps = " << pot.ip_eps
                       << " ip_R = " << pot.ip_R;
    }
    if (pot.ip_type == IntPotType::GEMA) {
        amrex::Print() << " ip_alpha = " << pot.ip_alpha;
    }
<<<<<<< HEAD
    if (pot.ip_type == IntPotType::WCA) {
        amrex::Print() << " ip_wca_rcap = " << pot.ip_wca_rcap
                       << " (mesh U capped at " << ip_U(0.0, pot) << ")"
                       << " ip_wca_cap_part = " << pot.ip_wca_cap_part;
=======
    if (pot.ip_type == IntPotType::PP) {
        amrex::Print() << " ip_pp_alpha = " << pot.ip_pp_alpha
                       << " ip_pp_beta = " << pot.ip_pp_beta
                       << " ip_pp_a = " << pot.ip_pp_a;
>>>>>>> origin
    }
    amrex::Print() << " ip_range = " << pot.ip_range
                   << " ip_mesh_kernel = " << mesh_kernel << "\n";
    amrex::Print() << "External potential: use_ext_pot = " << pot.use_ext_pot
                   << " ep_alpha = " << pot.ep_alpha
                   << " ep_beta = " << pot.ep_beta
                   << " ep_gamma = " << pot.ep_gamma << "\n";

    // Size of the potential dropped by the particle cutoff, relative to the
<<<<<<< HEAD
    // scale of U (|U(0)| for gema, hk, wca and gauss)
=======
    // scale of U (|U(0)| for gema, hk and pp)
>>>>>>> origin
    if (pot.use_int_pot && pot.ip_type == IntPotType::MORSE) {
        // range at which the attractive tail drops to 1e-5 of the scale, a
        // factor of 10 inside the 1e-4 truncation limit below
        const amrex::Real uscale = ip_Uscale(pot, pot.ip_re + 20.*pot.ip_R_att);
        const amrex::Real r_sugg = pot.ip_re + pot.ip_R_att*std::log(1.e5*pot.ip_eps_att/uscale);
        amrex::Print() << "Morse potential: |U| scale = " << uscale
                       << " suggested amr.ip_range >= " << r_sugg << "\n";
    }
    if (pot.use_int_pot && pot.ip_range > 0.0) {
        const amrex::Real uscale = ip_Uscale(pot, pot.ip_range);
        const amrex::Real ucut = std::abs(ip_U(pot.ip_range, pot));
        const amrex::Real trunc_rel = (uscale > 0.0) ? ucut/uscale : ucut;
        amrex::Print() << "Interaction potential truncation at ip_range: |U(ip_range)| = " << ucut
                       << " |U(ip_range)|/max|U| = " << trunc_rel << "\n";
        if (trunc_rel > 1.e-4) {
            amrex::Abort("amr.ip_range truncates the interaction potential by more than 1e-4 of max|U|; "
                         "increase amr.ip_range");
        }
    }
}

#endif
