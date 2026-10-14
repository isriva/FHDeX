#ifndef MY_KERNEL_H_
#define MY_KERNEL_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_Parser.H>
#include "InitProfile.H"

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init_rho (int i, int j, int k,
               amrex::Array4<amrex::Real> const& rho,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
               amrex::Real rho0,
               InitProfile const& prof)
{
    using amrex::Real;

    rho(i,j,k,0) = rho0;
    if (prof.nseg > 0) {
        const Real xc = prob_lo[0] + (i + 0.5)*dx[0];
        int s = 0;
        while (s < prof.nseg-1 && xc >= prof.x[s]) { ++s; }
        rho(i,j,k,0) = prof.val[s];
    }
}

// Face flux F = D(rho_f) drho/dx + sqrt(sigma(rho_fn)) * stochflux, with rho_f
// the arithmetic average of the two adjacent cell-centred densities and
// rho_fn the average of their non-negative parts, as for the noise amplitude
// in ../interaction; sigma is also clipped at 0 (e.g. SSEP for rho > 1).
// stochflux holds N(0, 1/(Lambda dt cellvol)) samples (see advance_rho).
// A wall (bc = 3) is a zero Dirichlet boundary: the ghost value is
// 2*rhobdry - rho, the face density is rhobdry and the noise is scaled by
// sqrt(2), as in ../interaction/mykernel.H.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_x (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxx,
                     amrex::Array4<amrex::Real const> const& stochfluxx,
                     amrex::Array4<amrex::Real const> const& rho, amrex::Real dxinv,
                     amrex::ParserExecutor<1> const& Dfn,
                     amrex::ParserExecutor<1> const& sigfn,
                     int dom_lo, int dom_hi, int bc_lo, int bc_hi)
{
    amrex::Real rhobdry = 0.;
    amrex::Real stofact = 1.;

    amrex::Real rhop, rhom, rhof, rhofn;

    if (i == dom_lo && bc_lo == amrex::BCType::ext_dir) {

        stofact = std::sqrt(2.);
        rhom = 2.*rhobdry - rho(i,j,k,0);
        rhop = rho(i,j,k,0);
        rhof = rhobdry;
        rhofn = rhobdry;

    } else if (i == dom_hi+1 && bc_hi == amrex::BCType::ext_dir) {

        stofact = std::sqrt(2.);
        rhop = 2.*rhobdry - rho(i-1,j,k,0);
        rhom = rho(i-1,j,k,0);
        rhof = rhobdry;
        rhofn = rhobdry;

    } else {

        rhop = rho(i,j,k,0);
        rhom = rho(i-1,j,k,0);
        rhof = 0.5*(rhop+rhom);
        rhofn = 0.5*(amrex::max(rhop,amrex::Real(0.))+amrex::max(rhom,amrex::Real(0.)));

    }

    fluxx(i,j,k,0) = Dfn(rhof)*(rhop-rhom) * dxinv
                   + stofact * stochfluxx(i,j,k) * std::sqrt(amrex::max(sigfn(rhofn), amrex::Real(0.)));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_y (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxy,
                     amrex::Array4<amrex::Real const> const& stochfluxy,
                     amrex::Array4<amrex::Real const> const& rho, amrex::Real dyinv,
                     amrex::ParserExecutor<1> const& Dfn,
                     amrex::ParserExecutor<1> const& sigfn,
                     int dom_lo, int dom_hi, int bc_lo, int bc_hi)
{
    amrex::Real rhobdry = 0.;
    amrex::Real stofact = 1.;

    amrex::Real rhop, rhom, rhof, rhofn;

    if (j == dom_lo && bc_lo == amrex::BCType::ext_dir) {

        stofact = std::sqrt(2.);
        rhom = 2.*rhobdry - rho(i,j,k,0);
        rhop = rho(i,j,k,0);
        rhof = rhobdry;
        rhofn = rhobdry;

    } else if (j == dom_hi+1 && bc_hi == amrex::BCType::ext_dir) {

        stofact = std::sqrt(2.);
        rhop = 2.*rhobdry - rho(i,j-1,k,0);
        rhom = rho(i,j-1,k,0);
        rhof = rhobdry;
        rhofn = rhobdry;

    } else {

        rhop = rho(i,j,k,0);
        rhom = rho(i,j-1,k,0);
        rhof = 0.5*(rhop+rhom);
        rhofn = 0.5*(amrex::max(rhop,amrex::Real(0.))+amrex::max(rhom,amrex::Real(0.)));

    }

    fluxy(i,j,k,0) = Dfn(rhof)*(rhop-rhom) * dyinv
                   + stofact * stochfluxy(i,j,k) * std::sqrt(amrex::max(sigfn(rhofn), amrex::Real(0.)));
}

#if (AMREX_SPACEDIM > 2)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_flux_z (int i, int j, int k,
                     amrex::Array4<amrex::Real> const& fluxz,
                     amrex::Array4<amrex::Real const> const& stochfluxz,
                     amrex::Array4<amrex::Real const> const& rho, amrex::Real dzinv,
                     amrex::ParserExecutor<1> const& Dfn,
                     amrex::ParserExecutor<1> const& sigfn,
                     int dom_lo, int dom_hi, int bc_lo, int bc_hi)
{
    amrex::Real rhobdry = 0.;
    amrex::Real stofact = 1.;

    amrex::Real rhop, rhom, rhof, rhofn;

    if (k == dom_lo && bc_lo == amrex::BCType::ext_dir) {

        stofact = std::sqrt(2.);
        rhom = 2.*rhobdry - rho(i,j,k,0);
        rhop = rho(i,j,k,0);
        rhof = rhobdry;
        rhofn = rhobdry;

    } else if (k == dom_hi+1 && bc_hi == amrex::BCType::ext_dir) {

        stofact = std::sqrt(2.);
        rhop = 2.*rhobdry - rho(i,j,k-1,0);
        rhom = rho(i,j,k-1,0);
        rhof = rhobdry;
        rhofn = rhobdry;

    } else {

        rhop = rho(i,j,k,0);
        rhom = rho(i,j,k-1,0);
        rhof = 0.5*(rhop+rhom);
        rhofn = 0.5*(amrex::max(rhop,amrex::Real(0.))+amrex::max(rhom,amrex::Real(0.)));

    }

    fluxz(i,j,k,0) = Dfn(rhof)*(rhop-rhom) * dzinv
                   + stofact * stochfluxz(i,j,k) * std::sqrt(amrex::max(sigfn(rhofn), amrex::Real(0.)));
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void update_rho (int i, int j, int k,
                 amrex::Array4<amrex::Real const> const& rhoold,
                 amrex::Array4<amrex::Real      > const& rhonew,
                 AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fluxx,
                              amrex::Array4<amrex::Real const> const& fluxy,
                              amrex::Array4<amrex::Real const> const& fluxz),
                 amrex::Real dt,
                 AMREX_D_DECL(amrex::Real dxinv,
                              amrex::Real dyinv,
                              amrex::Real dzinv))
{
    rhonew(i,j,k,0) = rhoold(i,j,k,0)
        +  dt * dxinv * (fluxx(i+1,j  ,k  ) - fluxx(i,j,k))
        +  dt * dyinv * (fluxy(i  ,j+1,k  ) - fluxy(i,j,k))
#if (AMREX_SPACEDIM > 2)
        +  dt * dzinv * (fluxz(i  ,j  ,k+1) - fluxz(i,j,k))
#endif
        ;
}

#endif
