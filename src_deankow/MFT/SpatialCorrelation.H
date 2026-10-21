#ifndef SpatialCorrelation_H_
#define SpatialCorrelation_H_

#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#include <string>

// Equal-time spatial correlation of density fluctuations along x,
//   C(x*, x) = <delta rho(x*) delta rho(x)>,
// averaged over time and over the rows of cells along the other directions
// (independent 1D realizations when those are ensemble directions), as in
// EvaluateSpatialCorrelations1D (src_compressible_stag/statsStag.cpp).
// Each sample (time, row) adds rho(x*), rho(x) and rho(x*) rho(x) to running
// sums, so C = <rho(x*) rho(x)> - <rho(x*)><rho(x)> about the time- and
// realization-averaged profile.
class SpatialCorrelation
{
public:
    void ReadParameters (amrex::ParmParse& pp);

    [[nodiscard]] bool Enabled () const { return m_int > 0; }

    // rho_scale multiplies the sampled field (e.g. phi -> particles per unit
    // length of a row in ../interaction); Lambda is used only to scale the
    // output columns
    void Init (amrex::Geometry const& geom, amrex::Real Lambda,
               amrex::GpuArray<int,AMREX_SPACEDIM> const& ens_dir,
               amrex::Real rho_scale = 1.);

    void Sample (int step, amrex::MultiFab const& rho);

    void Write () const;

private:
    // inputs
    amrex::Real m_x_star = -1.e300;
    int m_int = 0;
    int m_n_skip = 0;
    std::string m_plot_file {"spatial_corr"};

    amrex::Geometry m_geom;
    amrex::Real m_Lambda = 1.;
    amrex::Real m_rho_scale = 1.;
    int m_i_star = 0;
    int m_nx = 0;
    int m_nrows = 1;

    // sums over samples: rho(x*), and per x index rho(x), rho(x*) rho(x)
    amrex::Real m_sum_star = 0.;
    amrex::Gpu::DeviceVector<amrex::Real> m_sum1;
    amrex::Gpu::DeviceVector<amrex::Real> m_sum2;
    amrex::Gpu::DeviceVector<amrex::Real> m_rho_star;
    long m_nsamples = 0;
};

#endif
