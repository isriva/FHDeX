#include "SpatialCorrelation.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <cmath>
#include <fstream>

using namespace amrex;

void
SpatialCorrelation::ReadParameters (ParmParse& pp)
{
    pp.query("corr_x_star", m_x_star);
    pp.query("corr_int", m_int);
    pp.query("corr_n_skip", m_n_skip);
    pp.query("corr_plot_file", m_plot_file);
}

void
SpatialCorrelation::Init (Geometry const& geom, Real Lambda,
                          GpuArray<int,AMREX_SPACEDIM> const& ens_dir, Real rho_scale)
{
    if (!Enabled()) { return; }

    m_geom = geom;
    m_Lambda = Lambda;
    m_rho_scale = rho_scale;
    const Box& domain = geom.Domain();
    m_nx = domain.length(0);
    m_nrows = 1;
    for (int d = 1; d < AMREX_SPACEDIM; ++d) { m_nrows *= domain.length(d); }

    if (ens_dir[0]) { Abort("corr_*: x must not be an ensemble direction"); }
    const Real xlo = geom.ProbLo(0);
    const Real xhi = geom.ProbHi(0);
    if (m_x_star < xlo || m_x_star >= xhi) {
        Abort("corr_x_star must be set inside the domain along x");
    }
    m_i_star = domain.smallEnd(0)
             + static_cast<int>(std::floor((m_x_star - xlo)/geom.CellSize(0)));

    m_sum1.assign(m_nx, 0.);
    m_sum2.assign(m_nx, 0.);
    m_rho_star.resize(m_nrows);
    m_sum_star = 0.;
    m_nsamples = 0;

    Print() << "Spatial correlation: x* = " << m_x_star << " (cell " << m_i_star
            << "), sampled every " << m_int << " steps after step " << m_n_skip
            << ", over " << m_nrows << " rows\n";
}

void
SpatialCorrelation::Sample (int step, MultiFab const& rho)
{
    if (!Enabled() || step <= m_n_skip || (step - m_n_skip) % m_int != 0) { return; }

    const Box& domain = m_geom.Domain();
    const int xlo = domain.smallEnd(0);
    const int istar = m_i_star;
    const Real scale = m_rho_scale;
#if (AMREX_SPACEDIM > 1)
    const int ylo = domain.smallEnd(1);
#endif
#if (AMREX_SPACEDIM > 2)
    const int ny = domain.length(1);
    const int zlo = domain.smallEnd(2);
#endif
    // row index of a cell: p = (j - ylo) + ny*(k - zlo)
    auto row = [=] AMREX_GPU_DEVICE (int j, int k) noexcept -> int
    {
        int p = 0;
#if (AMREX_SPACEDIM > 1)
        p = j - ylo;
#else
        amrex::ignore_unused(j);
#endif
#if (AMREX_SPACEDIM > 2)
        p += ny*(k - zlo);
#else
        amrex::ignore_unused(k);
#endif
        return p;
    };

    // gather rho(x*) of every row
    Real* rs = m_rho_star.dataPtr();
    const int nrows = m_nrows;
    ParallelFor(nrows, [=] AMREX_GPU_DEVICE (int n) noexcept { rs[n] = 0.; });
    for (MFIter mfi(rho); mfi.isValid(); ++mfi) {
        Box bx = mfi.validbox();
        if (istar < bx.smallEnd(0) || istar > bx.bigEnd(0)) { continue; }
        bx.setSmall(0, istar);
        bx.setBig(0, istar);
        auto const& r = rho.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            rs[row(j,k)] = scale*r(i,j,k,0);
        });
    }
    Gpu::streamSynchronize();
    Gpu::HostVector<Real> rs_host(nrows);
    Gpu::copy(Gpu::deviceToHost, m_rho_star.begin(), m_rho_star.end(), rs_host.begin());
    ParallelDescriptor::ReduceRealSum(rs_host.dataPtr(), nrows);
    Gpu::copy(Gpu::hostToDevice, rs_host.begin(), rs_host.end(), m_rho_star.begin());
    for (int n = 0; n < nrows; ++n) { m_sum_star += rs_host[n]; }

    // accumulate rho(x) and rho(x*) rho(x) (local partial sums)
    Real* s1 = m_sum1.dataPtr();
    Real* s2 = m_sum2.dataPtr();
    for (MFIter mfi(rho); mfi.isValid(); ++mfi) {
        auto const& r = rho.const_array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real v = scale*r(i,j,k,0);
            HostDevice::Atomic::Add(&s1[i-xlo], v);
            HostDevice::Atomic::Add(&s2[i-xlo], v*rs[row(j,k)]);
        });
    }
    Gpu::streamSynchronize();
    m_nsamples += nrows;
}

void
SpatialCorrelation::Write () const
{
    if (!Enabled() || m_nsamples == 0) { return; }

    Gpu::HostVector<Real> s1(m_nx), s2(m_nx);
    Gpu::copy(Gpu::deviceToHost, m_sum1.begin(), m_sum1.end(), s1.begin());
    Gpu::copy(Gpu::deviceToHost, m_sum2.begin(), m_sum2.end(), s2.begin());
    ParallelDescriptor::ReduceRealSum(s1.dataPtr(), m_nx);
    ParallelDescriptor::ReduceRealSum(s2.dataPtr(), m_nx);

    if (!ParallelDescriptor::IOProcessor()) { return; }

    const auto dx = m_geom.CellSizeArray();
    Real dvperp = 1.;
    for (int d = 1; d < AMREX_SPACEDIM; ++d) { dvperp *= dx[d]; }
    const Real dv = dx[0]*dvperp;
    const Real n = static_cast<Real>(m_nsamples);
    const Real mstar = m_sum_star/n;

    std::ofstream f(m_plot_file + ".dat");
    f.precision(12);
    f << "# spatial correlation C(x*,x) = <delta rho(x*) delta rho(x)>, x* = " << m_x_star
      << " (cell " << m_i_star << "), " << m_nsamples << " samples (times x rows)\n";
    f << "# Lambda*dV*C at x = x* -> sigma/(2D); Lambda*dVperp*C at x != x* -> long-range C_LR(x*,x)\n";
    f << "# x i <rho(x)> C Lambda*dV*C Lambda*dVperp*C nsamples\n";
    const int xlo = m_geom.Domain().smallEnd(0);
    for (int i = 0; i < m_nx; ++i) {
        const Real m1 = s1[i]/n;
        const Real c = s2[i]/n - mstar*m1;
        const Real x = m_geom.ProbLo(0) + (i + 0.5)*dx[0];
        f << x << " " << i + xlo << " " << m1 << " " << c << " "
          << m_Lambda*dv*c << " " << m_Lambda*dvperp*c << " " << m_nsamples << "\n";
    }
}
