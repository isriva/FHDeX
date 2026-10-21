#include "myfunc.H"
#include "mykernel.H"
#include <AMReX_BCRec.H>
#include <AMReX_Math.H>
#include <AMReX_Reduce.H>
#include "rng_functions.H"

#include <fstream>
#include <limits>

using namespace amrex;

void advance_rho (MultiFab& rho_old,
                  MultiFab& rho_new,
                  Array<MultiFab, AMREX_SPACEDIM>& flux,
                  Array<MultiFab, AMREX_SPACEDIM>& stochFlux,
                  Real dt,
                  Real Lambda,
                  Real dorand,
                  ParserExecutor<1> const& Dfn,
                  ParserExecutor<1> const& sigfn,
                  Geometry const& geom,
                  BCRec const& bc,
                  GpuArray<Real,AMREX_SPACEDIM> const& rho_bc_lo,
                  GpuArray<Real,AMREX_SPACEDIM> const& rho_bc_hi,
                  GpuArray<int,AMREX_SPACEDIM> const& ens_dir)
{
    AMREX_D_TERM(const Real dxinv = geom.InvCellSize(0);,
                 const Real dyinv = geom.InvCellSize(1);,
                 const Real dzinv = geom.InvCellSize(2););

    const Box& domain_bx = geom.Domain();
    const auto dom_lo = lbound(domain_bx);
    const auto dom_hi = ubound(domain_bx);

    // white noise averaged over a face and a time step; the amplitude
    // sqrt(sigma(rho_f)) is applied in compute_flux_*
    Real variance = dxinv*dyinv/(dt*Lambda);

#if(AMREX_SPACEDIM > 2)
    variance *=dzinv;
#endif

    if(dorand != 0.){
        for (int d=0;d<AMREX_SPACEDIM;d++) {
            // no transport along an ensemble direction, so no noise either
            if (ens_dir[d]) {
                stochFlux[d].setVal(0.);
            } else {
                MultiFabFillRandom(stochFlux[d], 0, variance, geom);
            }
        }
    } else {
        for (int d=0;d<AMREX_SPACEDIM;d++) {
            stochFlux[d].setVal(0.);
        }
    }

    // Compute fluxes one grid at a time
    for ( MFIter mfi(rho_old); mfi.isValid(); ++mfi )
    {
        const Box& xbx = mfi.nodaltilebox(0);
        auto const& fluxx = flux[0].array(mfi);
        const Box& ybx = mfi.nodaltilebox(1);
        auto const& fluxy = flux[1].array(mfi);
#if (AMREX_SPACEDIM > 2)
        const Box& zbx = mfi.nodaltilebox(2);
        auto const& fluxz = flux[2].array(mfi);
#endif
        auto const& stochfluxx = stochFlux[0].const_array(mfi);
        auto const& stochfluxy = stochFlux[1].const_array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& stochfluxz = stochFlux[2].const_array(mfi);
#endif

        auto const& rho = rho_old.const_array(mfi);

        // fluxes along ensemble directions are zero, so the realizations
        // (rows of cells along them) are independent
        if (ens_dir[0]) {
            flux[0][mfi].setVal<RunOn::Device>(0., xbx);
        } else {
            amrex::ParallelFor(xbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    compute_flux_x(i,j,k,fluxx,stochfluxx,rho,dxinv,Dfn,sigfn,
                                   dom_lo.x, dom_hi.x, bc.lo(0), bc.hi(0),
                                   rho_bc_lo[0], rho_bc_hi[0]);
                });
        }

        if (ens_dir[1]) {
            flux[1][mfi].setVal<RunOn::Device>(0., ybx);
        } else {
            amrex::ParallelFor(ybx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    compute_flux_y(i,j,k,fluxy,stochfluxy,rho,dyinv,Dfn,sigfn,
                                   dom_lo.y, dom_hi.y, bc.lo(1), bc.hi(1),
                                   rho_bc_lo[1], rho_bc_hi[1]);
                });
        }
#if (AMREX_SPACEDIM > 2)
        if (ens_dir[2]) {
            flux[2][mfi].setVal<RunOn::Device>(0., zbx);
        } else {
            amrex::ParallelFor(zbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    compute_flux_z(i,j,k,fluxz,stochfluxz,rho,dzinv,Dfn,sigfn,
                                   dom_lo.z, dom_hi.z, bc.lo(2), bc.hi(2),
                                   rho_bc_lo[2], rho_bc_hi[2]);
                });
        }
#endif
    }

    // Advance the solution one grid at a time
    for ( MFIter mfi(rho_old); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& fluxx = flux[0].const_array(mfi);
        auto const& fluxy = flux[1].const_array(mfi);
#if (AMREX_SPACEDIM > 2)
        auto const& fluxz = flux[2].const_array(mfi);
#endif
        auto const& rhoOld = rho_old.const_array(mfi);
        auto const& rhoNew = rho_new.array(mfi);

        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            update_rho(i,j,k,rhoOld,rhoNew,
                       AMREX_D_DECL(fluxx,fluxy,fluxz),
                       dt,
                       AMREX_D_DECL(dxinv,dyinv,dzinv));
        });
    }
}

Real est_time_step (MultiFab const& rho, Real cfl,
                    ParserExecutor<1> const& Dfn,
                    Geometry const& geom,
                    BCRec const& bc,
                    GpuArray<Real,AMREX_SPACEDIM> const& rho_bc_lo,
                    GpuArray<Real,AMREX_SPACEDIM> const& rho_bc_hi,
                    GpuArray<int,AMREX_SPACEDIM> const& ens_dir)
{
    const Real* dx = geom.CellSize();

    // Forward Euler is stable when dt * lambda_max <= 2, with
    //   lambda_max <= max D(rho) * keff2_max,  keff2_max = sum_d 4/dx_d^2
    // (ensemble directions carry no flux, so they do not limit dt).
    // max D is taken over the cell values and, on wall faces, over D(rho_wall)
    // of each wall. cfl is the fraction of this limit that is used.
    Real keff2_max = 0.;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (!ens_dir[d]) { keff2_max += 4./(dx[d]*dx[d]); }
    }

    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    for (MFIter mfi(rho); mfi.isValid(); ++mfi) {
        auto const& r = rho.const_array(mfi);
        reduce_op.eval(mfi.validbox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                return { Dfn(r(i,j,k,0)) };
            });
    }
    Real Dmax = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealMax(Dmax);

    // D at the wall densities (the executor is device-callable, so evaluate
    // it in a device reduction)
    {
        GpuArray<Real,2*AMREX_SPACEDIM> rw;
        GpuArray<int,2*AMREX_SPACEDIM> is_wall;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            rw[2*d] = rho_bc_lo[d];  is_wall[2*d]   = (bc.lo(d) == BCType::ext_dir && !ens_dir[d]);
            rw[2*d+1] = rho_bc_hi[d]; is_wall[2*d+1] = (bc.hi(d) == BCType::ext_dir && !ens_dir[d]);
        }
        ReduceOps<ReduceOpMax> wop;
        ReduceData<Real> wdata(wop);
        wop.eval(2*AMREX_SPACEDIM, wdata,
            [=] AMREX_GPU_DEVICE (int n) -> ReduceTuple
            {
                return { is_wall[n] ? Dfn(rw[n]) : std::numeric_limits<Real>::lowest() };
            });
        Dmax = amrex::max(Dmax, amrex::get<0>(wdata.value()));
    }

    const Real lambda_max = Dmax * keff2_max;

    Real dt_est = std::numeric_limits<Real>::max();
    if (lambda_max > 0.) {
        dt_est = cfl * 2. / lambda_max;
    }
    return dt_est;
}

void write_struct_fact (StructFact& structFact, Geometry const& geom,
                        int step, Real time, int nsamples)
{
    // shifted S(k) on the k-space grid (plt_SF_mag...), leaves cov_mag finalized
    structFact.WritePlotFile(step, time, "plt_SF");

    // Shell average. After the shift, index i holds wavenumber index i - n/2,
    // k_d = 2 pi (i - n_d/2)/L_d. Shell j holds |k| in [(j-1/2)dk, (j+1/2)dk)
    // with dk = 2 pi/max L, up to the smallest Nyquist wavenumber.
    const Box& domain = geom.Domain();
    GpuArray<int,AMREX_SPACEDIM> center;
    GpuArray<Real,AMREX_SPACEDIM> kfac;
    Real lmax = 0.0, knyq = std::numeric_limits<Real>::max();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        center[d] = domain.length(d)/2;
        kfac[d] = 2.0*amrex::Math::pi<Real>()/geom.ProbLength(d);
        lmax = amrex::max(lmax, geom.ProbLength(d));
        knyq = amrex::min(knyq, amrex::Math::pi<Real>()/geom.CellSize(d));
    }
    const Real dk = 2.0*amrex::Math::pi<Real>()/lmax;
    const int jmax = static_cast<int>(std::floor(knyq/dk - 0.5));
    if (jmax < 1) { return; }

    Gpu::DeviceVector<Real> ssum(jmax+1, 0.0), ksum(jmax+1, 0.0);
    Gpu::DeviceVector<int> scnt(jmax+1, 0);
    Real* sp = ssum.dataPtr();
    Real* kp = ksum.dataPtr();
    int* cp = scnt.dataPtr();
    for (MFIter mfi(structFact.cov_mag); mfi.isValid(); ++mfi) {
        auto const& s = structFact.cov_mag.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const int idx[3] = {i, j, k};
            Real k2 = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                const Real kd = kfac[d]*(idx[d] - center[d]);
                k2 += kd*kd;
            }
            const Real kmag = std::sqrt(k2);
            const int b = static_cast<int>(std::floor(kmag/dk + 0.5));
            if (kmag > 0.0 && b >= 1 && b <= jmax) {
                amrex::HostDevice::Atomic::Add(&sp[b], s(i,j,k,0));
                amrex::HostDevice::Atomic::Add(&kp[b], kmag);
                amrex::HostDevice::Atomic::Add(&cp[b], 1);
            }
        });
    }
    Gpu::HostVector<Real> sh(jmax+1), kh(jmax+1);
    Gpu::HostVector<int> ch(jmax+1);
    Gpu::copy(Gpu::deviceToHost, ssum.begin(), ssum.end(), sh.begin());
    Gpu::copy(Gpu::deviceToHost, ksum.begin(), ksum.end(), kh.begin());
    Gpu::copy(Gpu::deviceToHost, scnt.begin(), scnt.end(), ch.begin());
    ParallelDescriptor::ReduceRealSum(sh.dataPtr(), jmax+1);
    ParallelDescriptor::ReduceRealSum(kh.dataPtr(), jmax+1);
    ParallelDescriptor::ReduceIntSum(ch.dataPtr(), jmax+1);

    if (ParallelDescriptor::IOProcessor()) {
        const std::string fname = amrex::Concatenate("sf_kshell_", step, 7) + ".txt";
        std::ofstream f(fname);
        f.precision(10);
        f << "# shell-averaged structure factor S(k) = Lambda |drho_hat|^2/(rhobar V) over " << nsamples
          << " samples, t = " << time << ", dk = " << dk << "\n";
        f << "# k(mean over shell) S nmodes\n";
        for (int b = 1; b <= jmax; ++b) {
            if (ch[b] > 0) { f << kh[b]/ch[b] << " " << sh[b]/ch[b] << " " << ch[b] << "\n"; }
        }
    }
    amrex::Print() << "Wrote structure factor (" << nsamples << " samples) at step " << step << "\n";
}
