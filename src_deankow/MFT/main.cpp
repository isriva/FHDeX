
#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Parser.H>
#include <AMReX_BCRec.H>

#include <iomanip>
#include <limits>
#include <string>

#include "rng_functions.H"

#include "myfunc.H"
#include "mykernel.H"
#include "InitProfile.H"
#include "StructFact.H"
#include "IntermediateScatteringFunction.H"
#include "SpatialCorrelation.H"
#include "chrono"

using namespace amrex;
using namespace std::chrono;

// Solves the stochastic PDE of macroscopic fluctuation theory,
//   d rho/dt = div[ D(rho) grad rho + sqrt(sigma(rho)/Lambda) eta ],
// on a single level with the finite-volume / Euler-Maruyama scheme of
// ../interaction (see mykernel.H for the face fluxes).
void main_main ();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    {
        // timer for profiling
        BL_PROFILE("main()");

        // wallclock time
        const auto strt_total = amrex::second();

        main_main();

        // wallclock time
        auto end_total = amrex::second() - strt_total;

        // print wallclock time
        ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
        amrex::Print() << "\nTotal Time: " << end_total << '\n';
    }

    amrex::Finalize();
}

void main_main ()
{
    ////////////////
    // runtime parameters

    // maximum number of steps and stop time
    int max_step = std::numeric_limits<int>::max();
    Real stop_time = std::numeric_limits<Real>::max();

    int seed = 0;
    Real dorand = 1.;

    // large parameter multiplying the noise variance, sqrt(sigma/Lambda)
    Real Lambda = -1.;

    // D(rho) and sigma(rho), functions of the variable rho
    std::string D_rho;
    std::string sigma_rho;

    // uniform initial density, or a piecewise-constant profile along x
    Real rho0 = 0.;
    InitProfile init_profile;

    Vector<int> bc_lo(AMREX_SPACEDIM,0);
    Vector<int> bc_hi(AMREX_SPACEDIM,0);

    // densities prescribed on the lo/hi walls (bc = 3) of each direction
    Vector<Real> rho_bc_lo_in(AMREX_SPACEDIM,0.);
    Vector<Real> rho_bc_hi_in(AMREX_SPACEDIM,0.);

    // ensemble directions: no transport along them, so each row of cells
    // along them is an independent realization
    GpuArray<int,AMREX_SPACEDIM> ens_dir;

    // structure factor of rho, accumulated every struct_fact_int steps
    // after n_steps_skip (0 = off) and written with each plotfile
    int struct_fact_int = 0;
    int n_steps_skip = 0;

    // intermediate scattering function of rho (isf_* inputs)
    IntermediateScatteringFunction isf;

    // spatial correlation <delta rho(x*) delta rho(x)> (corr_* inputs)
    SpatialCorrelation corr;

    Vector<int> n_cell(AMREX_SPACEDIM);
    Vector<int> max_grid_size(AMREX_SPACEDIM,32);

    // plotfile prefix and frequency
    std::string plot_file {"plt"};
    int plot_int = -1;

    // fraction of the explicit (forward Euler) stability limit used for dt
    Real cfl = 0.7;

    // if > 0, used as the time step instead of the cfl estimate
    Real fixed_dt = -1.;

    {
        ParmParse pp;  // Traditionally, max_step and stop_time do not have prefix.
        pp.query("max_step", max_step);
        pp.query("stop_time", stop_time);

        pp.query("seed", seed);
        pp.query("dorand", dorand);

        pp.query("Lambda", Lambda);
        if (Lambda <= 0.) {
            Abort("Lambda must be set to a positive value in the inputs");
        }
        pp.get("D_rho", D_rho);
        pp.get("sigma_rho", sigma_rho);

        pp.query("rho0", rho0);

        pp.query("struct_fact_int", struct_fact_int);
        pp.query("n_steps_skip", n_steps_skip);
        isf.ReadParameters(pp);
        corr.ReadParameters(pp);

        // read in BC; see Src/Base/AMReX_BC_TYPES.H for supported types
        pp.queryarr("bc_lo", bc_lo);
        pp.queryarr("bc_hi", bc_hi);
        pp.queryarr("rho_bc_lo", rho_bc_lo_in, 0, AMREX_SPACEDIM);
        pp.queryarr("rho_bc_hi", rho_bc_hi_in, 0, AMREX_SPACEDIM);

        Vector<int> ens(AMREX_SPACEDIM, 0);
        pp.queryarr("is_ensemble_dir", ens, 0, AMREX_SPACEDIM);
        int n_ens = 0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            ens_dir[d] = (ens[d] != 0);
            n_ens += ens_dir[d];
        }
        if (n_ens == AMREX_SPACEDIM) {
            amrex::Abort("is_ensemble_dir cannot be set in every direction");
        }

        // optional piecewise-constant initial profile along x
        std::vector<Real> prof_x, prof_val;
        pp.queryarr("init_profile_x", prof_x);
        pp.queryarr("init_profile_val", prof_val);
        if (!prof_val.empty()) {
            const int nseg = static_cast<int>(prof_val.size());
            if (nseg > InitProfile::max_seg) {
                amrex::Abort("init_profile_val has more than " + std::to_string(InitProfile::max_seg) + " values");
            }
            if (static_cast<int>(prof_x.size()) != nseg-1) {
                amrex::Abort("init_profile_x must have one fewer entry than init_profile_val");
            }
            init_profile.nseg = nseg;
            for (int n = 0; n < nseg; ++n) {
                init_profile.val[n] = prof_val[n];
            }
            for (int n = 0; n < nseg-1; ++n) {
                if (n > 0 && prof_x[n] <= prof_x[n-1]) { amrex::Abort("init_profile_x must be increasing"); }
                init_profile.x[n] = prof_x[n];
            }
        } else if (!prof_x.empty()) {
            amrex::Abort("init_profile_x given without init_profile_val");
        }
    }

    {
        ParmParse pp("amr");

        pp.getarr("n_cell", n_cell, 0, AMREX_SPACEDIM);
        pp.queryarr("max_grid_size", max_grid_size, 0, AMREX_SPACEDIM);
        pp.query("plot_file", plot_file);
        pp.query("plot_int", plot_int);
    }

    {
        ParmParse pp("adv");

        pp.query("cfl", cfl);
        pp.query("fixed_dt", fixed_dt);
    }

    // make BoxArray and Geometry; prob_lo, prob_hi and is_periodic are read
    // from geometry.* in the inputs
    BoxArray ba;
    Geometry geom;
    {
        IntVect dom_lo(AMREX_D_DECL(          0,           0,           0));
        IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));
        Box domain(dom_lo, dom_hi);

        ba.define(domain);
        ba.maxSize(IntVect(max_grid_size));

        geom.define(domain);
    }

    // periodic (bc = 0) or a Dirichlet wall (bc = 3) with the density
    // rho_bc_lo/hi of that side (default 0), as in ../interaction
    BCRec bc;
    GpuArray<Real,AMREX_SPACEDIM> rho_bc_lo, rho_bc_hi;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        if (bc_lo[idim] != BCType::int_dir && bc_lo[idim] != BCType::ext_dir) {
            amrex::Abort("Invalid bc_lo: use 0 (periodic) or 3 (Dirichlet wall with density rho_bc_lo)");
        }
        if (bc_hi[idim] != BCType::int_dir && bc_hi[idim] != BCType::ext_dir) {
            amrex::Abort("Invalid bc_hi: use 0 (periodic) or 3 (Dirichlet wall with density rho_bc_hi)");
        }
        if (rho_bc_lo_in[idim] < 0. || rho_bc_hi_in[idim] < 0.) {
            amrex::Abort("rho_bc_lo/rho_bc_hi must be non-negative");
        }
        rho_bc_lo[idim] = rho_bc_lo_in[idim];
        rho_bc_hi[idim] = rho_bc_hi_in[idim];
        if (bc_lo[idim] == BCType::ext_dir) {
            amrex::Print() << "Dirichlet wall at the lo side of direction " << idim
                           << ", rho = " << rho_bc_lo[idim] << "\n";
        }
        if (bc_hi[idim] == BCType::ext_dir) {
            amrex::Print() << "Dirichlet wall at the hi side of direction " << idim
                           << ", rho = " << rho_bc_hi[idim] << "\n";
        }
        const bool per = (bc_lo[idim] == BCType::int_dir && bc_hi[idim] == BCType::int_dir);
        if (per != geom.isPeriodic(idim)) {
            amrex::Abort("bc_lo/bc_hi = 0 must match geometry.is_periodic = 1 in each direction");
        }
        bc.setLo(idim, bc_lo[idim]);
        bc.setHi(idim, bc_hi[idim]);
    }

    DistributionMapping dm(ba);

    /////////////////////////////////////////
    //Initialise rngs
    /////////////////////////////////////////
    if (seed > 0) {
        // initializes the seed for C++ random number calls
        InitRandom(seed+ParallelDescriptor::MyProc(),
                   ParallelDescriptor::NProcs(),
                   seed+ParallelDescriptor::MyProc());
    } else if (seed == 0) {
        // initializes the seed for C++ random number calls based on the clock
        auto now = time_point_cast<nanoseconds>(system_clock::now());
        int randSeed = now.time_since_epoch().count();
        // broadcast the same root seed to all processors
        ParallelDescriptor::Bcast(&randSeed,1,ParallelDescriptor::IOProcessorNumber());
        InitRandom(randSeed+ParallelDescriptor::MyProc(),
                   ParallelDescriptor::NProcs(),
                   randSeed+ParallelDescriptor::MyProc());
    } else {
        Abort("Must supply non-negative seed");
    }

    // D(rho) and sigma(rho); the Parsers own the compiled code and must
    // outlive the executors
    Parser D_parser(D_rho);
    D_parser.registerVariables({"rho"});
    const auto Dfn = D_parser.compile<1>();

    Parser sigma_parser(sigma_rho);
    sigma_parser.registerVariables({"rho"});
    const auto sigfn = sigma_parser.compile<1>();

    amrex::Print() << "D(rho) = " << D_rho << ", sigma(rho) = " << sigma_rho
                   << ", Lambda = " << Lambda << "\n";

    // after advancing we use "swap"
    MultiFab rho_old(ba, dm, 1, 1);
    MultiFab rho_new(ba, dm, 1, 1);

    const auto problo = geom.ProbLoArray();
    const auto dx     = geom.CellSizeArray();
    const Real cellvol = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

    for (MFIter mfi(rho_new); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        auto const& rho_arr = rho_new.array(mfi);
        const InitProfile prof = init_profile;
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            init_rho(i,j,k,rho_arr,dx,problo,rho0,prof);
        });
    }
    rho_new.FillBoundary(geom.periodicity());
    MultiFab::Copy(rho_old, rho_new, 0, 0, 1, 1);

    // face fluxes
    Array<MultiFab, AMREX_SPACEDIM> flux;
    Array<MultiFab, AMREX_SPACEDIM> stochFlux;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        BoxArray edge_ba = ba;
        edge_ba.surroundingNodes(d);
        flux[d].define(edge_ba, dm, 1, 0);
        flux[d].setVal(0.);
        stochFlux[d].define(edge_ba, dm, 1, 0);
        stochFlux[d].setVal(0.);
    }

    // S(k) = Lambda |drho_hat(k)|^2 / (rhobar V), the analogue of |n_hat|^2/N
    // with cell counts n = Lambda rho dV, so a Poisson field gives S = 1.
    // StructFact accumulates |FFT(rho)|^2 / M and multiplies by 1/var_scaling,
    // so var_scaling = rhobar/(Lambda dV), with rhobar the initial mean density.
    const Real rhobar = rho_new.sum(0) / geom.Domain().d_numPts();
    if ((struct_fact_int > 0 || isf.Enabled()) && rhobar <= 0.) {
        amrex::Abort("the structure factor needs a positive mean initial density");
    }
    const Real sf_var_scaling = rhobar/(Lambda*cellvol);

    StructFact structFact;
    int sf_nsamples = 0;
    int sf_last_write_step = -1;
    if (struct_fact_int > 0) {
        const Vector<std::string> names {"rho"};
        const Vector<Real> var_scaling {sf_var_scaling};
        structFact.define(ba, dm, names, var_scaling);
        amrex::Print() << "Structure factor: sampled every " << struct_fact_int
                       << " steps after step " << n_steps_skip << "\n";
    }
    if (isf.Enabled()) { isf.Init(geom, ba, dm, sf_var_scaling); }
    if (corr.Enabled()) { corr.Init(geom, Lambda, ens_dir); }

    auto WritePlotFile = [&] (int step, Real time)
    {
        const std::string& plotfilename = amrex::Concatenate(plot_file, step, 6);
        amrex::Print() << "Writing plotfile " << plotfilename << "\n";
        MultiFab mf(ba, dm, 1, 0);
        MultiFab::Copy(mf, rho_new, 0, 0, 1, 0);
        WriteSingleLevelPlotfile(plotfilename, mf, {"rho"}, geom, time, step);
    };

    auto WriteStructFact = [&] (int step, Real time)
    {
        if (struct_fact_int <= 0 || sf_nsamples == 0 || step == sf_last_write_step) { return; }
        sf_last_write_step = step;
        write_struct_fact(structFact, geom, step, time, sf_nsamples);
    };

    Real cur_time = 0.;
    Real dt = 1.e100;
    int istep = 0;
    int last_plot_file_step = 0;

    if (plot_int > 0) {
        WritePlotFile(istep, cur_time);
    }

    // the ISF windows can start from the initial state
    isf.Sample(istep, cur_time, rho_new);

    for (int step = istep; step < max_step && cur_time < stop_time; ++step)
    {
        amrex::Print() << "\nSTEP " << step+1 << " starts ..." << std::endl;

        std::swap(rho_old, rho_new);
        rho_old.FillBoundary(geom.periodicity());

        // compute dt from the stability limit, change by at most a factor
        // 1.1 per step; a fixed dt overrides it
        constexpr Real change_max = 1.1;
        Real dt_new = amrex::min(est_time_step(rho_old, cfl, Dfn, geom, bc,
                                               rho_bc_lo, rho_bc_hi, ens_dir), change_max*dt);
        if (fixed_dt > 0.) { dt_new = fixed_dt; }
        if (cur_time + dt_new > stop_time - 1.e-3*dt_new) { dt_new = stop_time - cur_time; }
        dt = dt_new;

        advance_rho(rho_old, rho_new, flux, stochFlux, dt, Lambda, dorand,
                    Dfn, sigfn, geom, bc, rho_bc_lo, rho_bc_hi, ens_dir);

        cur_time += dt;
        istep = step+1;

        // integrate rho to check conservation
        Real sum_rho_old = rho_old.sum(0) * cellvol;
        Real sum_rho_new = rho_new.sum(0) * cellvol;

        amrex::Print() << "STEP " << step+1 << " ends." << " TIME = " << cur_time
                       << " DT = " << dt << " Int_old Int_new Diff (rho) = "    << std::setw(20) << std::setprecision(12)
                       << std::scientific <<  sum_rho_old << " " << std::setw(20) << std::setprecision(12)
                       << std::scientific <<  sum_rho_new << " " << std::setw(20) << std::setprecision(12)
                       << std::scientific << (sum_rho_new - sum_rho_old) << std::endl;

        if (struct_fact_int > 0 && step+1 > n_steps_skip
            && (step+1 - n_steps_skip) % struct_fact_int == 0) {
            MultiFab mf(ba, dm, 1, 0);
            MultiFab::Copy(mf, rho_new, 0, 0, 1, 0);
            structFact.FortStructure(mf);
            ++sf_nsamples;
        }
        isf.Sample(step+1, cur_time, rho_new);
        corr.Sample(step+1, rho_new);

        if (plot_int > 0 && (step+1) % plot_int == 0) {
            last_plot_file_step = step+1;
            WritePlotFile(step+1, cur_time);
            WriteStructFact(step+1, cur_time);
            isf.Write();
            corr.Write();
        }

        if (cur_time >= stop_time - 1.e-6*dt) { break; }
    }

    if (plot_int > 0 && istep > last_plot_file_step) {
        WritePlotFile(istep, cur_time);
    }
    WriteStructFact(istep, cur_time);
    isf.Write();
    corr.Write();
}
