#ifndef IntermediateScatteringFunction_H_
#define IntermediateScatteringFunction_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include "StructFact.H"

#include <deque>
#include <memory>
#include <string>

struct ISFPencilModes;

// Intermediate scattering function F(k,tau) = <n_hat(k,t0+tau) n_hat(k,t0)^*>/N
// of phi (component 0, level 0) for x wave vectors k = 2 pi q/Lx, normalized
// like the static structure factor so that F(k,0) = S(k). Time averages over
// rolling windows that start every isf_window_stride steps after
// isf_relax_time; each window is measured every isf_dt_meas steps up to a lag
// of isf_t_max steps, and only complete windows are accumulated.
// isf_do_1D = 1 transforms each x pencil separately and averages over the
// pencils (one realization per pencil when the other directions are ensemble
// directions); isf_do_1D = 0 uses the ky = kz = 0 modes of the full FFT.
class IntermediateScatteringFunction
{
public:
    IntermediateScatteringFunction ();
    ~IntermediateScatteringFunction ();

    IntermediateScatteringFunction (const IntermediateScatteringFunction&) = delete;
    IntermediateScatteringFunction& operator= (const IntermediateScatteringFunction&) = delete;

    void ReadParameters (amrex::ParmParse& pp);

    [[nodiscard]] bool Enabled () const { return m_n_modes > 0; }

    // var_scaling is the structure factor's: S = |FFT(phi)|^2/M / var_scaling
    void Init (amrex::Geometry const& geom, amrex::BoxArray const& ba,
               amrex::DistributionMapping const& dm, amrex::Real var_scaling);

    void Sample (int step, amrex::Real time, amrex::MultiFab const& phi);

    void Write () const;

private:
    struct ActiveWindow {
        int start_step = 0;
        amrex::Real start_time = 0.;
        amrex::Vector<amrex::Real> anchor_real;
        amrex::Vector<amrex::Real> anchor_imag;
        amrex::Vector<amrex::Real> sample_real;
        amrex::Vector<amrex::Real> sample_imag;
        amrex::Vector<amrex::Real> sample_lag_time;
        amrex::Vector<int> sample_seen;
    };

    void ExtractModes (amrex::MultiFab const& phi,
                       amrex::Vector<amrex::Real>& real,
                       amrex::Vector<amrex::Real>& imag);
    void Accumulate (amrex::Vector<amrex::Real> const& current_real,
                     amrex::Vector<amrex::Real> const& current_imag,
                     int step, amrex::Real time, bool starts_window);

    // inputs
    int m_n_modes = 0;
    int m_q_min = 1;
    int m_q_stride = 1;
    int m_dt_meas = 0;
    int m_t_max = 0;
    int m_window_stride = 0;
    int m_relax_time = 0;
    int m_do_1D = 0;
    std::string m_plot_file {"isf"};

    bool m_initialized = false;
    int m_last_step = -1;
    int m_n_lags = 0;
    int m_npencils = 1;
    amrex::Real m_product_scaling = 1.;
    amrex::Real m_lx = 1.;
    amrex::Vector<int> m_q_indices;

    StructFact m_fft_full;
    std::unique_ptr<ISFPencilModes> m_pencil_modes;

    std::deque<ActiveWindow> m_windows;
    amrex::Vector<amrex::Real> m_accum_real;
    amrex::Vector<amrex::Real> m_accum_imag;
    amrex::Vector<amrex::Real> m_accum_lag_time;
    amrex::Vector<int> m_accum_count;
    mutable int m_written_count = 0;   // windows in the last written output
};

#endif
