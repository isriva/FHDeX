#include "IntermediateScatteringFunction.H"

#include <AMReX.H>
#include <AMReX_FFT.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <utility>

using namespace amrex;

// Batched 1D FFTs along x of every pencil (fixed j,k) of the level-0 domain.
// Returns the selected modes q of each pencil scaled by 1/sqrt(nx), stored as
// values[p*n_modes + m] with pencil p = (j - ylo) + ny*(k - zlo).
struct ISFPencilModes
{
    using FFTType = amrex::FFT::R2C<Real, amrex::FFT::Direction::forward>;
    using SpectralMF = FabArray<BaseFab<GpuComplex<Real>>>;

    int nx = 0;
    int n_modes = 0;
    int npencils = 1;
    Box domain;
    std::unique_ptr<FFTType> fft;
    SpectralMF spectral;
    Gpu::DeviceVector<int> q_to_mode;
    Gpu::DeviceVector<Real> device_real;
    Gpu::DeviceVector<Real> device_imag;

    void define (Geometry const& geom, Vector<int> const& q_indices, int npencils_in)
    {
        domain = geom.Domain();
        nx = domain.length(0);
        n_modes = static_cast<int>(q_indices.size());
        npencils = npencils_in;

        amrex::FFT::Info info;
        info.setOneDMode(true);
        fft = std::make_unique<FFTType>(domain, info);
        auto const& [ba_fft, dm_fft] = fft->getSpectralDataLayout();
        spectral.define(ba_fft, dm_fft, 1, 0);

        Vector<int> host_q_to_mode(nx, -1);
        for (int m = 0; m < n_modes; ++m) {
            host_q_to_mode[q_indices[m]] = m;
        }
        q_to_mode.resize(nx);
        Gpu::copy(Gpu::hostToDevice, host_q_to_mode.begin(), host_q_to_mode.end(),
                  q_to_mode.begin());

        device_real.resize(npencils*n_modes);
        device_imag.resize(npencils*n_modes);
    }

    void extract (MultiFab const& phi, Vector<Real>& real, Vector<Real>& imag)
    {
        BL_PROFILE("ISFPencilModes::extract");

        const int n_values = npencils*n_modes;
        real.assign(n_values, 0.);
        imag.assign(n_values, 0.);

        Real* real_ptr = device_real.dataPtr();
        Real* imag_ptr = device_imag.dataPtr();
        ParallelFor(n_values, [=] AMREX_GPU_DEVICE (int n) noexcept
        {
            real_ptr[n] = 0.;
            imag_ptr[n] = 0.;
        });

        fft->forward(phi, spectral, 0, 0);

#if (AMREX_SPACEDIM > 1)
        const int ylo = domain.smallEnd(1);
        const int ny = domain.length(1);
#endif
#if (AMREX_SPACEDIM > 2)
        const int zlo = domain.smallEnd(2);
#endif
        const int nx_loc = nx;
        const int n_modes_loc = n_modes;
        const Real scale = 1./std::sqrt(static_cast<Real>(nx));
        int const* q_to_mode_ptr = q_to_mode.dataPtr();

        for (MFIter mfi(spectral); mfi.isValid(); ++mfi) {
            auto const& spec = spectral.const_array(mfi);
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (i < 0 || i >= nx_loc) { return; }
                const int mode = q_to_mode_ptr[i];
                if (mode < 0) { return; }
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
                const int idx = p*n_modes_loc + mode;
                const auto z = spec(i,j,k,0);
                real_ptr[idx] = z.real()*scale;
                imag_ptr[idx] = z.imag()*scale;
            });
        }

        Gpu::streamSynchronize();
        Gpu::copy(Gpu::deviceToHost, device_real.begin(), device_real.end(), real.begin());
        Gpu::copy(Gpu::deviceToHost, device_imag.begin(), device_imag.end(), imag.begin());
        ParallelDescriptor::ReduceRealSum(real.data(), n_values);
        ParallelDescriptor::ReduceRealSum(imag.data(), n_values);
    }
};

IntermediateScatteringFunction::IntermediateScatteringFunction () = default;

IntermediateScatteringFunction::~IntermediateScatteringFunction () = default;

void
IntermediateScatteringFunction::ReadParameters (ParmParse& pp)
{
    pp.query("isf_n_modes", m_n_modes);
    pp.query("isf_q_min", m_q_min);
    pp.query("isf_q_stride", m_q_stride);
    pp.query("isf_dt_meas", m_dt_meas);
    pp.query("isf_t_max", m_t_max);
    pp.query("isf_window_stride", m_window_stride);
    pp.query("isf_relax_time", m_relax_time);
    pp.query("isf_plot_file", m_plot_file);
    // do_1D is the name used by the noninteracting ensemble code
    if (!pp.query("isf_do_1D", m_do_1D)) { pp.query("do_1D", m_do_1D); }
}

void
IntermediateScatteringFunction::Init (Geometry const& geom, BoxArray const& ba,
                                      DistributionMapping const& dm, Real var_scaling)
{
    if (!Enabled() || m_initialized) { return; }

    if (m_do_1D != 0 && m_do_1D != 1) { Abort("isf_do_1D must be 0 or 1"); }
    if (m_q_min < 1) { Abort("isf_q_min must be >= 1"); }
    if (m_q_stride < 1) { Abort("isf_q_stride must be >= 1"); }
    if (m_dt_meas <= 0) { Abort("isf_dt_meas must be > 0 steps"); }
    if (m_t_max < 0) { Abort("isf_t_max must be >= 0 steps"); }
    if (m_t_max % m_dt_meas != 0) { Abort("isf_t_max must be a multiple of isf_dt_meas"); }
    if (m_window_stride <= 0) { Abort("isf_window_stride must be > 0 steps"); }
    if (m_relax_time < 0) { Abort("isf_relax_time must be >= 0 steps"); }

    const Box& domain = geom.Domain();
    const int nx = domain.length(0);
    if (m_q_min + (m_n_modes-1)*m_q_stride > nx/2) {
        Abort("isf: the largest requested x mode isf_q_min + (isf_n_modes-1)*isf_q_stride exceeds nx/2");
    }
    if (m_do_1D == 1) {
        ParmParse pp_top;
        Vector<int> ens(AMREX_SPACEDIM, 0);
        pp_top.queryarr("is_ensemble_dir", ens, 0, AMREX_SPACEDIM);
        if (ens[0] != 0) { Abort("isf_do_1D = 1 needs x not to be an ensemble direction"); }
    }

    m_n_lags = m_t_max/m_dt_meas + 1;
    m_q_indices.resize(m_n_modes);
    for (int m = 0; m < m_n_modes; ++m) { m_q_indices[m] = m_q_min + m*m_q_stride; }
    m_lx = geom.ProbLength(0);

    // S = |a|^2/var_scaling with a = FFT(phi)/sqrt(M), so the product of
    // two such modes over var_scaling is F, and F(k,0) = S(k). In 1D mode,
    // a_p = FFT_x(phi_p)/sqrt(nx) and N_p = N/npencils give
    // n_hat_p n_hat_p^*/N_p = a_p a_p^*/var_scaling, so the average over
    // pencils uses the same factor.
    m_product_scaling = 1./var_scaling;

    m_npencils = 1;
    if (m_do_1D == 0) {
        const Vector<std::string> names {"phi"};
        const Vector<Real> scaling {var_scaling};
        m_fft_full.define(ba, dm, names, scaling);
    } else {
        for (int d = 1; d < AMREX_SPACEDIM; ++d) { m_npencils *= domain.length(d); }
        m_pencil_modes = std::make_unique<ISFPencilModes>();
        m_pencil_modes->define(geom, m_q_indices, m_npencils);
    }

    m_windows.clear();
    m_accum_real.assign(m_n_lags*m_n_modes, 0.);
    m_accum_imag.assign(m_n_lags*m_n_modes, 0.);
    m_accum_lag_time.assign(m_n_lags, 0.);
    m_accum_count.assign(m_n_lags, 0);

    m_initialized = true;
    Print() << "Intermediate scattering function: " << m_n_modes << " x modes q = "
            << m_q_min << " + " << m_q_stride << "*m, lags of " << m_dt_meas
            << " steps up to " << m_t_max << ", windows every " << m_window_stride
            << " steps from step " << m_relax_time
            << (m_do_1D ? ", per x pencil" : ", ky = kz = 0 modes") << "\n";
}

void
IntermediateScatteringFunction::ExtractModes (MultiFab const& phi,
                                              Vector<Real>& real, Vector<Real>& imag)
{
    if (m_do_1D == 1) {
        m_pencil_modes->extract(phi, real, imag);
        return;
    }

    real.assign(m_n_modes, 0.);
    imag.assign(m_n_modes, 0.);

    MultiFab density(phi.boxArray(), phi.DistributionMap(), 1, 0);
    MultiFab::Copy(density, phi, 0, 0, 1, 0);
    MultiFab dft_real(phi.boxArray(), phi.DistributionMap(), 1, 0);
    MultiFab dft_imag(phi.boxArray(), phi.DistributionMap(), 1, 0);
    m_fft_full.ComputeFFT(density, dft_real, dft_imag);

    // the modes (q,0,0); ComputeFFT keeps the unshifted index layout
    Gpu::DeviceVector<int> q_dev(m_n_modes);
    Gpu::DeviceVector<Real> real_dev(m_n_modes, 0.), imag_dev(m_n_modes, 0.);
    Gpu::copy(Gpu::hostToDevice, m_q_indices.begin(), m_q_indices.end(), q_dev.begin());
    int const* q_ptr = q_dev.dataPtr();
    Real* real_ptr = real_dev.dataPtr();
    Real* imag_ptr = imag_dev.dataPtr();

    for (MFIter mfi(dft_real); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        if (!bx.contains(IntVect(AMREX_D_DECL(bx.smallEnd(0), 0, 0)))) { continue; }
        const int xlo = bx.smallEnd(0);
        const int xhi = bx.bigEnd(0);
        auto const& re = dft_real.const_array(mfi);
        auto const& im = dft_imag.const_array(mfi);
        ParallelFor(m_n_modes, [=] AMREX_GPU_DEVICE (int n) noexcept
        {
            const int q = q_ptr[n];
            if (q >= xlo && q <= xhi) {
                real_ptr[n] = re(q,0,0);
                imag_ptr[n] = im(q,0,0);
            }
        });
    }

    Gpu::streamSynchronize();
    Gpu::copy(Gpu::deviceToHost, real_dev.begin(), real_dev.end(), real.begin());
    Gpu::copy(Gpu::deviceToHost, imag_dev.begin(), imag_dev.end(), imag.begin());
    ParallelDescriptor::ReduceRealSum(real.data(), m_n_modes);
    ParallelDescriptor::ReduceRealSum(imag.data(), m_n_modes);
}

void
IntermediateScatteringFunction::Accumulate (Vector<Real> const& current_real,
                                            Vector<Real> const& current_imag,
                                            int step, Real time, bool starts_window)
{
    if (starts_window) {
        ActiveWindow w;
        w.start_step = step;
        w.start_time = time;
        w.anchor_real = current_real;
        w.anchor_imag = current_imag;
        w.sample_real.assign(m_n_lags*m_n_modes, 0.);
        w.sample_imag.assign(m_n_lags*m_n_modes, 0.);
        w.sample_lag_time.assign(m_n_lags, 0.);
        w.sample_seen.assign(m_n_lags, 0);
        m_windows.push_back(std::move(w));
    }

    const Real fac = m_product_scaling/static_cast<Real>(m_npencils);
    for (auto& w : m_windows) {
        const int delta = step - w.start_step;
        if (delta < 0 || delta > m_t_max || delta % m_dt_meas != 0) { continue; }
        const int lag = delta/m_dt_meas;

        // current * conj(anchor), averaged over pencils
        for (int m = 0; m < m_n_modes; ++m) {
            Real pr = 0., pi = 0.;
            for (int p = 0; p < m_npencils; ++p) {
                const int idx = p*m_n_modes + m;
                const Real ar = w.anchor_real[idx], ai = w.anchor_imag[idx];
                const Real cr = current_real[idx], ci = current_imag[idx];
                pr += ar*cr + ai*ci;
                pi += ar*ci - ai*cr;
            }
            w.sample_real[lag*m_n_modes + m] = fac*pr;
            w.sample_imag[lag*m_n_modes + m] = fac*pi;
        }
        w.sample_lag_time[lag] = time - w.start_time;
        w.sample_seen[lag] = 1;

        if (delta == m_t_max) {
            bool complete = true;
            for (int seen : w.sample_seen) { complete = complete && (seen != 0); }
            if (complete) {
                for (int l = 0; l < m_n_lags; ++l) {
                    for (int m = 0; m < m_n_modes; ++m) {
                        m_accum_real[l*m_n_modes + m] += w.sample_real[l*m_n_modes + m];
                        m_accum_imag[l*m_n_modes + m] += w.sample_imag[l*m_n_modes + m];
                    }
                    m_accum_lag_time[l] += w.sample_lag_time[l];
                    m_accum_count[l] += 1;
                }
            }
        }
    }

    while (!m_windows.empty() && step - m_windows.front().start_step >= m_t_max) {
        m_windows.pop_front();
    }
}

void
IntermediateScatteringFunction::Sample (int step, Real time, MultiFab const& phi)
{
    if (!m_initialized || step < m_relax_time || step == m_last_step) { return; }
    m_last_step = step;

    const bool starts_window = (step - m_relax_time) % m_window_stride == 0;
    bool due = starts_window;
    for (auto const& w : m_windows) {
        const int delta = step - w.start_step;
        if (delta >= 0 && delta <= m_t_max && delta % m_dt_meas == 0) { due = true; }
    }
    if (!due) { return; }

    Vector<Real> current_real, current_imag;
    ExtractModes(phi, current_real, current_imag);
    Accumulate(current_real, current_imag, step, time, starts_window);
}

void
IntermediateScatteringFunction::Write () const
{
    // m_accum_count[0] counts all complete windows; skip if nothing is new
    if (!m_initialized || m_accum_count[0] == m_written_count) { return; }
    m_written_count = m_accum_count[0];
    if (!ParallelDescriptor::IOProcessor()) { return; }

    for (int m = 0; m < m_n_modes; ++m) {
        const int q = m_q_indices[m];
        const Real kx = 2.0*Math::pi<Real>()*q/m_lx;
        const std::string fname = m_plot_file + (m_do_1D ? "_1D" : "")
                                + "_q" + std::to_string(q) + ".dat";
        std::ofstream os(fname);
        if (!os.good()) { FileOpenFailed(fname); }
        os << "# intermediate scattering function F(k,tau) = <n_hat(t+tau) n_hat(t)^*>/N,"
           << " F(k,0) = S(k); lag_time is the mean over windows\n";
        os << "# lag_index lag_time q_index kx F_real F_imag F_mag n_windows\n";
        os << std::setprecision(17);
        for (int lag = 0; lag < m_n_lags; ++lag) {
            const int count = m_accum_count[lag];
            if (count <= 0) { continue; }
            const int idx = lag*m_n_modes + m;
            const Real fr = m_accum_real[idx]/count;
            const Real fi = m_accum_imag[idx]/count;
            os << lag << " " << m_accum_lag_time[lag]/count << " " << q << " " << kx << " "
               << fr << " " << fi << " " << std::sqrt(fr*fr + fi*fi) << " " << count << "\n";
        }
    }
    Print() << "Wrote intermediate scattering function (" << m_accum_count[0]
            << " windows) to " << m_plot_file << (m_do_1D ? "_1D" : "") << "_q*.dat\n";
}
