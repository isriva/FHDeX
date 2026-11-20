#ifndef _TurbForcingComp_H_
#define _TurbForcingComp_H_

#include "common_functions.H"

class TurbForcingComp {

public:

    TurbForcingComp();

    void define(BoxArray ba_in, DistributionMapping dmap_in,
                const Real& a_in, const Real& b_in, const Real& c_in, const Real& d_in, const Real& alpha_in,
                const int& kf2_in);

    void Initialize(const Geometry& geom_in);

    void CalcTurbForcingComp(std::array< MultiFab, AMREX_SPACEDIM >& vel_f,
                        const Real& dt,
                        const int& update);

    std::tuple<amrex::Real, amrex::Real> getU(const int& i);

    void setU(const int& i, Real fs, Real fc);

    // number of forced wavevectors (one per +/-k pair)
    int nModes() const { return nmodes; }

    // number of OU coefficients per forcing type: 3 components x (cos, sin) per mode
    int nForcing() const { return 6*nmodes; }

private:

    // forced wavevectors (in units of 2*pi/L), filled in define() according to turb_kf2
    // each entry represents the pair +/-k through its cos(k.x) and sin(k.x) terms
    Gpu::HostVector<int> KX;
    Gpu::HostVector<int> KY;
    Gpu::HostVector<int> KZ;

    int nmodes = 0;

    Gpu::HostVector<Real> ForcingS;
    Gpu::HostVector<Real> ForcingC;

    Gpu::HostVector<Real> ForcingSold;
    Gpu::HostVector<Real> ForcingCold;

    std::array< MultiFab, AMREX_SPACEDIM> sines;
    std::array< MultiFab, AMREX_SPACEDIM> cosines;

    Real forcing_a; // 1st coeff for solenoidal forcing
    Real forcing_b; // 2nd coeff for solenoidal forcing
    Real forcing_c; // 1st coeff for dilational forcing
    Real forcing_d; // 2nd coeff for dilational forcing
    Real alpha;     // fraction of solenoidal forcing

};

#endif
