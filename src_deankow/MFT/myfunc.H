#ifndef MYFUNC_H_
#define MYFUNC_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_Parser.H>

#include "StructFact.H"

void advance_rho (amrex::MultiFab& rho_old,
                  amrex::MultiFab& rho_new,
                  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& flux,
                  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& stochFlux,
                  amrex::Real dt, amrex::Real Lambda, amrex::Real dorand,
                  amrex::ParserExecutor<1> const& Dfn,
                  amrex::ParserExecutor<1> const& sigfn,
                  amrex::Geometry const& geom,
                  amrex::BCRec const& bc,
                  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& rho_bc_lo,
                  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& rho_bc_hi,
                  amrex::GpuArray<int,AMREX_SPACEDIM> const& ens_dir);

// forward Euler stability limit cfl*2/lambda_max for the diffusion operator
amrex::Real est_time_step (amrex::MultiFab const& rho, amrex::Real cfl,
                           amrex::ParserExecutor<1> const& Dfn,
                           amrex::Geometry const& geom,
                           amrex::BCRec const& bc,
                           amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& rho_bc_lo,
                           amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& rho_bc_hi,
                           amrex::GpuArray<int,AMREX_SPACEDIM> const& ens_dir);

// write plt_SF and the shell-averaged sf_kshell_<step>.txt
void write_struct_fact (StructFact& structFact, amrex::Geometry const& geom,
                        int step, amrex::Real time, int nsamples);

#endif
