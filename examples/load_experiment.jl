# SPDX-License-Identifier: MPL-2.0
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>



import Random
Random.seed!(25)

using LinearAlgebra
using FFTW
using Printf

import PythonPlot as PLT # Pkg.add("PythonPlot") if you don't have this installed.
import NMRDataSetup as DSU


fig_num = 1
PLT.close("all")

# we advise using precision higher than Float32.
#T = Float32
T = Float64

# select the experiment you want to load.
experiment_full_path = joinpath(pwd(), "data", "DMEM")

# # Provide configurations.
# We need configs for fitting singlets and for signal conditioning.

# 0 ppm singlet component fitting.
freq_ref_config = DSU.FitSingletConfig{T}(
    
    frequency_center_ppm = zero(T), # the singlet is at 0 ppm.
    frequency_radius_ppm = T(0.3),

    # smaller λ means sharper line shape at the resonance frequency, less heavy tails.
    λ_lb = T(0.2),
    λ_ub = T(7.0),
)

# solvent singlet component fitting.
# We recommend pasing in `nothing` to disable fitting the solvent unless you know the solvent is the only resonance component within the specified ppm window.
solvent_config = DSU.FitSingletConfig{T}(

    frequency_center_ppm = T(4.9), # in this example, we assume the solvent is between 4.75 - 0.15 to 4.75 + 0.15 ppm.
    frequency_radius_ppm = T(0.2),
    
    # smaller λ means sharper line shape at the resonance frequency, less heavy tails.
    λ_lb = T(1e-3),
    λ_ub = T(20.0),
)
#solvent_config = nothing # uncomment this if you don't want to fit the solvent.

# For the data rescaling, frequency wrap-around so that we have a monotonic increasing frequency range that corresponds to -offset_ppm to SW-offset_ppm, and truncation of dead time from the FID data.
config = DSU.SetupConfig{T}(
    rescaled_max = T(10.0),
    max_CAR_ν_0ppm_difference_ppm = T(0.4), # used to assess whether the experiment is missing (or has a very low intensity) 0 ppm peak.
    offset_ppm = T(0.5),
    FID_name = "fid",
    settings_file_name = "acqu",
)

# # Load from file and set up the data.
output = DSU.setupBruker1Dspectrum(
    T,
    experiment_full_path,
    config;
    freq_ref_config = freq_ref_config,
    solvent_config = solvent_config,
)
data, spectrum, singlet_0ppm, singlet_solvent = DSU.unpackcontainer(output)

# This does not have dead time removal nor magnitude scaling.
unprocessed_fid = DSU.get_unprocessed_fid(data)

# This is a scaled and truncated version of `unprocessed_fid`. Truncation so that dead-time should be removed. Manually inspect and process further if dead-time is not removed.
s, offset_index, s_scale_factor = DSU.get_processed_fid(data)

# Sanity-check.
@assert norm(unprocessed_fid[offset_index:end] .* s_scale_factor -s) < eps(T)*100

# # Unpack.
# frequency in Hz.

# frequencies that correspond to the DFT of s, i.e., `fft(s)`.
U_DFT = DSU.get_DFT_freqs(spectrum)

# wrapped-around frequency, wrapped-around ppm, wrapped-around (wrt frequency) and scaled (wrt magnitude) DFT spectrum.
U_y, P_y, y = DSU.get_wraparound_spectrum(spectrum)

# set up conversion
freq_params = DSU.FrequencyConversionParameters(output)
hz2ppmfunc = uu->DSU.hz2ppm(uu, freq_params)
ppm2hzfunc = pp->DSU.ppm2hz(pp, freq_params)

# sanity-check
@assert norm(U_y - ppm2hzfunc.(P_y))/norm(U_y) < eps(T)*100
@assert norm(P_y - hz2ppmfunc.(U_y))/norm(P_y) < eps(T)*100

perm_inds = DSU.get_wraparound_permutation(spectrum)
y_scaling_factor = DSU.get_scaling_factor(spectrum)
@assert norm((fft(s) .* y_scaling_factor)[perm_inds] - y)/norm(fft(s)) < eps(T)*100

# The FID parameters of both of the singlets fitted.
p_0ppm = [singlet_0ppm.α; singlet_0ppm.β; singlet_0ppm.λ; singlet_0ppm.Ω]

# for plotting and displaying.
RE_0ppm = DSU.extractfreqreferror(output)
λ0 = singlet_0ppm.λ
P_0ppm_lb = freq_ref_config.frequency_center_ppm - freq_ref_config.frequency_radius_ppm
P_0ppm_ub = freq_ref_config.frequency_center_ppm + freq_ref_config.frequency_radius_ppm


# # Visualize the data spectrum.

# real.
PLT.figure(fig_num)
fig_num += 1

PLT.plot(P_y, real.(y))

PLT.title("Wrapped-around spectrum, real part")
PLT.gca().invert_xaxis()
PLT.xlabel("ppm")
PLT.legend()

# imaginary.
PLT.figure(fig_num)
fig_num += 1

PLT.plot(P_y, imag.(y))

PLT.title("Wrapped-around spectrum, imaginary part")
PLT.gca().invert_xaxis()
PLT.xlabel("ppm")
PLT.legend()


# ## Visualize the singlet fits.

# query the complex Lorentzian singlet model.
inds = findall(xx->(P_0ppm_lb < xx < P_0ppm_ub), P_y)
y_display = y[inds]
P_display = P_y[inds]
U_rad_display = U_y[inds] .* T(2*π)

q = uu->DSU.evalCL(uu, p_0ppm...)
q_display = q.(U_rad_display)

# ### frequency reference at 0 ppm.

# real.
PLT.figure(fig_num)
fig_num += 1

PLT.plot(P_display, real.(y_display), label = "y", linewidth = 2)
PLT.plot(P_display, real.(q_display), "--", label = "q, 0 ppm")

PLT.title("0 ppm: real part")
PLT.gca().invert_xaxis()
PLT.xlabel("ppm")
PLT.legend()
println("0 ppm relative error = $(@sprintf("%.5f", RE_0ppm))")

# imageinary.
PLT.figure(fig_num)
fig_num += 1

PLT.plot(P_display, imag.(y_display), label = "y", linewidth = 2)
PLT.plot(P_display, imag.(q_display), "--", label = "q, 0 ppm")

PLT.title("0 ppm: imaginary part")
PLT.gca().invert_xaxis()
PLT.xlabel("ppm")
PLT.legend()
println("0 ppm λ = $(@sprintf("%.3f", λ0))")


# check if we fitted the solvent.
if !isnothing(solvent_config)

    # ## Plot solvent.

    p_solvent = [singlet_solvent.α; singlet_solvent.β; singlet_solvent.λ; singlet_solvent.Ω]
    P_solvent_lb = solvent_config.frequency_center_ppm - solvent_config.frequency_radius_ppm
    P_solvent_ub = solvent_config.frequency_center_ppm + solvent_config.frequency_radius_ppm
    λs = singlet_solvent.λ
    RE_solvent = DSU.extractsolventferror(output)


    # query.
    inds = findall(xx->(P_solvent_lb < xx < P_solvent_ub), P_y)
    y_display = y[inds]
    P_display = P_y[inds]
    U_rad_display = U_y[inds] .* T(2*π)

    q = uu->DSU.evalCL(uu, p_solvent...)
    q_display = q.(U_rad_display)

    # ### solvent.

    # real.
    PLT.figure(fig_num)
    fig_num += 1

    PLT.plot(P_display, real.(y_display), label = "y", linewidth = 2)
    PLT.plot(P_display, real.(q_display), "--", label = "q, solvent")

    PLT.title("solvent: real part")
    PLT.gca().invert_xaxis()
    PLT.xlabel("ppm")
    PLT.legend()
    println("Solvent relative error = $(@sprintf("%.5f", RE_solvent))")

    # imagianry.
    PLT.figure(fig_num)
    fig_num += 1

    PLT.plot(P_display, imag.(y_display), label = "y", linewidth = 2)
    PLT.plot(P_display, imag.(q_display), "--", label = "q, solvent")

    PLT.title("solvent: imaginary part")
    PLT.gca().invert_xaxis()
    PLT.xlabel("ppm")
    PLT.legend()
    println("0 ppm λ = $(@sprintf("%.3f", λs))")
end

nothing

