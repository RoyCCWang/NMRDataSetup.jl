# SPDX-License-Identifier: MPL-2.0
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>



@kwdef struct SetupConfig{T}
    rescaled_max::T = convert(T, 10.0)
    max_CAR_ν_0ppm_difference_ppm::T = convert(T, 0.4) # used to assess whether the experiment is missing (or has a very low intensity) 0 ppm peak
    offset_ppm::T = convert(T, 0.5)
    FID_name::String = "fid"
    settings_file_name::String = "acqu"
end

struct SingletFIDParameters{T <: AbstractFloat}
    α::T
    β::T
    λ::T
    Ω::T # in radians.
    relative_error::T

    # default to an invalid singlet.
    function SingletFIDParameters(::Type{T}) where T <: AbstractFloat
        return new{T}(T(NaN), T(NaN), T(NaN), T(NaN), T(NaN))
    end

    function SingletFIDParameters(p::Union{Vector{T}, Memory{T}}, RE::T) where T <: AbstractFloat
        length(p) == 4 || error("There must only be 4 entries in the input.")

        # make sure α is positive.
        tmp = p[1]*cis(p[2])
        α, β = abs(tmp), angle(tmp)

        return new{T}(α, β, p[3], p[4], RE)
    end
end

function get_T2_reciprocal(A::SingletFIDParameters)
    return A.λ
end

function get_resonance_frequency_hz(A::SingletFIDParameters{T}) where T <: AbstractFloat
    return A.Ω / T(2*π)
end

function get_resonance_intensity(A::SingletFIDParameters)
    return A.α
end

function get_resonance_phase(A::SingletFIDParameters)
    return A.β
end

function get_fit_rel_error(A::SingletFIDParameters)
    return A.relative_error
end

function isvalidsinglet(s::SingletFIDParameters{T}) where T <: AbstractFloat

    if isfinite(s.α) && isfinite(s.β) && isfinite(s.λ) && isfinite(s.Ω)
        return true
    end

    return false
end


################# easier access functions.

struct OutputContainer1D{T <: AbstractFloat, ST <: NMRSettings}
    
    data::Data1D{T, ST}
    spectrum::SpectrumData1D{T}
    
    # singlets
    freq_ref::SingletFIDParameters{T}
    solvent::SingletFIDParameters{T}
end

function unpackcontainer(A::OutputContainer1D{T,ST}) where {T,ST}
    return A.data, A.spectrum, A.freq_ref, A.solvent
end

# for extracting a singlet's resonance frequency, in Hz.
function extractresonancefreq(singlet_0ppm::SingletFIDParameters{T}) where T <: AbstractFloat
    ν_0ppm = singlet_0ppm.Ω / twopi(T)
    return ν_0ppm
end

function extractfreqrefHz(A::OutputContainer1D{T,ST}) where {T <: AbstractFloat, ST}
    return extractresonancefreq(A.freq_ref)
end

function extractsolventfHz(A::OutputContainer1D{T,ST}) where {T <: AbstractFloat, ST}
    return extractresonancefreq(A.solvent)
end

function extractfreqreferror(A::OutputContainer1D{T,ST}) where {T <: AbstractFloat, ST}
    return A.freq_ref.relative_error
end

function extractsolventferror(A::OutputContainer1D{T,ST}) where {T <: AbstractFloat, ST}
    return A.solvent.relative_error
end

# get parameters for frequency to ppm conversion.
struct FrequencyConversionParameters{T<:AbstractFloat}
    fs::T
    SW::T
    ν_0ppm::T
end

function FrequencyConversionParameters(A::OutputContainer1D{T,ST}) where {T <: AbstractFloat, ST}
    return FrequencyConversionParameters(
        A.data.settings.fs,    
        A.data.settings.SW,
        extractfreqrefHz(A),
    )
end

function hz2ppm(u_Hz::T, A::FrequencyConversionParameters{T}) where T
    return (u_Hz - A.ν_0ppm)*A.SW/A.fs
end

function ppm2hz(p_ppm::T, A::FrequencyConversionParameters{T}) where T
    return A.ν_0ppm + p_ppm*A.fs/A.SW
end

############ for creating OutputContainer1D.

function setupBruker1Dspectrum(
    ::Type{T},
    experiment_full_path::String,
    config::SetupConfig{T};
    freq_ref_config::FitSingletConfig{T} = FitSingletConfig{T}(
        frequency_center_ppm = zero(T),
    ),
    solvent_config::Union{FitSingletConfig{T}, Nothing} = nothing,
    ) where T <: AbstractFloat

    offset_ppm = config.offset_ppm
    max_CAR_ν_0ppm_difference_ppm = config.max_CAR_ν_0ppm_difference_ppm

    # the packaged up version.
    data = loadBruker(
        T,
        experiment_full_path;
        rescaled_max = config.rescaled_max,
        FID_name = config.FID_name,
        settings_file_name = config.settings_file_name,
    )

    s, settings = data.s, data.settings
    fs, O1, SW = settings.fs, settings.O1, settings.SW

    # # fit the 0 ppm and solvent singlets.

    # using O1 as ν_0ppm.
    spectrum_O1 = SpectrumData1D(s, settings; offset_ppm = offset_ppm)

    # estimate ν_0ppm_new
    p_O1, rel_error_O1 = fitsinglet(spectrum_O1, freq_ref_config)
    if isempty(p_O1)

        if isnothing(solvent_config)
            println("There was an error fitting the singlet at 0 ppm. Terminate.")
        else
            println("There was an error fitting the singlet at 0 ppm. Terminate without fitting the solvent.")
        end

        return data, spectrum_O1, SingletFIDParameters(T), SingletFIDParameters(T)
    end 
    ν_0ppm = p_O1[end]/twopi(T)
    #@show ν_0ppm, fs-O1

    # sanity check: is this a sensible result for ν_0ppm?
    CAR = fs-O1
    ppm2hzfunc_O1 = pp->(CAR + pp*fs/SW)
    max_CAR_ν_0ppm_difference_Hz = ppm2hzfunc_O1(max_CAR_ν_0ppm_difference_ppm)
    if abs(ν_0ppm - CAR) > max_CAR_ν_0ppm_difference_Hz
        # we failed to find a 0 ppm peak.
        println("The fit result for the singlet at 0 ppm deviated too much from fs-O1. There might not be a singlet in the experiment, or try increasing `max_CAR_ν_0ppm_difference_Hz`, which has a current value of $max_CAR_ν_0ppm_difference_Hz.")
        return data, spectrum_O1, SingletFIDParameters(T), SingletFIDParameters(T)
    end 

    # use ν_0ppm to redo compute spectrum data container.
    spectrum = SpectrumData1D(
        s,
        ν_0ppm, SW, fs;
        offset_ppm = offset_ppm,
    )

    #hz2ppmfunc = uu->(uu - ν_0ppm)*SW/fs
    ppm2hzfunc = pp->(ν_0ppm + pp*fs/SW)

    # since we accepted ν_0ppm, we readjust p_O1 with it to get p_0ppm.
    p_0ppm = copy(p_O1)
    p_0ppm[end] = ppm2hzfunc(zero(T)) * twopi(T) # replace with 0 ppm.
    singlet_0ppm = SingletFIDParameters(p_0ppm, rel_error_O1)

    # fit solvent.
    singlet_solvent = SingletFIDParameters(T)

    if !isnothing(solvent_config)
        p_solvent, rel_error_solvent = fitsinglet(spectrum, solvent_config)
        if isempty(p_solvent) #|| rel_error_solvent > rel_error_threshold        
            
            println("There was an error fitting the singlet at 0 ppm. Terminate.")
            return data, spectrum, SingletFIDParameters(p_0ppm, rel_error_O1), SingletFIDParameters(T)
        end
        #ν_solvent = p_solvent[end]/twopi(T)
        #@show ν_solvent, hz2ppmfunc(ν_solvent)
        singlet_solvent = SingletFIDParameters(p_solvent, rel_error_solvent)
    end

    return OutputContainer1D(data, spectrum, singlet_0ppm, singlet_solvent)
end
