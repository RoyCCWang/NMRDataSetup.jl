# SPDX-License-Identifier: MPL-2.0
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>



module NMRDataSetup

using LinearAlgebra
using FFTW
#using Statistics

import ForwardDiff
import Optim

# constant values.
function twopi(::Type{T}) where T <: AbstractFloat
    return T(2)*T(π)
end


include("DSP.jl")

include("bruker_IO.jl")

include("fit_singlet.jl")

include("utils.jl")

include("frontend.jl")

export 

    # General data containers.
    Data1D,
    OutputContainer1D,

    # data spectrum for fitting.
    SpectrumData1D,

    # fit singlet.
    FitSingletConfig,
    fitsinglet,

    # Bruker IO.
    
    Bruker1D1HSettings,
    loadBruker,
    setupBruker1Dspectrum,

    # utilities
    unpackcontainer,
    extractfreqrefHz,
    extractsolventfHz,
    getinversemap
end

