# SPDX-License-Identifier: MPL-2.0
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>



#include("../src/NMRDataSetup.jl")
import NMRDataSetup

using LinearAlgebra
using FFTW
using Test


# verify the implementation of getDFTfreqrange().
# Uses the fact that the discrete Fourier transform and discrete-time Fourier transform is the same at particular frequencies.
function verifygetDFTfreqrangesingle(s_t::Vector{Complex{T}}, fs::T;
    allowed_descrepancy = 1e-12) where T

    t = NMRDataSetup.gettimerange(length(s_t), fs)
    DTFT_s = vv->NMRDataSetup.computeDTFT(s_t, vv, t)

    # DTFT vs. DFT. sanity check.
    U_DFT = NMRDataSetup.getDFTfreqrange(length(s_t), fs)
    DFT_s = fft(s_t)
    S_U = DTFT_s.(U_DFT)

    discrepancy = DFT_s - S_U
    relative_discrepancy = norm(discrepancy)/norm(DFT_s)
    return relative_discrepancy < allowed_descrepancy
    #return norm(discrepancy) < allowed_descrepancy
end



@testset "verifygetDFTfreqrange" begin

    ### test parameters.
    fs = 24.12 # arbitrary number.

    allowed_descrepancy = 1e-10
    N = 1000 # length of sequence for testing for the first case.
    num_trials = 100
    ### end test parameters

    ### test.

    for m = 1:num_trials
        @test verifygetDFTfreqrangesingle(randn(Complex{Float64}, N), fs; allowed_descrepancy = allowed_descrepancy)
    end

    N += 1 # test the odd length case.
    for m = 1:num_trials
        @test verifygetDFTfreqrangesingle(randn(Complex{Float64}, N), fs; allowed_descrepancy = allowed_descrepancy)
    end
    ### end test.
end
