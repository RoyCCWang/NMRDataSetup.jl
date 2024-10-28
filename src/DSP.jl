# SPDX-License-Identifier: MPL-2.0
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>



"""
    evalcomplexLorentzian(u, α, β::T, λ, Ω) where T <: Real

Evaluates the complex Lorentzian function. Returns `α*exp(im*β)/(λ + im*(2*π*u - Ω))`.
"""
function evalcomplexLorentzian(u, α, β::T, λ, Ω) where T <: Real
    #return α*exp(im*β)/(λ + im*(convert(T, 2*π)*u - Ω))
    return α*cis(β)/(λ + im*(convert(T, 2*π)*u - Ω))
end

"""
    gettimerange(N::Int, fs::T) where T

Returns the time stamps for a sequence, starting at time 0.
Returns `zero(T):Ts:(N-1)*Ts`, where `Ts = 1/fs`.
"""
function gettimerange(N::Int, fs::T) where T
    Ts::T = convert(T, 1/fs)

    #return zero(T):Ts:convert(T, (N-1)*Ts)
    return LinRange(zero(T), convert(T, (N-1)*Ts), N)
end

"""
    getwraparoundDFTfreqs(N::Int, fs::T, ν_begin::T) where T

Get the frequency indices for a sequence of length `N` that is sampled at `fs` Hz, with a starting frequency in `ν_begin` Hz.

...
# Inputs:
- `N::Int`: length of the time-domain sequence that was discrete-time Fourier transformed.
- `fs::T`: the sampling frequency in Hz.
- `ν_begin::T`: the wrap-around frequency in Hz.

# Outputs, in the order returned.

- `U_DFT::LinRange{T,Int}`: frequency indices (in Hz) for `fft(x)`, where `length(x)` is `N` and values in `x` were sampled at `fs`.
- `U_y::Vector{T}`: frequency indices (in Hz) for the shited version of `fft(x)` that has the first frequency larger than `ν_begin` as the first entry. When fitting 1D 1H NMR data in the frequency domain, `U_y` is used instead of `U_DFT` with `ν_begin` being the 0 ppm frequency in Hz. One can then convert `U_y`, which is in Hz, to ppm, by using a conversion function appropriate for this experiment. See the `hz2ppmfunc` output from `loadspectrum()`.
- `U_inds::Vector{Int}`: `ff(x)[U_inds]` would give the correct spectrum values that correspond to the frequency indices in `U_y`.

Note that `U_y` doesn't contain the same numerical values as `U_DFT[U_inds]` because the latter is not neccessarily a monotonically increasing sequence, but `U_y` is. They both describe the same discrete-time frequencies though.
Note that U_inds follows the Julia 1-indexing scheme. One should manually subtract all entries of U_inds by 1 if calling from a 0-indexing array language such as Python.
, , , , ,

# Examples
See load_experiment.jl in the /examples folder and the documentation website for usage examples.
"""
function getwraparoundDFTfreqs(N::Int, fs::T, ν_begin::T) where T

    U0 = getDFTfreqrange(N, fs)
    out, inds = wrapfreqrange(U0, ν_begin, fs)

    return U0, out, inds
end

# Assumes U0 is sorted in ascending order.
function wrapfreqrange(U0, ν_begin::T, fs::T) where T <: Real

    N = length(U0)
    ind = findfirst(xx->(xx>ν_begin), U0)

    # TODO handle these exceptions with more grace.
    @assert typeof(ind) == Int
    @assert ind <= N

    out = Memory{T}(undef, N)
    #out[1:ind] = U0[1:ind] .+ fs
    #out[ind+1:end] = U0[ind+1:end]

    M = N-ind
    out[1:M] = U0[ind+1:end]
    out[M+1:end] = U0[1:ind] .+ fs

    inds = Memory{Int}(1:N)
    inds[1:M] = Memory{Int}(ind+1:N)
    inds[M+1:end] = Memory{Int}(1:ind)

    #inds = collect(1:N)
    #inds[1:M] = collect(ind+1:N)
    #inds[M+1:end] = collect(1:ind)

    return out, inds
end

"""
    getDFTfreqrange(N::Int, fs::T) where T

Returns the frequency stamps for a DFT sequence, computed by fft().
    Starting at frequency 0 Hz. Returns `LinRange(0, fs-fs/N, N)`.
"""
function getDFTfreqrange(N::Int, fs::T) where T
    a = zero(T)
    b = convert(T, fs-fs/N)

    return LinRange(a, b, N)
end

"""
    computeDTFT(h, u::T, X) where T <: Real

Returns the discrete-time Fourier transform (DTFT) of a sequence h, where X is the corresponding time stamp of h, i.e.:
returns `sum( h[i]*exp(-im*2*π*u*X[i]) for i = 1:length(h) )`
"""
function computeDTFT(h, u::T, X) where T <: Real

    running_sum = zero(Complex{T})
    negative_two_pi_u = convert(T, -2*π)*u
    for i in eachindex(X)
        #running_sum += h[i]*exp(-im*2*π*u*X[i])
        running_sum += h[i]*cis(negative_two_pi_u*X[i])
    end

    return running_sum
end

"""
getinversemap(forward_mapping::Vector{Int})

Obtain the indices for the inverse mapping. Usage:

```
y = randn(6)
inds = [4;6;3;5;1;2]
h = y[inds]

inv_inds = getinversemap(inds)
y_rec = h[inv_inds]
@show norm(y - y_rec) # should be practically zero.
```
"""
function getinversemap(forward_mapping::Vector{Int})

    reverse_mapping = Vector{Int}(undef, length(forward_mapping))
    for i in eachindex(reverse_mapping)
        k = forward_mapping[i]
        reverse_mapping[k] = i
    end

    return reverse_mapping
end