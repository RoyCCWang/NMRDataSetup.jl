# SPDX-License-Identifier: MPL-2.0
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>



############ Lorentzians
# complex Lorentzian (CL), absorption Lorentzian (AL), dispersive Lorentzian (DL)

function evalFID(t::T, α::T, β::T, λ::T, Ω::T) where T <: Real
    return α*cis(Ω*t + β)*exp(-λ*t)
end

function evalCL(u_rad::T, α::T, β::T, λ::T, Ω::T) where T <: Real
    #return α*exp(im*β)/(λ + im*(u_rad - Ω))
    return α*cis(β)/(λ + im*(u_rad - Ω))
end

# based on evalAL and evalDl. For use with automatic differentiation on a, b, c, d.
function evalCLsplit(u_rad::T, a::T2, b::T2, c::T2, d::T2) where {T <: Real, T2 <: Real}
    
    Δud = u_rad-d
    tmp = a/(c^2 + Δud^2)
    sin_b, cos_b = sin(b), cos(b)

    # based on: u = u_rad.
    #DL = tmp*(c*sin_b + d*cos_b - u*cos_b)
    #AL = tmp*(-d*sin_b + u*sin_b + c*cos_b)
    DL = tmp*(c*sin_b - Δud*cos_b)
    AL = tmp*(Δud*sin_b + c*cos_b)

    return AL, DL
end

function evalAL(u, α, β::T, λ, Ω) where T <: Real
    
    a, b, c, d = α, β, λ, Ω
    tmp = a/(c^2 + (u-d)^2)

    return tmp*(-d*sin(b) + u*sin(b) + c*cos(b))
end

function evalDL(u, α, β::T, λ, Ω) where T <: Real
    
    a, b, c, d = α, β, λ, Ω
    tmp = a/(c^2 + (u-d)^2)

    return tmp*(c*sin(b) + d*cos(b) - u*cos(b))
end

# cost function.This is the l-2 cost, squared.
function eval_quadratic_cost(
    p::Union{Vector{T2}, Memory{T2}},
    y::Memory{Complex{T}},
    U_rad,
    ) where {T <: Real, T2 <: Real}

    cost = zero(T2)
    for m in eachindex(U_rad, y)

        qr, qi = evalCLsplit(U_rad[m], p[1], p[2], p[3], p[4])

        cost += (qr-real(y[m]))^2 + (qi-imag(y[m]))^2
    end

    return cost
end

# we have  norm(fft(data.s) .* spectrum.scale_factor), norm( spectrum.y) is the same.
struct SpectrumData1D{T <: AbstractFloat}
    U_DFT::LinRange{T,Int}
    U_y::Memory{T}
    P_y::Memory{T}
    y::Memory{Complex{T}}
    
    # for reference.
    DFT2y_inds::Memory{Int}
    scale_factor::T
    
    function SpectrumData1D(
        s::Memory{Complex{T}},
        settings::Bruker1D1HSettings{T};
        offset_ppm = convert(T, 0.5),
        ) where T <: AbstractFloat
        
        fs, O1, SW = settings.fs, settings.O1, settings.SW
        ν_0ppm = fs - O1
        
        return SpectrumData1D(s, ν_0ppm, SW, fs; offset_ppm = offset_ppm)
    end

    function SpectrumData1D(
        s_t::Memory{Complex{T}},
        ν_0ppm::T,
        SW::T,
        fs::T;
        offset_ppm = convert(T, 0.5),
        ) where T <: AbstractFloat

        hz2ppmfunc = uu->(uu - ν_0ppm)*SW/fs
        ppm2hzfunc = pp->(ν_0ppm + pp*fs/SW)

        offset_Hz = ν_0ppm - (ppm2hzfunc(offset_ppm)-ppm2hzfunc(zero(T)))

        # # get the data spectrum for fitting.
        tmp = Memory{Complex{T}}(fft(s_t)) #./ length(s_t))
        DFT_s_scaled, scale_factor = scaletimeseries(tmp)
        #scale_factor = scale_factor/length(s_t)

        U_DFT, U_y, DFT2y_inds = getwraparoundDFTfreqs(length(s_t), fs, offset_Hz)
        P_y = map(hz2ppmfunc, U_y)

        # reorder to get the spectrum for fitting.
        y = DFT_s_scaled[DFT2y_inds]

        #y2DFT_inds = getinversemap(DFT2y_inds)

        return new{T}(U_DFT, U_y, P_y, y, DFT2y_inds, scale_factor)
    end
end

# such that the ppm is continuous, from offset_ppm onwards.
function get_wraparound_spectrum(A::SpectrumData1D)
    return A.U_y, A.P_y, A.y
end

function get_DFT_freqs(A::SpectrumData1D)
    return A.U_DFT
end

function get_wraparound_permutation(A::SpectrumData1D)
    return A.DFT2y_inds
end

function get_scaling_factor(A::SpectrumData1D)
    return A.scale_factor
end

@kwdef struct FitSingletConfig{T}
    frequency_center_ppm::T
    frequency_radius_ppm::T = convert(T, 0.3)
    window_shrink_factor::T = convert(T, 3)
    min_dynamic_range::T = convert(T, 0.05) # Put NaN if you want to skip this check, and force fit the singlet. if maximum(abs(fit samples))/norm(fit samples) is less than min_dynamic_range, there is probably not a singlet in the region. Terminate without fitting, and report an unsuccessful status.
    λ_lb::T = convert(T, 1e-3)
    λ_ub::T = convert(T, 20.0)
    α_abs_bound::T = convert(T, 1e3) # strictly positive. α_lb := -α_abs_max
    N_phase_starts::Int = 8
    N_T2_starts::Int = 5
    #grad_tol::T = convert(T, 1e-6)
end

function fitsinglet(
    spectrum::SpectrumData1D{T},
    C::FitSingletConfig{T},
    ) where T <: AbstractFloat

    # pase and error check.

    α_abs_bound = C.α_abs_bound
    @assert α_abs_bound > zero(T)
    U_y, P_y, y = spectrum.U_y, spectrum.P_y, spectrum.y

    frequency_radius_ppm = C.frequency_radius_ppm
    window_shrink_factor = C.window_shrink_factor
    min_dynamic_range, λ_lb, λ_ub = C.min_dynamic_range, C.λ_lb, C.λ_ub
    window_lb = C.frequency_center_ppm - frequency_radius_ppm
    window_ub = C.frequency_center_ppm + frequency_radius_ppm

    inds = findall(xx->(window_lb < xx < window_ub), P_y)
    if isempty(inds)
        println("Error: cannot set up cost function. Please specify a larger `frequency_radius_ppm`. The current value is $frequency_radius_ppm")
        return Vector{T}(undef, 0), convert(T, Inf)
    end

    # get the samples near the CAR frequency, which should be near the 0 ppm component.
    P_CAR = P_y[inds]
    y_CAR = y[inds]
    U_CAR = U_y[inds]

    # We're assuming there is only one peak in the frequency interval U_cost.
    max_abs_y_CAR, ind = findmax(abs.(y_CAR))
    Ω0_ppm = P_CAR[ind]
    Ω0_Hz = U_CAR[ind]
    Ω0 = Ω0_Hz * convert(T, 2*π)

    # # Shrink analysis window to minimize baseline effects during fit, and estimate baseline.
    r_lb = convert(T, Ω0_ppm - frequency_radius_ppm/window_shrink_factor)
    r_ub = convert(T, Ω0_ppm + frequency_radius_ppm/window_shrink_factor)
    inds = findall(xx->(r_lb<xx<r_ub), P_y)

    y_cost = y[inds]
    cost_scale_factor = maximum(abs.(y))
    y_cost = map(xx->xx/cost_scale_factor, y_cost) # noramlize so that we can pick grad_tol easier for the optim problem.
    
    P_cost = P_y[inds]
    U_cost = U_y[inds]
    
    # baseline.
    real_itp = uu->evallinearitp(U_cost[begin], real(y_cost[begin]), U_cost[end], real(y_cost[end]), uu)
    baseline_real_Ω0 = real_itp(Ω0_Hz)

    imag_itp = uu->evallinearitp(U_cost[begin], imag(y_cost[begin]), U_cost[end], imag(y_cost[end]), uu)
    baseline_imag_Ω0 = imag_itp(Ω0_Hz)

    baseline_abs_Ω0 = abs(Complex(baseline_real_Ω0, baseline_imag_Ω0))
    baseline_real = real_itp.(U_cost)
    baseline_imag = imag_itp.(U_cost)

    baseline = Memory{Complex{T}}(undef, length(baseline_real))
    for i in eachindex(baseline, baseline_real, baseline_imag)
        baseline[i] = Complex(baseline_real[i], baseline_imag[i])
    end

    # Check if there even a strong enough signal for the singlet in the interval we're looking at.
    dynamic_range = (max_abs_y_CAR - baseline_abs_Ω0)/max_abs_y_CAR
    #@show dynamic_range
    if isfinite(min_dynamic_range)
        if dynamic_range < min_dynamic_range
            println("Singlet signal too weak. Terminate without fitting. Decrease `min_dynamic_range` if you still want to fit.")
            return Vector{T}(undef, 0), convert(T, Inf)
        end
    end

    # verical adjustment to get the samples for fitting.
    y_cost_adj = similar(y_cost)
    y_cost_adj .= y_cost .- baseline
    
    U_rad_cost = similar(U_cost)
    U_rad_cost .= U_cost .* twopi(T)

    # # Initial iterate.
    α0 = maximum(abs.(y_cost_adj))
    p0 = [α0; zero(T); one(T); Ω0]


    # cost function. TODO implement the analytical derivatives.
    f = pp->eval_quadratic_cost(pp, y_cost_adj, U_rad_cost)
    df! = (gg,pp)->ForwardDiff.gradient!(gg, f, pp)
    
    # bounds.
    _, ind = findmin(P_cost)
    Ω_lb = U_rad_cost[ind]
    
    _, ind = findmax(P_cost)
    Ω_ub = U_rad_cost[ind]

    lbs = [ -α_abs_bound; -twopi(T); λ_lb; Ω_lb;]
    ubs = [ α_abs_bound; twopi(T); λ_ub; Ω_ub;]

    # # Optimize.

    # use Optim.jl.
    d2f! = (hh,pp)->ForwardDiff.hessian!(hh, f, pp)
    ret_BFGS = Optim.optimize(f, df!, d2f!, p0, Optim.BFGS())
    p_star_Optim_BFGS = Optim.minimizer(ret_BFGS)
    p_star_Optim_BFGS = clamp.(p_star_Optim_BFGS, lbs, ubs)

    # finish it up in case BFGS terminates with large residual gradient norm.
    ret_CG = Optim.optimize(f, df!, p_star_Optim_BFGS, Optim.ConjugateGradient())
    p_star_Optim_CG = Optim.minimizer(ret_CG)
    p_star_Optim_CG = clamp.(p_star_Optim_CG, lbs, ubs)

    # ## Setup a batch of initial iterates.

    # prepare initial iterates.
    N_phase_starts = 8
    phase_range = LinRange(zero(T), twopi(T), N_phase_starts+1)[begin:end-1]
    N_T2_starts = 5
    T2_range = LinRange(convert(T, 0.2), convert(T, 10.0), N_T2_starts)

    # fit.
    fdf! = (gg,pp)->evalfdf!(gg, pp, f, df!)
    rets_CG, ind_CG = runCGbatch(
        f,
        df!,
        #lbs,
        #ubs,
        p0,
        T2_range,
        phase_range;
        #grad_tol = C.grad_tol,
    )
    ps = collect( Optim.minimizer(rets_CG[n]) for n in eachindex(rets_CG))
    p_star_batch = clamp.(ps[ind_CG], lbs, ubs)

    # compute error, then choose solution.
    Z = norm(y_cost_adj)
    rel_error_Optim_BFGS = sqrt(f(p_star_Optim_BFGS))/Z
    rel_error_Optim_CG = sqrt(f(p_star_Optim_CG))/Z
    rel_error_batch = sqrt(f(p_star_batch))/Z
    # @show rel_error_Optim
    # @show rel_error_batch
    # @show norm(y_cost_adj)

    # compute relative error of the fit before we scale the best p_star.
    rel_error, ind = findmin( [rel_error_Optim_BFGS; rel_error_Optim_CG; rel_error_batch] )
    
    p_star = p_star_batch
    if ind == 1
        p_star = p_star_Optim_BFGS
    elseif ind == 2
        p_star == p_star_Optim_CG
    end

    # scale back to spectrum.y.
    p_star[begin] = p_star[begin]*cost_scale_factor

    return p_star, rel_error
end


# TODO remove this once analytical derivatives are implemented for eval_quadratic_cost().
function evalfdf!(g::Vector{T}, p::Vector{T}, f, df!) where T <: AbstractFloat
    df!(g, p)
    return f(p)
end


function runCGbatch(
    f,
    df!,
    #lbs::Vector{T},
    #ubs::Vector{T},
    p0::Vector{T},
    λ0s,
    β0s;
    #grad_tol::T = convert(T, 1e-4),
    ) where T <: AbstractFloat

    N = length(λ0s) * length(β0s)
    rets = Vector{Any}(undef, N)
    
    m = 0
    for ind_T2 in eachindex(λ0s)
        for ind_phase in eachindex(β0s)
         
            p = copy(p0)
            p[2] = β0s[ind_phase]
            p[3] = λ0s[ind_T2]

            m += 1
            rets[m] = Optim.optimize( f, df!, p, Optim.ConjugateGradient())
        end
    end

    _, ind = findmin( Optim.minimizer(rets[m]) for m in eachindex(rets) )

    return rets, ind
end
