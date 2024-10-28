# SPDX-License-Identifier: MPL-2.0
# Copyright © 2024 Roy Chih Chung Wang <roy.c.c.wang@proton.me>



function evallinearitp(x0, y0, x1, y1, x::T) where T
    numerator = y0*(x1-x) + y1*(x-x0)
    return numerator/(x1-x0)
end


function scaletimeseries(s_data::Memory{Complex{T}}; new_max::T = convert(T, 10)) where T
    
    scale_factor = new_max/maximum( abs(s_data[n]) for n in eachindex(s_data) )
    
    s = Memory{Complex{T}}(undef, length(s_data))
    for i in eachindex(s, s_data)
        s[i] = convert(Complex{T}, s_data[i]) * scale_factor
    end

    return s, scale_factor
end
