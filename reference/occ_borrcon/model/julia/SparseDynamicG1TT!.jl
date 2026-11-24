function SparseDynamicG1TT!(T::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    SparseDynamicResidTT!(T, y, x, params, steady_state)
@inbounds begin
T[3] = get_power_deriv(y[10],params[6],1)
T[4] = (-T[3])
T[5] = get_power_deriv(y[18],params[6],1)
T[6] = (-(params[4]*params[2]*T[5]))
end
    return nothing
end

