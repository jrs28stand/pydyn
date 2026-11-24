function SparseDynamicG2!(T::Vector{<: Real}, g2_v::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    @assert length(T) >= 6
    @assert length(g2_v) == 7
    @assert length(y) == 24
    @assert length(x) == 2
    @assert length(params) == 8
@inbounds begin
g2_v[1]=(-((T[1]*T[1]*(-(get_power_deriv(y[10],params[6],2)))-T[4]*(T[1]*T[3]+T[1]*T[3]))/(T[1]*T[1]*T[1]*T[1])));
g2_v[2]=(T[2]*T[2]*(-(params[4]*params[2]*get_power_deriv(y[18],params[6],2)))-T[6]*(T[2]*T[5]+T[2]*T[5]))/(T[2]*T[2]*T[2]*T[2]);
g2_v[3]=(-(params[1]*(-1)/(y[5]*y[5])));
g2_v[4]=(-1)/(y[13]*y[13]);
g2_v[5]=(-((-1)/(y[10]*y[10])));
g2_v[6]=(-((-1)/(y[9]*y[9])));
g2_v[7]=(-((-1)/(y[13]*y[13])));
end
    return nothing
end

