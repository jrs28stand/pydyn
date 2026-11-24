function SparseDynamicResid!(T::Vector{<: Real}, residual::AbstractVector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    @assert length(T) >= 2
    @assert length(residual) == 8
    @assert length(y) == 24
    @assert length(x) == 2
    @assert length(params) == 8
@inbounds begin
    residual[1] = (y[11]) - (y[18]);
    residual[2] = (y[10]) - (y[13]+y[9]-params[4]*y[1]+x[2]);
residual[3] = y[12]*params[8]+(y[9]-y[13]*params[3])*(1-params[8]);
    residual[4] = (y[12]) - (1/T[1]-params[4]*params[2]/T[2]);
    residual[5] = (log(y[13])) - (params[1]*log(y[5])+x[1]);
    residual[6] = (y[14]) - (log(y[10])-log((steady_state[2])));
    residual[7] = (y[15]) - (log(y[9])-log((steady_state[1])));
    residual[8] = (y[16]) - (log(y[13])-log((steady_state[5])));
end
    return nothing
end

