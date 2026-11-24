function SparseStaticResid!(T::Vector{<: Real}, residual::AbstractVector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real})
    @assert length(T) >= 1
    @assert length(residual) == 8
    @assert length(y) == 8
    @assert length(x) == 2
    @assert length(params) == 8
@inbounds begin
    residual[1] = (y[3]) - (y[2]);
    residual[2] = (y[2]) - (y[5]+y[1]-y[1]*params[4]+x[2]);
residual[3] = y[1]-y[5]*params[3];
    residual[4] = (y[4]) - (1/T[1]-params[4]*params[2]/T[1]);
    residual[5] = (log(y[5])) - (log(y[5])*params[1]+x[1]);
    residual[6] = (y[6]) - (log(y[2])-log((y[2])));
    residual[7] = (y[7]) - (log(y[1])-log((y[1])));
    residual[8] = (y[8]) - (log(y[5])-log((y[5])));
end
    return nothing
end

