function SparseStaticG1!(T::Vector{<: Real}, g1_v::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real})
    @assert length(T) >= 2
    @assert length(g1_v) == 16
    @assert length(y) == 8
    @assert length(x) == 2
    @assert length(params) == 8
@inbounds begin
g1_v[1]=(-(1-params[4]));
g1_v[2]=1;
g1_v[3]=(-(1/y[1]-1/(y[1])));
g1_v[4]=(-1);
g1_v[5]=1;
g1_v[6]=(-((-T[2])/(T[1]*T[1])-(-(params[4]*params[2]*T[2]))/(T[1]*T[1])));
g1_v[7]=(-(1/y[2]-1/(y[2])));
g1_v[8]=1;
g1_v[9]=1;
g1_v[10]=(-1);
g1_v[11]=(-params[3]);
g1_v[12]=1/y[5]-params[1]*1/y[5];
g1_v[13]=(-(1/y[5]-1/(y[5])));
g1_v[14]=1;
g1_v[15]=1;
g1_v[16]=1;
end
    return nothing
end

