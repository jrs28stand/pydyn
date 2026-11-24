function SparseDynamicG1!(T::Vector{<: Real}, g1_v::Vector{<: Real}, y::Vector{<: Real}, x::Vector{<: Real}, params::Vector{<: Real}, steady_state::Vector{<: Real})
    @assert length(T) >= 6
    @assert length(g1_v) == 22
    @assert length(y) == 24
    @assert length(x) == 2
    @assert length(params) == 8
@inbounds begin
g1_v[1]=params[4];
g1_v[2]=(-(params[1]*1/y[5]));
g1_v[3]=(-1);
g1_v[4]=1-params[8];
g1_v[5]=(-(1/y[9]));
g1_v[6]=1;
g1_v[7]=(-(T[4]/(T[1]*T[1])));
g1_v[8]=(-(1/y[10]));
g1_v[9]=1;
g1_v[10]=params[8];
g1_v[11]=1;
g1_v[12]=(-1);
g1_v[13]=(1-params[8])*(-params[3]);
g1_v[14]=1/y[13];
g1_v[15]=(-(1/y[13]));
g1_v[16]=1;
g1_v[17]=1;
g1_v[18]=1;
g1_v[19]=(-1);
g1_v[20]=T[6]/(T[2]*T[2]);
g1_v[21]=(-1);
g1_v[22]=(-1);
end
    return nothing
end

