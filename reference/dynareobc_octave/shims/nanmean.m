function m = nanmean(x, varargin)
    % Octave shim: mean ignoring NaN
    n = ~isnan(x); x(~n) = 0;
    if isempty(varargin), d = find(size(x) ~= 1, 1); if isempty(d), d = 1; end, else, d = varargin{1}; end
    m = sum(x, d) ./ sum(n, d);
end
