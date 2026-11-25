% Helper for tools/accept_octave.py: save Dynare's results of the last run, stripped of what -mat7-binary cannot hold.
% This file is part of pydyn (GPL-3.0-or-later).
function accept_save(outfile)
  global M_ oo_ options_
  S = struct();
  if exist('oo_', 'var') && ~isempty(oo_), S.oo_ = accept_strip(oo_, 0); end
  if exist('M_', 'var') && ~isempty(M_), S.M_ = accept_strip(M_, 0); end
  if exist('options_', 'var') && ~isempty(options_), S.options_ = accept_strip(options_, 0); end
  save('-mat7-binary', outfile, '-struct', 'S');
end

function y = accept_strip(x, depth)
  % keep numeric, logical, char, cell, struct; drop everything else (dseries, dates, handles, objects)
  y = [];
  if depth > 12, return, end
  if isstruct(x)
    if numel(x) > 200, y = []; return, end
    if numel(x) == 1
      y = struct();
      f = fieldnames(x);
      for i = 1:numel(f)
        v = accept_strip(x.(f{i}), depth + 1);
        if ~(isequal(size(v), [0 0]) && ~ischar(v) && ~iscell(v) && ~isstruct(v) && ~isempty(strfind('', 'x')) ) || true
          if ~isempty(v) || ischar(v) || iscell(v) || isstruct(v) || isnumeric(v)
            y.(f{i}) = v;
          end
        end
      end
    else
      y = cell(size(x));
      for i = 1:numel(x), y{i} = accept_strip(x(i), depth + 1); end
    end
  elseif iscell(x)
    y = cell(size(x));
    for i = 1:numel(x), y{i} = accept_strip(x{i}, depth + 1); end
  elseif isnumeric(x) || islogical(x)
    if issparse(x), x = full(x); end
    if numel(x) > 5e6, y = []; else, y = x; end
  elseif ischar(x)
    y = x;
  else
    y = [];     % function handle, object (dseries, dates, ...)
  end
end
