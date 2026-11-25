% Octave reference run of dseries field assignment, lag and merge semantics (Dynare 7.2 in the path):
%     octave --no-gui --quiet r_dseries_assign.m (from this folder, after addpath(<dynare>/matlab); dynare_config)
% The statements are those of tests/test_dseries_unified.py::SCRIPT; results are saved to dseries_assign.mat.
a = dseries([1 2 3; 4 5 6; 7 8 9], dates('2000Q1'), {'b','a','c'});
x = dseries([10;11;12], dates('2000Q2'), 'z');
a.z = x;
d1 = a.data; n1 = a.name;
a{'b'} = dseries([0;0;0;0],dates('2000Q1'),'q');
d2 = a.data;
a.init = dates('2001Q1'); i3 = a.init.char;
l = a.lag(1); ld = l.data;
a{'a'}(dates('2001Q2'):dates('2001Q3')) = [100;200]; d4 = a.data;
b = dseries([1 2; 3 4; 5 6], dates('2000Q1'), {'u','v'});
b.u(dates('2000Q2')) = 50; d5 = b.data;
b{'v'} = b{'u'}.lag(1); d6 = b.data; n6 = b.name;
save -mat7-binary dseries_assign.mat d1 n1 d2 i3 ld d4 d5 d6 n6
