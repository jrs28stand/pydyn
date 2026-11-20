addpath /home/claude/dl/src72/matlab
S = load('collard_2001_simulated_moments_ref.mat'); d=S.oo_.dr;
r=struct('ghx',d.ghx,'ghu',d.ghu,'ghxx',d.ghxx,'ghxu',d.ghxu,'ghuu',d.ghuu,'ghs2',d.ghs2,'ys',d.ys,'params',S.M_.params,'Sigma_e',S.M_.Sigma_e);
save('-mat7-binary','clean_collard_order2.mat','-struct','r');
dynare collard_o3 nograph
d=oo_.dr; r=struct('ys',d.ys,'params',M_.params,'Sigma_e',M_.Sigma_e);
fn=fieldnames(d);
for i=1:numel(fn), if ~isempty(regexp(fn{i},'^(g_\d|gh[xus0-9]+)$','once')), r.(fn{i})=d.(fn{i}); end, end
disp(fieldnames(r)');
save('-mat7-binary','clean_collard_order3.mat','-struct','r');
