% smo_sing.mod variants: 1 stops with error 422; 2 and 3 switch to the univariate smoother.
addpath /home/claude/dl/src72/matlab
res = struct();
for v = 1:3
  pause(1.5); rehash;
  try
    eval(sprintf('dynare smo_sing nograph -DVARIANT=%d', v));
    res.(sprintf('v%d_failed', v)) = 0;
    res.(sprintf('v%d_x', v)) = oo_.SmoothedVariables.x;
    res.(sprintf('v%d_x_upd', v)) = oo_.UpdatedVariables.x;
    res.(sprintf('v%d_e', v)) = oo_.SmoothedShocks.e;
  catch err
    res.(sprintf('v%d_failed', v)) = 1;
    res.(sprintf('v%d_message', v)) = err.message;
  end
end
save('-mat7-binary', 'clean_smo_sing.mat', '-struct', 'res');
disp(res)
