function obc_octave( InputFileName, varargin )
% Minimal driver for DynareOBC under Octave: what dynareOBCSetup does, without updates, MEX compilation or
% requirement downloads.
    P = '/home/claude/ref/dynareOBC';
    addpath( '/home/claude/ref/d45/dynare-4.5.4/matlab' );
    dirs = { '/Core/', '/Core/Utils/', '/Core/MChecks/', '/Core/MODProcessing/', '/Extern/nlma/', '/Core/BaseSimulation/', ...
             '/Core/Global/', '/Core/ModelSolution/', '/Core/OBCSimulation/', '/Core/Display/', '/Extern/DoubleDouble/', ...
             '/Core/Estimation/', '/Core/InnerProblem/', '/Core/Setup/', '/Core/Output/' };
    for i = 1 : numel( dirs ), if exist( [ P dirs{i} ], 'dir' ), addpath( [ P dirs{i} ] ); end, end
    addpath( genpath( [ P '/Extern/YALMIP' ] ) );
    addpath( '/home/claude/work/obcoct/mex' );
    addpath( '/home/claude/work/obcoct/shims' );
    try, pkg load statistics; catch, end
    global dynareOBC_ M_ options_ oo_
    dynareOBC_ = struct;
    dynareOBC_.DynareVersion = 4.5;
    dynareOBC_.BaseFileName = strrep( InputFileName, '.mod', '' );
    dynareOBC_ = SetDefaultOptions( dynareOBC_ );
    basevarargin = cell( 1, 0 );
    for i = 1 : length( varargin )
        [ basevarargin, dynareOBC_ ] = ProcessArgument( varargin{ i }, basevarargin, dynareOBC_ );
    end
    if dynareOBC_.Global || dynareOBC_.CubatureRegions > 1 || dynareOBC_.CubatureCATCHDegree > 0 || dynareOBC_.HigherOrderSobolDegree > 0
        dynareOBC_.Cubature = true;
    end
    dynareOBC_ = rmfield( dynareOBC_, { 'FastCubature', 'ImportanceSampling', 'NoCubature' } );
    basevarargin( end + 1 : end + 6 ) = { 'noclearall', 'console', 'nograph', 'nointeractive', '-DdynareOBC=1', 'nolinemacro' };
    dynareOBC_.OtherMOD = [];
    dynareOBC_ = dynareOBCCore( InputFileName, basevarargin, dynareOBC_, @() 0 );
end
