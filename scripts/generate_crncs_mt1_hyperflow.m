% This reuses some of the scripts included in the crcns-mt1 dataset.
addpath('/mnt/e/data_derived/crcns-mt1/crcns-mt1-MatLab-scripts');
%%
basepath = '/mnt/e/data_derived/crcns-mt1/crcns-mt1-data/';
ds = dir([basepath '*.mat']);

%%
histmat = histogram2(mtdata.eyeloc(:,1), mtdata.eyeloc(:,2), -5:0.2:5, -5:0.2:5);

histmat.DisplayStyle = 'tile';

%%
ii = 10;
cellname = ds(ii).name;
load([basepath cellname]);


%%
for ii = 1:length(ds)
    cellname = ds(ii).name;
    load([basepath cellname]);
    
    data = struct();
    data.name = cellname(1:end-4);
    data.stmatcheshf = false;
    
    width = round(3 * mean(std(mtdata.aperturecenter)) + mtdata.aperturediameter);

    disp(' Reconstructing the velocity field ...')
    params = struct('designsizex', width, 'designsizey', width, 'spatres', width/56,...
        'maskdiameter',mtdata.aperturediameter);
    Nx = round(params.designsizex/params.spatres(1));

    % Reconstruct the stimulus
    stimorg = GetVelField(params, mtdata.opticflows, mtdata.aperturecenter);

    % Downsample temporally 2-fold.
    eyeloc = (mtdata.eyeloc(1:2:end-1, :) + mtdata.eyeloc(2:2:end, :))/2;
    spkbinned = (mtdata.spkbinned(1:2:end-1) + mtdata.spkbinned(2:2:end));
    stimorg = (stimorg(1:2:end-1, :) + stimorg(2:2:end, :)) / 2;
    
    % Make sure most of the stimulus is within view.
    npixels = max(sum(abs(stimorg)>0, 2));
    validx = sum(abs(stimorg)>0, 2) > npixels / 2;
    
    validx = validx & (abs(eyeloc(:,1)) < 3) & (abs(eyeloc(:,2)) < 3);
    
    %data.
    % add a handful of times after for padding
    ntau = 3;
    Xidx = bsxfun(@plus, (1:size(stimorg, 1))' - 10 + ntau, (0:9));
    
    % Sampling grid
    rg = 0:params.spatres:params.designsizex;
    rg = rg(1:end-1);
    rg = rg - mean(rg);
    [xi, yi] = meshgrid(rg, rg);
    
    goodidx = validx & all((Xidx >= 1) & (Xidx <= size(stimorg, 1)), 2);
    
    fprintf('Cell %s, Good idx: %.3f, total samples %d\n', cellname, mean(goodidx), sum(goodidx));
    
    framerate = 500 / mtdata.dt;
    t = (1:size(stimorg,1))' / framerate;
    
    data.Y_hf = spkbinned(goodidx);
    data.stim_hf = stimorg;
    data.stimidx_hf = Xidx(goodidx, :);
    data.gridx_hf = xi;
    data.gridy_hf = yi;
    data.t = t(goodidx);
    
    save(sprintf('/mnt/e/data_derived/crcns-mt1/designmats/cell%02d.mat', str2double(cellname(5:end-4))), '-struct', 'data', '-v7.3');
end



%% Reconstruct the velocity field: ...
% actual screen size for the experiment was 49 deg x 36 deg
% the center of the velocity field is given in mtdata.RFloc
% please make sure the sampling grid does not go off the screen, or
% manually set part of the velocity field to be zeros

disp(' Reconstructing the velocity field ...')
params = struct('designsizex',36,'designsizey',36,'spatres',1.5,...
    'maskdiameter',mtdata.aperturediameter);
Nx = round(params.designsizex/params.spatres(1));

% Reconstruct the stimulus
stimorg = GetVelField(params, mtdata.opticflows, mtdata.aperturecenter);

% Downsample temporally 2-fold.
eyeloc = (mtdata.eyeloc(1:2:end, :) + mtdata.eyeloc(2:2:end, :))/2;
spkbinned = (mtdata.spkbinned(1:2:end) + mtdata.spkbinned(2:2:end));
stimorg = (stimorg(1:2:end, :) + stimorg(2:2:end, :)) / 2;

% Convert speed to a logarithmic scale
% stim = LogSpeedTune(stimorg); 

%%

% Convert speed to a logarithmic scale
stim = LogSpeedTune(stimorg); 

% Visualize one frame of the stimulus (in this case, the 100th frame)
figure(1); clf; 
PlotSpatialk( stim(100,:), Nx, params.spatres ); title(' Example Velocity Field')

%% Calculate spike-triggered average
disp(' Performing Spike Triggered Analysis ...')
Ndelay = round(1000/(mtdata.dt * 2));
[staTK, staSK, STAst] = TwoDSTA( stim, spkbinned, [], Ndelay );

%%
% Plot sta 
figure(2); clf;
subplot(2,2,1); PlotSpatialk(staSK, Nx, params.spatres);
subplot(2,2,2); plot((0:Ndelay-1)*mtdata.dt*2,staTK)

%%


