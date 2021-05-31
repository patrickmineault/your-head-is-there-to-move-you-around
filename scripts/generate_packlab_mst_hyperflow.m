allcells
for i = 1:61
    i
    metadata = cells(i);
    
    % To be consistent
    metadata.hf.samplingratet = 30;
    metadata.hf.designsizet = 10/30;
    hfdata = readHyperFlow('../', metadata.hf);
    hfdesign = getHyperFlowDesign(hfdata, metadata, false);
    
    data = struct();
    data.name = metadata.name;
    data.stmatcheshf = metadata.stmatcheshf;
    
    data.Y_hf = hfdesign.y;
    data.stim_hf = hfdesign.Xraw;
    data.stimidx_hf = hfdesign.Xidx;
    data.gridx_hf = hfdesign.gridx;
    data.gridy_hf = hfdesign.gridy;
    
    if ~strcmp(metadata.st.spikes, '')
        stdata = readSuperTune(metadata.st);
        stdesign = getSuperTuneDesign(stdata, metadata.st);
        
        data.Y_st = stdesign.y;
        data.Yall_st = stdesign.yall;
        data.X_st = stdesign.X;
        data.gridx_st = stdesign.gridx;
        data.gridy_st = stdesign.gridy;
    end
    
    save(sprintf('exporteddata/%s.mat', metadata.name), '-struct', 'data', '-v7.3');
end