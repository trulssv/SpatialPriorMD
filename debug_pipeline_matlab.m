%% debug_pipeline_matlab.m
%
% Debug script for the MATLAB pipeline (UZB calibration phantom).
%
% Mirrors the AttenuationMD.m / UZB_main.m pipeline exactly:
%   - Loads water/iodine basis maps and mask directly from NIfTI files
%   - No SegmentSimple, no GetMu, no bowtie correction
%   - Solves with the INITIAL weight functions (files/weights.mat)
%
% The mask is loaded directly from mask_vol.nii — no segmentation is computed.
%
% Run from the AttenuationMD workspace root.
%
% Outputs (saved to debug/matlab_intermediates.mat)
% --------------------------------------------------
%   aB_w, aB_i   : [H x W]  background basis images  (water, iodine)
%   aR_w, aR_i   : [H x W]  contrast-region images   (water, iodine)
%   mask         : [H x W]  binary contrast mask (loaded from NIfTI)
%   AB_w_sino, AB_i_sino : [nDet x nAngles]  basis pathlength sinograms
%   AR_w_sino, AR_i_sino : [nDet x nAngles]  contrast pathlength sinograms
%   AC_I_gt, AC_Gd_gt    : [nDet x nAngles]  GT contrast sinograms
%   AC_I_pred, AC_Gd_pred: [nDet x nAngles]  predicted contrast sinograms
%   rotAngles    : [1 x nAngles]  projection angles [deg]

clear all; close all; clc;

%% ----------------------------------------------------------------
%  1. Set up paths (mirrors UZB_main.m)
%% ----------------------------------------------------------------
addpath(genpath('./'))

%% ----------------------------------------------------------------
%  2. Select scan
%% ----------------------------------------------------------------
CAL_SCAN  = 'CT99.449.2.2.1765289969.251640.303.12253';   % calibration
EVAL_SCAN = 'CT99.456.2.1.1765364800.382548.302.12277';   % evaluation

scan_id   = CAL_SCAN;   % <-- change to EVAL_SCAN to debug evaluation scan
scan_dir  = fullfile('./UZB/data/mega_mono', scan_id);

DENSITY_I  = 4933;   % mg/mL (= mg/cm^3)
DENSITY_GD = 7900;   % mg/mL
pixelstomm = 200 / 512;  % FOV 200 mm / 512 pixels

output_dir = './debug';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

fprintf('Scan: %s\n', scan_id);

%% ----------------------------------------------------------------
%  3. Load mid-Z slice from NIfTI volumes
%% ----------------------------------------------------------------
fprintf('[1/4] Loading NIfTI data...\n');

water_vol  = double(niftiread(fullfile(scan_dir, 'water_basis_vol.nii')));   % volume fraction
iodine_vol = double(niftiread(fullfile(scan_dir, 'iodine_basis_vol.nii'))); % volume fraction
igt_vol    = double(niftiread(fullfile(scan_dir, 'iodine_gt.nii')));         % mg/mL
gdgt_vol   = double(niftiread(fullfile(scan_dir, 'gadolinium_gt.nii')));     % mg/mL
mask_vol   = double(niftiread(fullfile(scan_dir, 'mask_vol.nii')));          % 0 or 1

% Take the mid-Z axial slice  (data layout: Z x H x W)
nz  = size(water_vol, 1);
mid = round(nz / 2);

water  = squeeze(water_vol(mid,  :, :));
iodine = squeeze(iodine_vol(mid, :, :));
igt    = squeeze(igt_vol(mid,    :, :));
gdgt   = squeeze(gdgt_vol(mid,   :, :));
mask   = squeeze(mask_vol(mid,   :, :)) > 0.5;   % logical

% Clamp basis images (negative values = reconstruction artefacts)
water  = max(water,  0);
iodine = max(iodine, 0);

fprintf('     Image size: %dx%d,  pixelstomm=%.4f,  mask pixels: %d\n', ...
    size(water, 1), size(water, 2), pixelstomm, sum(mask(:)));

%% ----------------------------------------------------------------
%  4. Segment into background (aB) and contrast region (aR)
%     Mirrors AttenuationMD.m exactly:
%       wB(mask) = 1  => background water = 1 inside mask
%       iB(mask) = 0  => background iodine = 0 inside mask
%% ----------------------------------------------------------------
fprintf('[2/4] Segmenting into aB and aR...\n');

aB_w = water;
aB_w(mask) = 1;          % background water: set mask region to 1 (pure water)
aR_w = water - aB_w;     % contrast contribution: 0 outside, (water-1) inside

aB_i = iodine;
aB_i(mask) = 0;          % background iodine: zero inside mask
aR_i = iodine - aB_i;    % contrast contribution: iodine inside mask, 0 outside

% GT contrast volume fractions
igt_vf  = igt  / DENSITY_I;
gdgt_vf = gdgt / DENSITY_GD;

%% ----------------------------------------------------------------
%  5. Fanbeam forward projection  -->  AB, AR sinograms
%     Mirrors AttenuationMD.m (no bowtie correction)
%% ----------------------------------------------------------------
fprintf('[3/4] Fanbeam projection...\n');

D_px         = 625 / pixelstomm;
spacing      = 0.1;
rot_increment = 1;

fan_args = {'FanSensorGeometry', 'arc', ...
            'FanSensorSpacing',   spacing, ...
            'FanRotationIncrement', rot_increment};

proj_fn = @(img) pixelstomm * fanbeam(img, D_px, fan_args{:});

% Obtain angle vectors from first projection call
[AB_w_sino, ~, rotAngles] = fanbeam(aB_w, D_px, fan_args{:});
AB_w_sino = pixelstomm * AB_w_sino;

AB_i_sino  = proj_fn(aB_i);
AR_w_sino  = proj_fn(aR_w);
AR_i_sino  = proj_fn(aR_i);
AC_I_gt    = proj_fn(igt_vf);
AC_Gd_gt   = proj_fn(gdgt_vf);

fprintf('     Sinogram size: %d (detectors) x %d (angles)\n', ...
    size(AB_w_sino, 1), size(AB_w_sino, 2));

AB = {AB_w_sino, AB_i_sino};
AR = {AR_w_sino, AR_i_sino};

%% ----------------------------------------------------------------
%  6. Solve with initial weights (files/weights.mat)
%     Solve.m loads weights.mat, MuGdimm.mat, MuIimm.mat, MuWaimm.mat
%     from the MATLAB path — all found in files/ and files/materials/.
%% ----------------------------------------------------------------
fprintf('[4/4] Solving with initial weights (files/weights.mat)...\n');

AC_est = Solve(AB, AR);

AC_I_pred  = AC_est{1};
AC_Gd_pred = AC_est{2};

fprintf('     AC_I  range: [%.6f, %.6f]\n', min(AC_I_pred(:)),  max(AC_I_pred(:)));
fprintf('     AC_Gd range: [%.6f, %.6f]\n', min(AC_Gd_pred(:)), max(AC_Gd_pred(:)));

%% ----------------------------------------------------------------
%  7. Save intermediates for Python comparison
%% ----------------------------------------------------------------
out_file = fullfile(output_dir, 'matlab_intermediates.mat');
fprintf('Saving intermediates to %s...\n', out_file);

save(out_file, ...
    'aB_w', 'aB_i', ...
    'aR_w', 'aR_i', ...
    'mask', ...
    'AB_w_sino', 'AB_i_sino', ...
    'AR_w_sino', 'AR_i_sino', ...
    'AC_I_gt', 'AC_Gd_gt', ...
    'AC_I_pred', 'AC_Gd_pred', ...
    'rotAngles', 'pixelstomm');

fprintf('Saved.\n');

%% ----------------------------------------------------------------
%  8. Plot MATLAB results
%% ----------------------------------------------------------------
fprintf('Plotting MATLAB results...\n');

% --- Figure 1: Segmentations aB and aR ---
fh1 = figure('Name', 'MATLAB Debug: Segmentations (aB, aR)', 'NumberTitle', 'off', ...
             'Position', [100 100 1000 450]);

subplot(2,2,1);
imagesc(aB_w);  colormap(gca, 'gray'); colorbar;
title('aB: water (background)');  axis image off;

subplot(2,2,2);
imagesc(aB_i);  colormap(gca, 'gray'); colorbar;
title('aB: iodine (background)'); axis image off;

subplot(2,2,3);
imagesc(aR_w);  colormap(gca, 'hot'); colorbar;
title('aR: water (contrast region)'); axis image off;

subplot(2,2,4);
imagesc(aR_i);  colormap(gca, 'hot'); colorbar;
title('aR: iodine (contrast region)'); axis image off;

sgtitle(sprintf('MATLAB Segmentations  (%s)', scan_id), 'Interpreter', 'none');
saveas(fh1, fullfile(output_dir, 'matlab_segmentations.png'));

% --- Figure 2: Pathlength sinograms AB and AR ---
fh2 = figure('Name', 'MATLAB Debug: Sinograms AB, AR', 'NumberTitle', 'off', ...
             'Position', [150 150 1200 500]);

subplot(2,2,1);
imagesc(rotAngles, [], AB_w_sino); colorbar;
title('AB: water sino'); xlabel('Angle [deg]'); ylabel('Det. index'); axis tight;

subplot(2,2,2);
imagesc(rotAngles, [], AB_i_sino); colorbar;
title('AB: iodine sino'); xlabel('Angle [deg]'); ylabel('Det. index'); axis tight;

subplot(2,2,3);
imagesc(rotAngles, [], AR_w_sino); colorbar;
title('AR: water sino (contrast region)'); xlabel('Angle [deg]'); ylabel('Det. index'); axis tight;

subplot(2,2,4);
imagesc(rotAngles, [], AR_i_sino); colorbar;
title('AR: iodine sino (contrast region)'); xlabel('Angle [deg]'); ylabel('Det. index'); axis tight;

sgtitle('MATLAB Pathlength Sinograms AB and AR');
saveas(fh2, fullfile(output_dir, 'matlab_sinograms_AB_AR.png'));

% --- Figure 3: Predicted vs GT contrast pathlengths AC ---
fh3 = figure('Name', 'MATLAB Debug: Predicted vs GT AC', 'NumberTitle', 'off', ...
             'Position', [200 200 1200 500]);

subplot(2,3,1);
imagesc(rotAngles, [], AC_I_pred); colorbar;
title('AC pred: I sino'); xlabel('Angle [deg]'); ylabel('Det. index'); axis tight;

subplot(2,3,2);
imagesc(rotAngles, [], AC_I_gt); colorbar;
title('AC GT: I sino'); xlabel('Angle [deg]'); ylabel('Det. index'); axis tight;

subplot(2,3,3);
im = imagesc(rotAngles, [], AC_I_pred - AC_I_gt); colorbar; axis tight;
clim_val = max(abs([min(AC_I_pred(:)-AC_I_gt(:)), max(AC_I_pred(:)-AC_I_gt(:))]));
clim([-clim_val clim_val]);
title('Residual (pred - GT): I'); xlabel('Angle [deg]'); ylabel('Det. index');

subplot(2,3,4);
imagesc(rotAngles, [], AC_Gd_pred); colorbar;
title('AC pred: Gd sino'); xlabel('Angle [deg]'); ylabel('Det. index'); axis tight;

subplot(2,3,5);
imagesc(rotAngles, [], AC_Gd_gt); colorbar;
title('AC GT: Gd sino'); xlabel('Angle [deg]'); ylabel('Det. index'); axis tight;

subplot(2,3,6);
imagesc(rotAngles, [], AC_Gd_pred - AC_Gd_gt); colorbar; axis tight;
clim_val = max(abs([min(AC_Gd_pred(:)-AC_Gd_gt(:)), max(AC_Gd_pred(:)-AC_Gd_gt(:))]));
clim([-clim_val clim_val]);
title('Residual (pred - GT): Gd'); xlabel('Angle [deg]'); ylabel('Det. index');

sgtitle('MATLAB: Predicted vs GT Contrast Pathlengths AC');
saveas(fh3, fullfile(output_dir, 'matlab_AC_pred_vs_gt.png'));

fprintf('All plots saved to %s/\n', output_dir);
