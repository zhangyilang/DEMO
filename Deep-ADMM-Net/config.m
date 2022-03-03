%% network setting

global nnconfig;
nnconfig.FilterNumber = 8;
nnconfig.FilterSize = 3 ;
nnconfig.Stage = 10;
nnconfig.Padding = 1;
nnconfig.LinearLabel = double(-1:0.02:1);
nnconfig.WeightDecay = 0;
nnconfig.ImageSize = [256,256];

nnconfig.EnableGPU = false;

nnconfig.NoiseType = 'Motion';
nnconfig.OutlierProb = 0.001;
nnconfig.MotionProb = 0.05;
nnconfig.MotionDist = 29;    % 2.5cm
nnconfig.MotionDelay = 1/20; % 0.1pi/2pi = 1/20
nnconfig.SNR = 20;
