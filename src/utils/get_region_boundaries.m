%% MATLAB
% This script helps generate images with neuron region boundaries

function L = get_region_boundaries(args)
	filename = args.arg1;
	image = imread(filename);
	[B, L] = bwboundaries(image, 'holes');
end
