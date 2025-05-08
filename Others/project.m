clc; clear; close all;

% Path to dataset and ground truth file
dataset_path = 'Crowd_PETS/S2/L1/Time_12-34/View_001';
gt_file = 'PETS-S2L1/gt/gt.txt';

% Read ground truth data
gt_data = readmatrix(gt_file);

% Extract columns from gt.txt
frame_numbers = gt_data(:, 1);   % Frame index
pedestrian_ids = gt_data(:, 2);  % Unique pedestrian ID
bbox_left = gt_data(:, 3);       % Bounding box X
bbox_top = gt_data(:, 4);        % Bounding box Y
bbox_width = gt_data(:, 5);      % Width
bbox_height = gt_data(:, 6);     % Height

% Get total number of frames
unique_frames = unique(frame_numbers);

% Background Estimation using Median Method
numFrames = 50; % Number of frames to use for background estimation
vid4D = [];
for i = 1:numFrames
    frameFile = fullfile(dataset_path, sprintf('frame_%04d.jpg', unique_frames(i)));
    if exist(frameFile, 'file')
        vid4D(:,:,:,i) = imread(frameFile);
    end
end
bkg = median(double(vid4D), 4);
bkg = uint8(bkg); % Convert back to uint8 format

% Background subtraction parameters
thr = 40;
minArea = 1;
se = strel('disk',2);

% Initialize figure
figure; hold on;

% Loop through frames
for i = 1:length(unique_frames)
    frame_idx = unique_frames(i);
    img_file = fullfile(dataset_path, sprintf('frame_%04d.jpg', frame_idx)); % Update filename pattern
    
    if exist(img_file, 'file')
        imgfr = imread(img_file);
        
        % Display original image
        subplot(2,2,1); imshow(imgfr); title('Original Frame'); hold on;
        
        % Ground Truth Bounding Boxes
        idx = frame_numbers == frame_idx;
        for j = find(idx)'
            rectangle('Position', [bbox_left(j), bbox_top(j), bbox_width(j), bbox_height(j)], 'EdgeColor', 'r', 'LineWidth', 2);
            text(bbox_left(j), bbox_top(j)-5, sprintf('ID: %d', pedestrian_ids(j)), 'Color', 'yellow', 'FontSize', 10);
        end
        hold off;
        
        % Background subtraction for object detection
        imgdif = (abs(double(bkg(:,:,1))-double(imgfr(:,:,1)))>thr) | ...
                 (abs(double(bkg(:,:,2))-double(imgfr(:,:,2)))>thr) | ...
                 (abs(double(bkg(:,:,3))-double(imgfr(:,:,3)))>thr);
        
        bw1 = imclose(imgdif, se);
        bw2 = imerode(bw1, se);
        
        % Display background subtraction steps
        subplot(2,2,2); imshow(imgdif); title('Difference Image');
        subplot(2,2,3); imshow(bw1); title('Morphological Closing');
        subplot(2,2,4); imshow(bw2); title('Eroded Image');
        
        % Label detected objects
        [lb, num] = bwlabel(bw2);
        regionProps = regionprops(lb, 'Area', 'Centroid');
        inds = find([regionProps.Area] > minArea);
        
        % Draw detected bounding boxes
        for j = 1:length(inds)
            [lin, col] = find(lb == inds(j));
            upLPoint = min([lin col]);
            dWindow = max([lin col]) - upLPoint + 1;
            rectangle('Position', [fliplr(upLPoint) fliplr(dWindow)], 'EdgeColor', 'g', 'LineWidth', 2);
        end
        
        drawnow;
    else
        fprintf('Frame %d not found. Skipping...\n', frame_idx);
    end
end
