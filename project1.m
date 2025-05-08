close all, clear all
viewName='View_001';

gtData = readmatrix('gt.txt');
detector = peopleDetectorACF;

imgbk= 0;
thr=40;
minArea=300;
seqlength = 794;
baseNum = 0;
labelCount = 0;
alpha = 0.05; % Background update rate
heatDecay= 0.9;
numFrames=398;
frameBuffer=[];

%Control guide:
% 0: detector only with its success plot
% 1: 0 + heatmaps
% 2: 0 + Yolo algorithm and correspondent success plot
control = 0;

numFP = zeros(1, 10);       % One FP count per threshold
numFN = zeros(1, 10);       % One FN count per threshold
numTP = zeros(1,10);        % One TP count per threshold

% Store last five positions for each label
labelTrajectories = containers.Map('KeyType', 'int32', 'ValueType', 'any');


% heatmap initialization

se = strel('disk', 2);

figure; hold on;
backgroundPool = [];

for i=0:seqlength
    imgbki=imread(fullfile(viewName, sprintf('frame_%.4d.jpg', baseNum+i)));
    backgroundStack(:, :, :, i + 1) = imgbki;
end

imgbk = median(backgroundStack, 4); % Compute background image with median

hImg = [];
staticHeatmap = zeros(size(imgbk, 1), size(imgbk, 2));
dynamicHeatmap = zeros(size(imgbk, 1), size(imgbk, 2), seqlength);
current_frame = 1;

for i=0:seqlength
    imgfr= imread(fullfile(viewName, sprintf('frame_%.4d.jpg', baseNum+i)));
    if control==2
        subplot(1,2,1);
    elseif control==1
        subplot(2,2,1);
    end

    % Initialize bounding box arrays for this frame
    gtBoxes = [];      % [x1, y1, x2, y2]
    detBoxes = [];     % [x1, y1, x2, y2]

    gtBoxesIdentity = [];  % [x1, y1, x2, y2, identity]
    detBoxesIdentity = []; % [x1, y1, x2, y2, identity]
    
    if isempty(hImg)
        hImg = imshow(imgfr);
    else
        set(hImg, 'CData', imgfr);  
    end
    
    imgdif= (abs(double(imgbk(:,:,1))- double(imgfr(:,:,1)))> thr) | ...
        (abs(double(imgbk(:,:,2))- double(imgfr(:,:,2)))> thr) | ...
        (abs(double(imgbk(:,:,3))- double(imgfr(:,:,3)))> thr);
    

    bw = imgdif;
    bw = imopen(bw, se); % open -> erosion then dilation (removes small spots smaller than se)
    bw = imclose(bw, se); % close -> dilation then erosion (connect small dark gaps between objects)
    bw = imfill(bw, 'holes'); % fill -> fill completely enclosed back regions

    [lb num]=bwlabel(bw);
    regionProps= regionprops(lb, 'area', 'FilledImage', 'Centroid' );
    inds = find([regionProps.Area] > minArea);
    
    regnum=length(inds);
 
    % Delete previous rectangles
    delete(findall(gca, 'Type', 'rectangle'));
    delete(findall(gca, 'Type', 'text'));      % Delete old labels
    delete(findall(gca, 'Type', 'hggroup')); % Delete previous circles

    if regnum

        if control==1
            if current_frame == 1
                dynamicHeatmap(:, :, current_frame) = zeros(size(imgbk,1), size(imgbk,2));
            else
                dynamicHeatmap(:, :, current_frame) = dynamicHeatmap(:, :, current_frame - 1) * heatDecay;
            end
        end

        for j=1:regnum
           [lin col]= find(lb == inds(j));
           upLPoint = min([lin col]);
           dWindow = max([lin col]) - upLPoint + 1;
           centroid = regionProps(inds(j)).Centroid;

           labelID = inds(j);
            
            % Store last 5 positions per label
            if isKey(labelTrajectories, labelID)
                positions = labelTrajectories(labelID);
                positions = [positions; centroid];
                if size(positions, 1) > 5
                    positions(1, :) = [];
                end
                labelTrajectories(labelID) = positions;
            else
                labelTrajectories(labelID) = centroid;
            end

           bboxXYWH = [fliplr(upLPoint), fliplr(dWindow)];
           rectangle('Position', bboxXYWH, 'EdgeColor', [1,1,0], 'linewidth', 2);  % Draw yellow rectangles

           % Store as [x1, y1, x2, y2]
           x1 = bboxXYWH(1);
           y1 = bboxXYWH(2);
           x2 = x1 + bboxXYWH(3) - 1;
           y2 = y1 + bboxXYWH(4) - 1;
           detBoxes = [detBoxes; x1, y1, x2, y2];

           detBoxesIdentity = [detBoxesIdentity; x1, y1, x2, y2, -1];


           if control==1
               % Update Static Heatmap (accumulate centroids with Gaussian weighting)
               staticHeatmap = updateHeatmap(staticHeatmap, centroid, 30);
    
               % Update Dynamic Heatmap (for this frame)
               dynamicHeatmap(:, :, current_frame) = updateHeatmap(dynamicHeatmap(:, :, current_frame), centroid, 30);
           end

           % Draw last 5 positions as small yellow dots
           positions = labelTrajectories(labelID);
           viscircles(positions, repmat(3, size(positions,1), 1), 'Color', 'yellow');
       end
    end

    currentFrameGT= gtData(gtData(:,1) == (baseNum + i), :);

    if ~isempty(currentFrameGT)
        for k= 1:size(currentFrameGT, 1)

            identityNumber = currentFrameGT(k, 2);

            x = currentFrameGT(k, 3);
            y = currentFrameGT(k, 4);
            w = currentFrameGT(k, 5);
            h = currentFrameGT(k, 6);
            
            x1 = x;
            y1 = y;
            x2 = x + w - 1;
            y2 = y + h - 1;
            gtBoxes = [gtBoxes; x1, y1, x2, y2];

            gtBoxesIdentity = [gtBoxesIdentity; x1, y1, x2, y2, identityNumber];

            rectangle('Position', [x, y, w, h], 'EdgeColor', [1, 0, 0], 'LineWidth', 2); % Draw red rectangles

        end
    end

    for j = 1:size(gtBoxesIdentity, 1) % Red boxes

        maxIoU = 0;
        bestLabel = 0;
        
        gt = gtBoxesIdentity(j, 1:4); % Extract bounding box coordinates
        gtIdentity = gtBoxesIdentity(j, 5); % Extract identity label
        
        for i = 1:size(detBoxesIdentity, 1) % Yellow boxes

            det = detBoxesIdentity(i, 1:4); % Extract bounding box coordinates
            detIdentity = detBoxesIdentity (i, 5); % Maybe is not needed
            
            iou = computeIoU(gt, det);
            
            if iou > maxIoU
                maxIoU = iou;
                bestLabel = gtIdentity;
            end
        end

        detBoxesIdentity(j, 5) = bestLabel;
    end

    labelCount = 25;

    for i = 1:size(detBoxesIdentity, 1) % For the yellow boxes that are not in the GT

        if detBoxesIdentity(i, 5) == -1
            detBoxesIdentity(i, 5) = labelCount;
            labelCount = labelCount + 1;
        end

    end

    for i = 1:regnum % To display the labels
        % Assign and display label
        if control == 0
            text(detBoxesIdentity(i, 1)-30, detBoxesIdentity(i, 2)-30, num2str(detBoxesIdentity(i, 5)), 'Color', 'black', 'FontSize', 12, 'FontWeight', 'bold');
        else 
            text(detBoxesIdentity(i, 1)-30, detBoxesIdentity(i, 2)-70, num2str(detBoxesIdentity(i, 5)), 'Color', 'black', 'FontSize', 12, 'FontWeight', 'bold');

        end

    end

    if control==1
        subplot(2,2,2);
        imagesc(dynamicHeatmap(:, :, current_frame));
        colormap('jet');
        colorbar;
        title('Dynamic Heatmap');
    
   
        subplot(2,2,3);
        imagesc(staticHeatmap);
        colormap('jet');
        colorbar;
        title('Static Heatmap of Pedestrian Trajectories');
        current_frame = current_frame + 1;
    end
        


    %fprintf('Frame %d: %d GT boxes, %d Detected boxes\n', baseNum + i, size(gtBoxes,1), size(detBoxes,1));
    C_matrices = cell(1, 10);
    
    for j=1:10
        C_matrix = computeCmatrix(gtBoxes, detBoxes, j/10);
        C_matrices{j} = C_matrix;
        tp_flag = 0;

        % Check for False Negatives (rows of all 0s)
        if size(C_matrix, 1) > 0 && any(all(C_matrix == 0, 2))
            numFN(j) = numFN(j) + 1;
            tp_flag = 1;
        end

        % Check for False Positives (columns of all 0s)
        if size(C_matrix, 2) > 0 && any(all(C_matrix == 0, 1))
            numFP(j) = numFP(j) + 1;
            tp_flag = 1;
        end

        % If there were no FPs and FNs then we had an 100 per cent match
        if tp_flag == 0
            numTP(j) = numTP(j) + 1;
        end 
    end

    labelCount = 0;

    drawnow
end



numFP2 = zeros(1, 10);       % One FP count per threshold
numFN2 = zeros(1, 10);       % One FN count per threshold
numTP2 = zeros(1, 10);       % One TP count per threshold

if control==2
    for i=0:seqlength
        imgfr= imread(fullfile(viewName, sprintf('frame_%.4d.jpg', baseNum+i)));
      
        % Initialize bounding box arrays for this frame
        gtBoxes = [];      % [x1, y1, x2, y2]
        
        
        imgdif= (abs(double(imgbk(:,:,1))- double(imgfr(:,:,1)))> thr) | ...
            (abs(double(imgbk(:,:,2))- double(imgfr(:,:,2)))> thr) | ...
            (abs(double(imgbk(:,:,3))- double(imgfr(:,:,3)))> thr);
        
    
        bw = imgdif;
        bw = imopen(bw, se);
        bw = imclose(bw, se);
        bw = imfill(bw, 'holes');
        [lb num]=bwlabel(bw);
        regionProps= regionprops(lb, 'area', 'FilledImage', 'Centroid' );
        inds = find([regionProps.Area] > minArea);
        
        
    
        regnum=length(inds);
     
        % Delete previous rectangles
        delete(findall(gca, 'Type', 'rectangle'));
        delete(findall(gca, 'Type', 'text'));      % Delete old labels
        delete(findall(gca, 'Type', 'hggroup')); % Delete previous circles
        
    
        %YOLO
        [bboxes,scores] = detect(detector,imgfr);
        detBoxes = [];
        % Store as [x1, y1, x2, y2]
        for j = 1:size(bboxes,1)        
           x1 = bboxes(j,1);
           y1 = bboxes(j,2);
           x2 = x1 + bboxes(j,3) - 1;
           y2 = y1 + bboxes(j,4) - 1;
           detBoxes = [detBoxes; x1, y1, x2, y2];
        end
    
        
        imgfr = insertObjectAnnotation(imgfr,'rectangle',bboxes,scores);
    
        subplot(1,2,2);
        imshow(imgfr)
        title('YOLO');
           
    
        currentFrameGT= gtData(gtData(:,1) == (baseNum + i), :);
    
        if ~isempty(currentFrameGT)
            for k= 1:size(currentFrameGT, 1);
                x = currentFrameGT(k, 3);
                y = currentFrameGT(k, 4);
                w = currentFrameGT(k, 5);
                h = currentFrameGT(k, 6);
                
                x1 = x;
                y1 = y;
                x2 = x + w - 1;
                y2 = y + h - 1;
                gtBoxes = [gtBoxes; x1, y1, x2, y2];
    
                rectangle('Position', [x, y, w, h], 'EdgeColor', [1, 0, 0], 'LineWidth', 2);
    
            end
        end
    
        C_matrices2 = cell(1, 10);
        for j=1:10
            C_matrix = computeCmatrix(gtBoxes, detBoxes, j/10);
            C_matrices2{j} = C_matrix;
            tp_flag2 = 0;


    
            % Check for False Negatives (rows of all 0s)
            if size(C_matrix, 1) > 0 && any(all(C_matrix == 0, 2))
                numFN2(j) = numFN2(j) + 1;
                tp_flag2 = 1;
            end
    
            % Check for False Positives (columns of all 0s)
            if size(C_matrix, 2) > 0 && any(all(C_matrix == 0, 1))
                numFP2(j) = numFP2(j) + 1;
                tp_flag2 = 1;
            end

            if tp_flag2 == 0
                numTP2(j) = numTP2(j) + 1;
            end
        end
    
        labelCount = 0;
    
        drawnow
    end
end



%Dynamic display (se formos usar temos de meter dentro do for para ser
%mostrado em cada frame)
% figure;
% imagesc(dynamicHeatmap(:, :, seqlength));
% colormap('hot');
% colorbar;
% title('Dynamic Heatmap (Last Frame)');

% auxiliar function to update heatmap (Gaussian distance metric)
function updatedHeatmap = updateHeatmap(heatmap, centroid, sigma)
    [rows, cols] = size(heatmap);
    [X, Y] = meshgrid(1:cols, 1:rows);
    % Gaussian distribution centered at the centroid
    gaussianDist = exp(-((X - centroid(1)).^2 + (Y - centroid(2)).^2) / (2 * sigma^2));
    updatedHeatmap = heatmap + gaussianDist;
end



function C_matrix = computeCmatrix(gtBoxes, algBoxes, iouThreshold)
    numGT = size(gtBoxes, 1);
    numDet = size(algBoxes, 1);
    C_matrix = zeros(numGT, numDet);
    numberOfones = 0;

    for i = 1:numGT
        gt = gtBoxes(i, :);
        for j = 1:numDet
            det = algBoxes(j, :);
            iou = computeIoU(gt, det);
            if iou > iouThreshold
                C_matrix(i, j) = 1;
                numberOfones = numberOfones + 1;
            end
        end
    end


end

function iou = computeIoU(boxA, boxB)
    % boxA and boxB are [x1, y1, x2, y2]

    % Calculate intersection coordinates
    x_left = max(boxA(1), boxB(1));
    y_top = max(boxA(2), boxB(2));
    x_right = min(boxA(3), boxB(3));
    y_bottom = min(boxA(4), boxB(4));

    if x_right < x_left || y_bottom < y_top
        iou = 0;
        return;
    end

    intersectionArea = (x_right - x_left + 1) * (y_bottom - y_top + 1);

    boxAArea = (boxA(3) - boxA(1) + 1) * (boxA(4) - boxA(2) + 1);
    boxBArea = (boxB(3) - boxB(1) + 1) * (boxB(4) - boxB(2) + 1);

    unionArea = boxAArea + boxBArea - intersectionArea;

    iou = intersectionArea / unionArea;

end


% IoU thresholds from 0.1 to 1.0
thresholds = 0.1:0.1:1.0;

% Convert FP/FN counts to percentages
fp_percent = 100 * numFP / seqlength;
if control==2
    fp2_percent = 100 * numFP2 / seqlength;
    fn2_percent = 100 * numFN2 / seqlength;
    tp2_percent = 100 * numTP2 / seqlength;
end
fn_percent = 100 * numFN / seqlength;
tp_percent = 100 * numTP / seqlength;

% Plot FP and FN percentages per threshold
figure;
plot(thresholds, fp_percent, '-o', 'LineWidth', 2);
hold on;
plot(thresholds, fn_percent, '-s', 'LineWidth', 2);
hold on;
plot(thresholds, tp_percent, '--s', 'LineWidth',2);
if control==2
    hold on;
    plot(thresholds, fp2_percent, '-^', 'LineWidth', 2);
    hold on;
    plot(thresholds, fn2_percent, '-', 'LineWidth', 2);
    hold on;
    plot(thresholds, tp2_percent, '--', 'LineWidth', 2);
end

xlabel('IoU Threshold');
ylabel('% of Frames with TP / FP / FN');
title(' True Positive and False Positive / Negative per IoU Threshold');
if control==2
    legend('False Positives', 'False Negatives', 'True Positives', 'YOLOs False Positives', 'YOLOs False Negatives', 'Yolo True Positives');
else
    legend('False Positives', 'False Negatives', 'True Positives');
end

grid on;
