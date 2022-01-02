close all
clearvars
clc
disp('Elias Assaf 315284729 - Jameel Nassar 206985152')
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1.1
%%to do

%% Section 1.2
%%to do

%% Section 1.3
mona_org = imread("mona_org.jpg");
mona_org = double(rgb2gray(mona_org))/256;

%% Section 1.4
figure;
imshow(mona_org);hold on;title('detecting SURF features using Matlab defualt Input Arguments.');

tic
points = detectSURFFeatures(mona_org);
time = toc;
plot(points);
fprintf('The runtime was %i, We used the defualt Input Arguments. \n', time);
fprintf('We found %i features \n', length(points));

%% Section 1.5
figure;
imshow(mona_org);hold on;title('detecting SURF features using Rectangular ROI.');
tic
points = detectSURFFeatures(mona_org,'ROI', [59, 5, 128, 120]);
time = toc;
plot(points);
fprintf('The runtime was %i, We used Rectangular region of interest between [59, 5, 128, 120]. \n', time);
fprintf('We found %i features \n', length(points));


%% Section 1.6
figure;sgtitle("detecting SURF features using different number of octaves")
subplot(1,2,1);imshow(mona_org);hold on;title('NumOctaves = 2')
tic
points = detectSURFFeatures(mona_org,'NumOctaves', 2);
time = toc;
plot(points);
fprintf('The runtime was %i, We used 2 as the number of octaves. \n', time);
fprintf('We found %i features \n', length(points));


subplot(1,2,2);imshow(mona_org);hold on;title('NumOctaves = 5')
tic
points = detectSURFFeatures(mona_org,'NumOctaves', 5);
time = toc;
plot(points);
fprintf('The runtime was %i, We used 5 as the number of octaves. \n', time);
fprintf('We found %i features \n', length(points));


%% Section 1.7

figure;sgtitle("detecting SURF features using different number of scale levels per octave")
subplot(1,2,1);imshow(mona_org);hold on;title('NumScaleLevels = 3')
tic
points = detectSURFFeatures(mona_org,'NumScaleLevels', 3);
time = toc;
plot(points);
fprintf('The runtime was %i, We used 3 as the number of scale levels per octave. \n', time);
fprintf('We found %i features \n', length(points));


subplot(1,2,2);imshow(mona_org);hold on;title('NumScaleLevels = 6')
tic
points = detectSURFFeatures(mona_org,'NumScaleLevels', 6);
time = toc;
plot(points);
fprintf('The runtime was %i, We used 6 as the number of octaves. \n', time);
fprintf('We found %i features \n', length(points));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1.2
%% Section 1.2.1
straight_mona = imread([pwd '\straight_mona\straight_mona.PNG']);
straight_mona = double(rgb2gray(straight_mona))/256;

crooked_mona = imread([pwd '\straight_mona\crooked_mona.jpg']);
crooked_mona = double(crooked_mona)/256;


points_straight = detectSURFFeatures(straight_mona);
points_crooked = detectSURFFeatures(crooked_mona);

figure;
subplot(1,2,1);imshow(straight_mona);hold on;plot(points_straight.selectStrongest(10))
subplot(1,2,2);imshow(crooked_mona);hold on;plot(points_crooked.selectStrongest(10))

%some ref
%https://www.mathworks.com/help/vision/ug/find-image-rotation-and-scale-using-automated-feature-matching.html


[featuresOriginal,  validPtsOriginal]  = extractFeatures(straight_mona,  points_straight);
[featuresDistorted, validPtsDistorted] = extractFeatures(crooked_mona, points_crooked);

indexPairs = matchFeatures(featuresOriginal, featuresDistorted);


%calculate the angle between two points and zeros axis
matchedOriginal  = validPtsOriginal(indexPairs(:,1));
matchedDistorted = validPtsDistorted(indexPairs(:,2));

Original_ref = matchedOriginal(1);
Original_nref = matchedOriginal(2:end);
Distorted_ref = matchedDistorted(1);
Distorted_nref = matchedDistorted(2:end);

%https://stackoverflow.com/questions/22200453/calculate-angle-between-horizontal-axis-and-two-points
angle_o = atan2((Original_ref.Location(:,2) - Original_nref.Location(:,2)),(Original_ref.Location(:,1) - Original_nref.Location(:,1)));
angle_d = atan2((Distorted_ref.Location(:,2) - Distorted_nref.Location(:,2)),(Distorted_ref.Location(:,1) - Distorted_nref.Location(:,1)));
avg_rotate = mean(angle_d - angle_o) * 180/pi;


figure; ax = axes;
showMatchedFeatures(straight_mona,crooked_mona,matchedOriginal,matchedDistorted,'montage','Parent',ax);
title('matched features');
legend('ptsOriginal','ptsDistorted');

fixed_mona = imrotate(crooked_mona,avg_rotate);
% Remove zero rows
fixed_mona( all(~fixed_mona,2), : ) = [];
% Remove zero columns
fixed_mona( :, all(~fixed_mona,1) ) = [];

figure;
subplot(1,2,1);imshow(straight_mona);
subplot(1,2,2);imshow(fixed_mona);


%%
%%1.3
mona_org = imread("mona_org.jpg");
mona_org = double(rgb2gray(mona_org))/256;

mona_images = {'Mona1_Y.jpg';'Mona2_Y.jpg';'Mona3_Y.jpg';'Mona4_Y.jpg';'Mona5_N.jpg';'Mona6_N.jpg';'Mona7_Y.jpg';'Mona8_Y.jpg';'Mona9_Y.jpg';'Mona10_Y.jpg';'Mona11_N.jpg';'Mona12_N.jpg'};

for idx = 1:length(mona_images)
    img_name = mona_images{idx};
    mona_img_name = [pwd '\mona\' img_name];
    mona_fake = double(rgb2gray(imread(mona_img_name)))/256;
    [common_f,vpts1,vpts2] = is_mona(mona_org,mona_fake);
    figure; ax = axes;
    showMatchedFeatures(mona_org,mona_fake,vpts1,vpts2,'montage','Parent',ax);
    title(ax, ['features of ' img_name ' | Number of mathced features = ' num2str(common_f) ]);
    if(~isempty(vpts1))
        legend(ax, 'Matched features 1','Matched features 2');
    end
end

%%
%section 2
id1 = '315284729';

figure;
ID2QR(id1);

%%
qr_easy = imread('./1.png');
qr_easy = double(rgb2gray(qr_easy))/256;
% figure;imshow(qr_easy);
% qr_easy_corners = ginput(4);
qr_easy_corners = [407 363;955 305;379 971;919 1061];
[qr1,qr_mat1] = get_qr(qr_easy,qr_easy_corners);
figure;
subplot(1,3,1);imshow(qr_easy);hold on; plot(qr_easy_corners(:,1),qr_easy_corners(:,2),'r*');title('Original Image');
subplot(1,3,2);imshow(qr1);title('QR image After projective transformation');
subplot(1,3,3);imshow(qr_mat1);title('Detected QR Matrix');
sgtitle('Image 1')
fprintf('Restored id from image 1 is: ');
fprintf('%g', get_id(qr_mat1));
fprintf('\n');


qr_medium = imread('./2.png');
qr_medium = double(rgb2gray(qr_medium))/256;
% figure;imshow(qr_medium);
% qr_medium_corners = ginput(4);
qr_medium_corners = [489 391;793 297;471 937;759 1055];
[qr2,qr_mat2] = get_qr(qr_medium,qr_medium_corners);
figure;
subplot(1,3,1);imshow(qr_medium);hold on; plot(qr_medium_corners(:,1),qr_medium_corners(:,2),'r*');title('Original Image');
subplot(1,3,2);imshow(qr2);title('QR image After projective transformation');
subplot(1,3,3);imshow(qr_mat2);title('Detected QR Matrix');
sgtitle('Image 2')
fprintf('Restored id from image 2 is: ');
fprintf('%g', get_id(qr_mat1));
fprintf('\n');


qr_hard = imread('./3.png');
qr_hard = double(rgb2gray(qr_hard))/256;
% figure;imshow(qr_hard);
% qr_hard_corners = ginput(4);
qr_hard_corners = [313 289;495 139;281 911;443 1093];
[qr3,qr_mat3] = get_qr(qr_hard,qr_hard_corners);
figure;
subplot(1,3,1);imshow(qr_hard);hold on; plot(qr_hard_corners(:,1),qr_hard_corners(:,2),'r*');title('Original Image');
subplot(1,3,2);imshow(qr3);title('QR image After projective transformation');
subplot(1,3,3);imshow(qr_mat3);title('Detected QR Matrix');
sgtitle('Image 3')
fprintf('Restored id from image 3 is: ');
fprintf('%g', get_id(qr_mat1));
fprintf('\n');


%%

function [common_f,vpts1_vec,vpts2_vec] = is_mona(mona,img)
common_f = 0;
vpts1_vec = [];
vpts2_vec = [];
for octave_num=1:5
mona_f = detectSURFFeatures(mona,'ROI', [80, 8, 80, 120],'NumOctaves', 6,'NumScaleLevels', 8,'MetricThreshold',500);
img_f = detectSURFFeatures(img,'NumOctaves', 6,'NumScaleLevels', 8,'MetricThreshold',500);
[f1, vpts1]  = extractFeatures(mona,  mona_f);
[f2, vpts2] = extractFeatures(img, img_f);
indexPairs = matchFeatures(f1, f2,"MaxRatio",0.8);
common_f = common_f + length(indexPairs);
if(isempty(vpts1_vec))
    vpts1_vec = vpts1(indexPairs(:,1));
    vpts2_vec = vpts2(indexPairs(:,1));
else
    vpts1_vec = [vpts1_vec ;vpts1(indexPairs(:,1))];
    vpts2_vec = [vpts2_vec ;vpts2(indexPairs(:,1)) ];
end
end
end

function [qr,qr_mat] = get_qr(img,corners)
fixed_points = [1 1; 384 1; 1 384; 384 384];
tform = fitgeotrans(fixed_points, corners, 'projective');
invtform = invert(tform);
qr = imwarp(img, invtform, 'cubic','OutputView', imref2d( size(img) ));
qr = qr(1:384,1:384);
qr_mat = zeros(6,6);
for i = 0:5
    for j = 0:5
        qr_mat(i+1,j+1) = qr(32 + i*64, 32 + j*64) > 0.4;
    end
end
end

function id  = get_id(bits)
id = zeros(1,9);
bits = bits(:);
N = length(bits);

for i=0:(N/4 - 1)
    b = bits(1+4*i:(4*(i+1)));
    num = bin2dec(sprintf('%d',b));
    id(i+1) = num;
end
end