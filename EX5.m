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


figure;
showMatchedFeatures(straight_mona,crooked_mona,matchedOriginal,matchedDistorted);
title('Putatively matched points (including outliers)');
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
    len = its_mona(mona_org,mona_fake);
    fprintf('We found %i similar features points in %s \n', len,img_name);
end

%%
%section 2
id1 = '315284729';

figure;
ID2QR(id1);

%%
qr_easy = imread('./11.jpg');
qr_easy = double(rgb2gray(qr_easy))/256;
figure;imshow(qr_easy);
%qr_easy_corners = ginput(4);
%qr_easy_corners = [3638 2598;2309 2732;3700 1140;2394 921];
qr_easy_corners = [554 462;1867 336;485 1938;1786 2151];


qr_medium = imread('./21.jpg');
qr_medium = double(rgb2gray(qr_medium))/256;
figure;imshow(qr_medium);
%qr_medium_corners = ginput(4);
%qr_medium_corners = [3433 2522;2701 2745;3482 1216;2773 935];
qr_medium_corners = [490 497;1214 266;442 1811;1130 2090];


qr_hard = imread('./31.jpg');
qr_hard = double(rgb2gray(qr_hard))/256;
figure;imshow(qr_hard);
%qr_hard_corners = ginput(4);
%qr_hard_corners = [3861 2772;3433 3129;3937 1278;3562 859];
qr_hard_corners = [431 492;863 117;362 1983;728 2403];


%%
fixed_points = [1 1; 256 1; 1 256; 256 256];
tform = fitgeotrans(fixed_points, qr_easy_corners, 'projective');
invtform = invert(tform);
out = imwarp(qr_easy, invtform, 'linear');

figure;imshow(out);colormap gray;

tform = fitgeotrans(fixed_points, qr_medium_corners, 'projective');
invtform = invert(tform);
out = imwarp(qr_medium, invtform, 'linear');

figure;imshow(out);colormap gray;

tform = fitgeotrans(fixed_points, qr_hard_corners, 'projective');
invtform = invert(tform);
out = imwarp(qr_hard, invtform, 'linear');

figure;imshow(out);colormap gray;

%%

function same = its_mona(mona,img)

same = 0;

for num_octave=1:5
mona_f = detectSURFFeatures(mona,'ROI', [80, 8, 80, 120],'NumOctaves', num_octave,'NumScaleLevels', 10,'MetricThreshold',300);
img_f = detectSURFFeatures(img,'NumOctaves', num_octave,'NumScaleLevels', 10,'MetricThreshold',300);
[featuresOriginal,  ~]  = extractFeatures(mona,  mona_f);
[featuresDistorted, ~] = extractFeatures(img, img_f);
indexPairs = matchFeatures(featuresOriginal, featuresDistorted);
same = same + length(indexPairs);
end

end