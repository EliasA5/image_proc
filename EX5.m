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

matched_f = find_matching_features(points_straight,points_crooked);

%figure;
%subplot(1,2,1);imshow(straight_mona);hold on;plot(points_straight(matched_f(:,1)));
%subplot(1,2,2);imshow(crooked_mona);hold on;plot(points_crooked(matched_f(:,2)));



[featuresOriginal,  validPtsOriginal]  = extractFeatures(straight_mona,  points_straight);
[featuresDistorted, validPtsDistorted] = extractFeatures(crooked_mona, points_crooked);

indexPairs = matchFeatures(featuresOriginal, featuresDistorted);

matchedOriginal  = validPtsOriginal(indexPairs(:,1));
matchedDistorted = validPtsDistorted(indexPairs(:,2));








function matched = find_matching_features(features1,features2)
%find the common features based on some checks on the featues
idx = 0;
N1 = length(features1);
N2 = length(features2);
matched = zeros(N1*N2,2);
for i = 1:N1
    f1 = features1(i);
    for j = 1:N2
        f2 = features2(j);
        
        scale_check = (abs(f1.Scale - f2.Scale)  < 1);
        sign_check = f1.SignOfLaplacian == f2.SignOfLaplacian;
        metric_check = (abs(f1.Metric - f2.Metric)  < 00);
        if(scale_check && sign_check && metric_check)
            idx = idx + 1;
            matched(idx,:) = [i j];%add the matched index

        end

    end
end
non_zero = all(matched ~= [0 0],2);
matched = matched(non_zero,:);%get the non zeros
end

