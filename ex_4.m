close all
clc
disp('Elias Assaf 315284729 - Jameel Nassar 206985152')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1.1
cameraman = imread("cameraman.tif");
cameraman = double(cameraman)/256;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1.2
figure;sgtitle("Cameraman filtered with prewitt edge detector with different thresh hold");
cameraman_prewitt_mag_1 = dip_prewitt_edge(cameraman, 0.3);
cameraman_prewitt_mag_2 = dip_prewitt_edge(cameraman, 0.45);
subplot(1,2,1);imshow(cameraman_prewitt_mag_1);title("thresh hold = 0.3");
subplot(1,2,2);imshow(cameraman_prewitt_mag_2);title("thresh hold = 0.45");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1.3

figure;sgtitle("Canny edge detector")
[cameraman_canny_default, cameraman_canny_default_threshhold] = edge(cameraman, 'Canny');
canny_thresh = [0.3 0.42];
cameraman_canny_custom = edge(cameraman, 'Canny', canny_thresh);

subplot(1,2,1);imshow(cameraman_canny_default);title(["default thresh hold = " num2str(cameraman_canny_default_threshhold)]);
subplot(1,2,2);imshow(cameraman_canny_custom);title(["custom thresh hold = " num2str(canny_thresh)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 2.1

floor = imread("floor.jpg");
floor = rgb2gray(floor)/256;
%default filter is sobel which we learned early in the course
BW_floor = edge(floor);

[hough_mat_1_1,R_vec1,theta_vec1] = dip_hough_lines(BW_floor,1,1);
[hough_mat_5_4,R_vec2,theta_vec2] = dip_hough_lines(BW_floor,5,4);
figure;sgtitle('Hough transform of floor');
subplot(1,2,1);imshow(imadjust(rescale(hough_mat_1_1)),'XData',theta_vec1,'YData',R_vec1, 'InitialMagnification','fit');
title("R_0 = 1, \theta_0 = 1");
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
colormap(gca,hot);


subplot(1,2,2);
imshow(imadjust(rescale(hough_mat_5_4)),'XData',theta_vec2,'YData',R_vec2,  'InitialMagnification','fit');
title("R_0 = 5, \theta_0 = 4");xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on; colormap(gca,hot);

peaks_1_1 = houghpeaks(hough_mat_1_1, 4);
peaks_5_4 = houghpeaks(hough_mat_5_4, 4);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 2.2

coffee = imread("coffee.jpg");
coffee = double(rgb2gray(coffee))/256;

BW_coffee = edge(coffee);

tic;[hough_mat_circ_1_1, R_circ_vec1, theta_circ_vec1] = dip_hough_circles(BW_coffee, 1, 1);toc;
[hough_mat_circ_4_10, R_circ_vec2, theta_circ_vec2] = dip_hough_circles(BW_coffee, 4, 10);
hough_circ_peaks_1_1 = dip_houghpeaks3d(hough_mat_circ_1_1);
hough_circ_peaks_4_10 = dip_houghpeaks3d(hough_mat_circ_4_10);

figure;imshow(hough_mat_circ_1_1(:,:, 4), []);
figure;imshow(coffee);hold on;
viscircles(hough_circ_peaks_1_1(:, 1:2), R_circ_vec1(hough_circ_peaks_1_1(:, 3)),'EdgeColor','b');

figure;imshow(coffee);hold on;
viscircles(hough_circ_peaks_4_10(:, 1:2), R_circ_vec2(hough_circ_peaks_4_10(:, 3)),'EdgeColor','b');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Functions
function [magnitude] = dip_prewitt_edge(img, thresh)
    filt_x = 1/6 * repmat([-1 0 1], 3, 1);
    filt_y = rot90(filt_x);
    Gx = conv2(img, filt_x, 'same');
    Gy = conv2(img, filt_y, 'same');
    magnitude = sqrt(Gx.^2 + Gy.^2);
    magnitude = (magnitude >= thresh);
end

function [hough_mat, R_vec, theta_vec] = dip_hough_lines(BW, R_0, theta_0)
    
    [M,N] = size(BW);
    R_value = round(sqrt(M^2 + N^2));
    R_vec = -R_value:R_0:R_value;
    theta_vec = -90:theta_0:90;
    hough_mat = zeros(length(R_vec),length(theta_vec));
    [x_vec, y_vec] = find(BW == 1);
    x_vec = x_vec.';
    y_vec = y_vec.';

    for i = 1:length(theta_vec)
        theta = theta_vec(i) * (pi / 180);
        r = cos(theta) .* y_vec + sin(theta) .* x_vec;
        r = interp1(R_vec, R_vec, r, 'nearest');
        r = round((r+R_value + 1)/R_0);
        for j = r
            hough_mat(j, i) = hough_mat(j, i) + 1;
        end
    end

end

function [hough_mat, R_vec, theta_vec] = dip_hough_circles(BW, R_0, theta_0)

    [M,N] = size(BW);
    R_vec = 80:R_0:100;
    theta_vec = 0:theta_0:360-theta_0;
    hough_mat = zeros(M, N, length(R_vec));
    [x_vec, y_vec] = find(BW == 1);
    x_vec = x_vec.';
    y_vec = y_vec.';
    for k = 1:length(R_vec)
        r = R_vec(k);
        for i = 1:length(theta_vec)
            theta = theta_vec(i) * (pi / 180);
            a = abs(round(x_vec - r*cos(theta)));
            b = abs(round(y_vec - r*sin(theta)));
            a_w = a(a>0 & b>0 & a<=M & b<=N);
            b_w = b(a>0 & b>0 & a<=M & b<=N);
            for j=1:length(a_w)
                hough_mat(a_w(j), b_w(j), k) = hough_mat(a_w(j), b_w(j), k) + 1;
            end
        end
    end

end

function [peaks] = dip_houghpeaks3d(H)
    peaks = dip_houghpeaks3d_custom(H, 5);
end

function [peaks] = dip_houghpeaks3d_custom(H, numOfPeaks)
    peaks = zeros(numOfPeaks, 3);
    for i = 1:numOfPeaks
        [val, idx] = max(H(:));
        [idx1, idx2, idx3] = ind2sub(size(H), idx);
        peaks(i, :) = [idx2, idx1, idx3];
        H(idx1, idx2, idx3) = 0;
    end
end
