close all
clear vars
clc
disp('Elias Assaf 315284729 - Jameel Nassar 206985152')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1
img = double(imread("image.png"));
[rgb_img, r_img, g_img, b_img] = dip_normalize_rgb_img(img);
figure;imshow(rgb_img);title('Original Image');
figure;colorbar;sgtitle('Each color channel');
subplot(1,3,1);imshow(r_img);title('Red Channel');colorbar;
subplot(1,3,2);imshow(g_img);title('Green Channel');colorbar;
subplot(1,3,3);imshow(b_img);title('Blue Channel');colorbar;

figure;sgtitle('RGB to Gray Comaprison');
subplot(1,2,1);imshow(dip_rgb2gray(rgb_img));title('Our function');colorbar;
subplot(1,2,2);imshow(rgb2gray(rgb_img));title('Matlab built in function');colorbar;


%TODO: 1.6

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 2
[c, y, m, k] = dip_rgb_to_cymk(r_img, g_img, b_img);
figure;sgtitle('CYMK channels');
subplot(2,2,1);imshow(c);title('Cyan');colorbar;
subplot(2,2,2);imshow(y);title('Yellow');colorbar;
subplot(2,2,3);imshow(m);title('Magenta');colorbar;
subplot(2,2,4);imshow(k);title('Black');colorbar;

figure;title('CYMK image');
imshowCYMK(c,y,m,k);


%TODO: 2.6

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 3
[h, s, v] = dip_rgb2hsv(r_img, g_img, b_img);

figure;sgtitle('HSV channels our function')
subplot(1,3,1);imshow(h);title('Hue');colormap('hsv');colorbar;
subplot(1,3,2);imshow(s);title('Saturation');colorbar;
subplot(1,3,3);imshow(v);title('Value');colorbar;

hsv_img = rgb2hsv(rgb_img);
figure;sgtitle('HSV channels matlab function')
subplot(1,3,1);imshow(hsv_img(:,:,1));title('Hue');colormap('hsv');colorbar;
subplot(1,3,2);imshow(hsv_img(:,:,2));title('Saturation');colorbar;
subplot(1,3,3);imshow(hsv_img(:,:,3));title('Value');colorbar;

%TODO: 3.6, 3.7

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 4
lab_img = rgb2lab(rgb_img);
figure;sgtitle('L*a*b channels')
subplot(1,3,1);imshow(lab_img(:,:,1));title('L');colorbar;
subplot(1,3,2);imshow(lab_img(:,:,2));title('a');colorbar;
subplot(1,3,3);imshow(lab_img(:,:,3));title('b');colorbar;

%TODO: 4.6

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 5
cap1 = double(imread("cap2.png"));
cap1_norm = dip_normalize_rgb_img(cap1);
[cap1_circled, cap1_filter] = dip_find_cap(cap1_norm);
figure;sgtitle('Cap 1')
subplot(1,2,1);title('Circled');imshow(cap1_circled);
subplot(1,2,2);title('Filter');imshow(cap1_filter);

cap2 = double(imread("cap2.png"));
cap2_norm = dip_normalize_rgb_img(cap2);
[cap2_circled, cap2_filter] = dip_find_cap(cap2_norm);
figure;sgtitle('Cap 2')
subplot(1,2,1);title('Circled');imshow(cap2_circled);
subplot(1,2,2);title('Filter');imshow(cap2_filter);

cap3 = double(imread("cap3.png"));
cap3_norm = dip_normalize_rgb_img(cap3);
[cap3_circled, cap3_filter] = dip_find_cap(cap3_norm);
figure;sgtitle('Cap 3')
subplot(1,2,1);title('Circled');imshow(cap3_circled);
subplot(1,2,2);title('Filter');imshow(cap3_filter);

%TODO: section 5

function [cap_img, filter] = dip_find_cap(img)
    [cap_h, cap_s, cap_v] = dip_rgb2hsv(img(:,:,1), img(:,:,2), img(:,:,3));
    cap_h_filt = zeros(size(cap_h));
    cap_s_filt = zeros(size(cap_s));
    cap_v_filt = zeros(size(cap_v));
    cap_h_filt((cap_h > 0.6) & (cap_h < 0.7)) = 1;
    cap_s_filt((cap_s > 0.40) & (cap_s < 0.88)) = 1;
    cap_v_filt((cap_v > 0.07) & (cap_v < 0.30)) = 1;
    
    filter = cap_h_filt .* cap_s_filt .* cap_v_filt;
    filter = medfilt2(filter, [7 7]);
    
    [x,y] = meshgrid(1:size(filter,2), 1:size(filter,1));
    min_x = min(x(filter == 1));
    max_x = max(x(filter == 1));
    min_y = min(y(filter == 1));
    max_y = max(y(filter == 1));
    
    circle_x = floor((min_x+max_x)/2);
    circle_y = floor((min_y+max_y)/2);
    
    cap_img = insertShape(img, 'circle', [circle_x circle_y (max_x-circle_x + 3)]);
end

function [h, s, v] = dip_rgb2hsv(r, g, b)
    %according to the equations stated in:
    % https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    [C_max, I_max] = max(cat(3, r, g, b), [], 3);
    C_min = min(cat(3, r, g, b), [], 3);
    delta = C_max - C_min;
    mask = (C_max ~= 0);
    s = zeros(size(C_max));
    s(mask) = delta(mask) ./ C_max(mask);
    s(isnan(s)) = 0;

    h = zeros(size(C_max));
    r_mask = (I_max == 1);
    h(r_mask) = mod( (g(r_mask)-b(r_mask)) ./ (delta(r_mask)), 6);
    g_mask = (I_max == 2);
    h(g_mask) =  2 + ((b(g_mask)-r(g_mask)) ./ (delta(g_mask)));
    b_mask = (I_max == 3);
    h(b_mask) =  4 + ((r(b_mask)-g(b_mask)) ./ (delta(b_mask)));
    h(isnan(h)) = 0;
    h = h/6;

    v = C_max;
end

function [c, y, m, k] = dip_rgb_to_cymk(r, g, b)
    k = min(cat(3, 1-r, 1-g, 1-b), [], 3);
    c = (1-r-b)./(1-b);
    m = (1-g-b)./(1-b);
    y = (1-b-k)./(1-k);
end

function [norm_img, r, g, b] = dip_normalize_rgb_img(rgb_img)
    r = dip_normalize_img(rgb_img(:,:,1));
    g = dip_normalize_img(rgb_img(:,:,2));
    b = dip_normalize_img(rgb_img(:,:,3));
    norm_img = cat(3, r, g, b);
end

function [norm_img] = dip_normalize_img(img)
    %normalize according to the equation given in the ex1 pdf
    norm_img = (img - min(img(:)))./ (max(img(:)) - min(img(:)));
end


function [gray_img] = dip_rgb2gray(img)
    gray_img = 0.2989 * img(:,:,1) + 0.5870 * img(:,:,2) + 0.1140 * img(:,:,3);
end