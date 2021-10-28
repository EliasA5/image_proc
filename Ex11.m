close all;
disp('Elias Assaf 315284729 - Jameel Nassar 206985152')

picasso = imread('picasso.jpg');
picasso_grayscale = rgb2gray(picasso);
picasso_normalized = dip_GN_imread('picasso.jpg');

figure;
imagesc(picasso_grayscale);
colormap gray;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 1.2

for i = [256,128,32,4]
    [picasso_histo_matlab, bins_matlab] = imhist(picasso_grayscale, i);
    [picasso_histo_func, bins_func] = dip_histogram(picasso_grayscale, i);
    figure;
    hold on;
    sgtitle(['calculating image histograms with ' int2str(i) ' bins'])
    subplot(1,2,1);
    stem(bins_func, picasso_histo_func, 'k|')
    title('Our Function')
    subplot(1,2,2);
    stem(bins_matlab,picasso_histo_matlab, 'k|')
    title('Built in Function')
    hold off;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 1.3
figure;
subplot(2,2,1);
imagesc(picasso_grayscale);
colormap gray;
for i = 1:3
    action_vec = ["mul","mul","add"];
    parameter_vec = [2,0.6,200];
    adj_img = adjust_brightness(picasso_grayscale,action_vec(i),parameter_vec(i));
    subplot(2,2,1+i);
    imagesc(adj_img);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 1.4

figure;
colormap gray;
subplot(1,3,1);
adj_img = adjust_contrast(picasso_normalized,0.45,0.9);
imagesc(adj_img);
subplot(1,3,2);
adj_img = adjust_contrast(picasso_normalized,0.4,0.5);
imagesc(adj_img);
subplot(1,3,3);
adj_img = adjust_contrast(picasso_normalized,1,0);
imagesc(adj_img);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [gray_img_norm] = dip_GN_imread(file_name)
    img = imread(file_name);
    img_gray = double(rgb2gray(img));
    gray_img_norm = (img_gray - min(img_gray(:)))./ (max(img_gray(:) - min(img_gray(:))));
end


function [counts, bins] = dip_histogram(img,nbins)
    img = double(img);
    bins = (0:(nbins-1))*(255/(nbins-1));
    counts = zeros(nbins,1);
    bound_vec = (0:nbins)*(255/(nbins-1));
    for i = (1:nbins)
        low_b = bound_vec(i);
        high_b = bound_vec(i+1);
        counts(i) = sum((img(:) >= low_b)&(img(:) < high_b));
    end
end

function [adj_img] = adjust_brightness(img,action,parameter)
    if action == "mul"
        adj_img = img .* parameter ;
        adj_img = (adj_img - min(adj_img(:)))./ (max(adj_img(:) - min(adj_img(:))));
    elseif action == "add"
        adj_img = img + parameter ;
        adj_img = (adj_img - min(adj_img(:)))./ (max(adj_img(:) - min(adj_img(:))));
    else
        adj_img = img;
    end
end


function [adj_img] = adjust_contrast(img,range_low,range_high)
    adj_img = img;
    adj_img((img < range_low)) = range_low;
    adj_img((img > range_high)) = range_high;
end