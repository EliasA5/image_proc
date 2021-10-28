close all;
disp('Elias Assaf 315284729 - Jameel Nassar 206985152')

picasso = imread('picasso.jpg');
picasso_grayscale = double(rgb2gray(picasso));
picasso_normalized = dip_GN_imread('picasso.jpg');

figure;
imagesc(picasso_grayscale);
set(gca,'XTick',[], 'YTick', [])
colormap gray;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 1.2

for i = [256,128,32,4]
    [picasso_histo_matlab, bins_matlab] = imhist(dip_GN_imread('picasso.jpg'), i);
    [picasso_histo_func, bins_func] = dip_histogram(dip_GN_imread('picasso.jpg'), i);
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
%make figure and place original image
figure;
sgtitle('Brightness adjustments')
subplot(2,2,1);
imagesc(picasso_normalized);
set(gca,'XTick',[], 'YTick', [])
title('Original image')
colormap gray;
colorbar;
%define the 3 operators and thier parameters and calculate the rest of the
%images and place them in the same figure
action_vec = ["mul", "mul", "add"];
parameter_vec = [1.64, 0.23, 0.41];
for i = 1:3
    adj_img = adjust_brightness(picasso_normalized, action_vec(i), parameter_vec(i));
    subplot(2,2,1+i);
    imagesc(adj_img);
    set(gca,'XTick',[], 'YTick', [])
    title([ 'Action = ' + action_vec(i) + ', Parameter = ' num2str(parameter_vec(i)) ])
    colormap gray;
    colorbar;
    %the next 2 lines force the colorbar to be between 0 and 1 to see the
    %correct colors
    caxis manual
    caxis([0 1]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 1.4

figure;
colormap gray;
colorbar;
sgtitle('Contranst adjustments with different ranges')
colormap gray;
subplot(2,2,1);
imagesc(picasso_normalized)
set(gca,'XTick',[], 'YTick', [])
title('Original image')
caxis manual
caxis([0 1]);

subplot(2,2,2);
adj_img = adjust_contrast(picasso_normalized,0.45,0.9);
imagesc(adj_img);
set(gca,'XTick',[], 'YTick', [])
title('[0.45,0.9]')
caxis manual
caxis([0 1]);

subplot(2,2,3);
adj_img = adjust_contrast(picasso_normalized,0.4,0.5);
imagesc(adj_img);
set(gca,'XTick',[], 'YTick', [])
title('[0.4,0.5]')
caxis manual
caxis([0 1]);

subplot(2,2,4);
adj_img = adjust_contrast(picasso_normalized,1,0);
imagesc(adj_img);
set(gca,'XTick',[], 'YTick', [])
title('[1,0]')
caxis manual
caxis([0 1]);



figure;
sgtitle('Non linear mapping with clipping')
colormap gray;
subplot(1,2,1);
adj_img = picasso_normalized;
adj_img((adj_img < 0.45)) = 0.45;
adj_img((adj_img > 0.9)) = 0.9;
imagesc(adj_img);
set(gca,'XTick',[], 'YTick', [])
title('[0.45,0.9]')
colorbar;
caxis manual
caxis([0 1]);

subplot(1,2,2);
adj_img = picasso_normalized;
adj_img((adj_img < 0.2)) = 0.2;
adj_img((adj_img > 0.8)) = 0.8;
imagesc(adj_img);
set(gca,'XTick',[], 'YTick', [])
title('[0.2,0.8]')
colorbar;
caxis manual
caxis([0 1]);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 1.5

bits_vec = [6,4,2,1];
figure;
sgtitle('Quantizing image using different bits')
colormap gray;
set(gca,'XTick',[], 'YTick', [])
for i = 1:4
    quantized_img = quantize_img(picasso_normalized, bits_vec(i));
    subplot(2,2,i);
    imagesc(quantized_img);
    set(gca,'XTick',[], 'YTick', [])
    title([num2str(bits_vec(i)) ' bit'])
    colorbar;
    caxis manual
    caxis([0 1]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 1.6

dog_normalized = dip_GN_imread('dog.jpg');
dog_equalized = histeq(dog_normalized);
[dog_histo, dog_bins] = dip_histogram(dog_normalized, 256);
[dog_histo_eq, dog_bins_eq] = dip_histogram(dog_equalized, 256);
figure;
colormap gray;
subplot(2,2,1);
imagesc(dog_normalized);
title('Grayscale Dog Image')
set(gca,'XTick',[], 'YTick', [])
colorbar;

subplot(2,2,2);
stem(dog_histo, dog_bins, 'k|')
title('Dog Image Histogram')

subplot(2,2,3);
imagesc(dog_equalized);
set(gca,'XTick',[], 'YTick', [])
title('Equalized Dog Image')
colorbar;
caxis manual
caxis([0 1]);

subplot(2,2,4);
stem(dog_histo_eq, dog_bins_eq, 'k|')
title('Equalized Dog Image Histogram')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 1.7
filfil_normalized = dip_GN_imread('filfil.jpg');
[filfil_histo, filfil_bins] = dip_histogram(filfil_normalized, 256);
city_normalized = dip_GN_imread('city.jpg');
[city_histo, city_bins] = dip_histogram(city_normalized, 256);
face = double(imread('face.jpg'));
face_normalized = (face - min(face(:)))./ (max(face(:)) - min(face(:)));
[face_histo, face_bins] = dip_histogram(face_normalized, 256);
figure;
colormap gray;

subplot(3,2,1);
imagesc(filfil_normalized);
set(gca,'XTick',[], 'YTick', [])
title('Filfil Image')
colorbar;
subplot(3,2,2);
stem(filfil_histo, filfil_bins, 'k|')
title('Filfil Histogram')

subplot(3,2,3);
imagesc(city_normalized);
set(gca,'XTick',[], 'YTick', [])
title('City Image')
colorbar;
subplot(3,2,4);
stem(city_histo, city_bins, 'k|')
title('City Histogram')

subplot(3,2,5);
imagesc(face_normalized);
set(gca,'XTick',[], 'YTick', [])
title('Face Image')
colorbar;
subplot(3,2,6);
stem(face_histo, face_bins, 'k|')
title('Face Histogram')


filfil_city = imhistmatch(filfil_normalized, city_normalized);
[filfil_city_histo, filfil_city_bins] = dip_histogram(filfil_city, 256);
filfil_face = imhistmatch(filfil_normalized, face_normalized);
[filfil_face_histo, filfil_face_bins] = dip_histogram(filfil_face, 256);

figure;
colormap gray;

subplot(2,2,1);
imagesc(filfil_city);
set(gca,'XTick',[], 'YTick', [])
title('Filfil City Combined')
colorbar;
subplot(2,2,2);
stem(filfil_city_histo, filfil_city_bins, 'k|')
title('Filfil City Histogram')

subplot(2,2,3);
imagesc(filfil_face);
set(gca,'XTick',[], 'YTick', [])
title('Filfil Face Combined')
colorbar;
subplot(2,2,4);
stem(filfil_face_histo, filfil_face_bins, 'k|')
title('Filfil Face Histogram')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% 2.2
dog_normalized = dip_GN_imread('dog.jpg');



function [gray_img_norm] = dip_GN_imread(file_name)
    img = imread(file_name);
    img_gray = double(rgb2gray(img));
    gray_img_norm = (img_gray - min(img_gray(:)))./ (max(img_gray(:)) - min(img_gray(:)));
end


function [counts, bins] = dip_histogram(img,nbins)
    bins = (0:(nbins-1))/(nbins-1);
    counts = zeros(nbins,1);
    for i = (1:nbins)
        low_b = (i-1.5)/(nbins-1);
        high_b = (i-0.5)/(nbins-1);
        counts(i) = sum((img(:) >= low_b)&(img(:) < high_b));
    end

end

function [adj_img] = adjust_brightness(img, action, parameter)
    if action == "add"
        func = @(x,y) x + y;
    elseif action == "mul"
        func = @(x,y) x * y;
    else
        error('Not a valid operator')
    end
    adj_img = func(img, parameter);
end


function [adj_img] = adjust_contrast(img,range_low,range_high)
    adj_img = img *(range_high-range_low) + range_low;
end

function [quantized_img] = quantize_img(img,bits)
    quantized_img = (((img * 255 + 1) ./ 2^(8 - bits)) - 1) / 255;
    levels = 2^(bits);
    levesl_vec = (0:(levels))/(levels);
    val_vec = (0:(levels-1))/(levels-1);
    for i = (1:levels)
        low_b = levesl_vec(i);
        high_b = levesl_vec(i+1);
        quantized_img((img(:) >= low_b)&(img(:) < high_b)) = val_vec(i);
    end
end

function [filtered_img] = filter(img, k, filt)
    to_extend = floor(k/2);
    filtered_img_y = [img(1:to_extend, :); img ; img(end-to_extend+1:end, :)];
    filtered_img_x = [filtered_img_y(:, 1:to_extend), filtered_img_y , filtered_img_y(:, end-to_extend+1:end)];
    filtered_img = zeros(size(filtered_img_x));
    for m=(1:size(img, 1))+to_extend
        for n=(1:size(img, 2))+to_extend
                filtered_img(m,n) = sum(filt .* filtered_img_x((m:m+k-1)-to_extend, (n:n+k-1)-to_extend));
        end
    end
    filtered_img = filtered_img( (1:size(img, 1))+to_extend, (1:size(img, 2))+to_extend);
end

function [filtered_img] = mean_filter(img,k) %assume k is odd
    if mod(k,2) == 0
        error('filter size must be an odd number');
    end
    filt = 1/k^2 * ones(k);
    filtered_img = filter(img, k, filt);
end

function [filtered_img] = median_filter(img, k)
    to_extend = floor(k/2);
    filtered_img_y = [img(1:to_extend, :); img ; img(end-to_extend+1:end, :)];
    filtered_img_x = [filtered_img_y(:, 1:to_extend), filtered_img_y , filtered_img_y(:, end-to_extend+1:end)];
    filtered_img = zeros(size(filtered_img_x));
    for m=(1:size(img, 1))+to_extend
        for n=(1:size(img, 2))+to_extend
                filtered_img(m,n) = median(filtered_img_x((m:m+k-1)-to_extend, (n:n+k-1)-to_extend), 'all');
        end
    end
    filtered_img = filtered_img( (1:size(img, 1))+to_extend, (1:size(img, 2))+to_extend);
end