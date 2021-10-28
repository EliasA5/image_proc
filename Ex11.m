close all;
disp('Elias Assaf 315284729 - Jameel Nassar 206985152')

picasso = imread('picasso.jpg');
<<<<<<< HEAD
picasso_grayscale = rgb2gray(picasso);

=======
picasso_grayscale = double(rgb2gray(picasso));
>>>>>>> e0c9c4d3a909f53b08ac2e7e980b2989185f798c

figure;
imagesc(picasso_grayscale);
colormap gray;

for i = [256,128,32,4]
<<<<<<< HEAD
    [picasso_histo_matlab, bins_matlab] = imhist(picasso_grayscale, i);
    [picasso_histo_func, bins_func] = dip_histogram(picasso_grayscale, i);
=======
    [picasso_histo_matlab, bins_matlab] = imhist(dip_GN_imread('picasso.jpg'), i);
    [picasso_histo_func, bins_func] = dip_histogram(dip_GN_imread('picasso.jpg'), i);
>>>>>>> e0c9c4d3a909f53b08ac2e7e980b2989185f798c
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



<<<<<<< HEAD
=======


>>>>>>> e0c9c4d3a909f53b08ac2e7e980b2989185f798c
function [gray_img_norm] = dip_GN_imread(file_name)
    img = imread(file_name);
    img_gray = double(rgb2gray(img));
    gray_img_norm = (img_gray - min(img_gray(:)))./ (max(img_gray(:) - min(img_gray(:))));
end

<<<<<<< HEAD
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
=======

function [counts, bins] = dip_histogram(img,nbins)
    bins = (1:nbins)/nbins;
    counts = zeros(nbins,1);
    for pixel = img(:).'
        counts(floor(pixel*(nbins-1))+1) = counts(floor(pixel*(nbins-1))+1) + 1;
    end

>>>>>>> e0c9c4d3a909f53b08ac2e7e980b2989185f798c
end