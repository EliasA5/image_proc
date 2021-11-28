close all
clc
disp('Elias Assaf 315284729 - Jameel Nassar 206985152')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1.1

beatles = imread("beatles.png");
beatles_norm = double(rgb2gray(beatles))/255;
beatles_fft2 = dip_fft2(beatles_norm);
beatles_fft2_matlab = fft2(beatles_norm);
figure;sgtitle("fft2 of beatles")
subplot(2,2,1);imagesc(log10(abs(dip_fftshift(beatles_fft2))));title("our version Amplitude");colorbar;
subplot(2,2,2);imagesc(angle(dip_fftshift(beatles_fft2)));title("phase");colorbar;
subplot(2,2,3);imagesc(log10(abs(dip_fftshift(beatles_fft2_matlab))));title("matlab version Amplitude");colorbar;
subplot(2,2,4);imagesc(angle(dip_fftshift(beatles_fft2_matlab)));title("phase");colorbar;


beatles_ifft2 = dip_ifft2(beatles_fft2);
beatles_ifft2_matlab = ifft2(beatles_fft2_matlab);
figure;sgtitle("ifft2 of beatles")
subplot(2,2,1);imshow(beatles_norm);title("original image");
subplot(2,2,2);imshow(real(beatles_ifft2));title("our ifft");
subplot(2,2,4);imshow(real(beatles_ifft2_matlab));title("matlab ifft");


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1.2.1
load("freewilly.mat");
imprisoned_willy = freewilly;
figure;imshow(imprisoned_willy);title("imprisoned willy");
[M,N] = size(imprisoned_willy);
[X_grid, Y_grid] = meshgrid(0:N-1, 0:M-1);
prison = 0.5 * sin(2*pi*10/N * X_grid);
figure;imshow(prison);title("prison")
%freed_willy = imprisoned_willy - prison;
%figure;imshow(freed_willy);title("freed willy by -");
prison_fft = fft2(prison);
figure;imshow(abs(dip_fftshift(prison_fft)));title("prison fft2")
Free_Willy(imprisoned_willy);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1.2.2


square_1 = zeros(128, 128);
[square_1_X, square_1_y] = meshgrid(44:83, 44:83);
idx_1 = sub2ind(size(square), square_1_X, square_1_y);
square(idx_1) = 1;
square_fft = fft2(square);
figure;
subplot(1,2,1);imshow(square);title("square 1")
subplot(1,2,2);imshow(abs(dip_fftshift(square_fft)));title("square1 fft2")


square_2 = zeros(128, 128);
[square_2_X, square_2_y] = meshgrid(64:103, 64:103);
idx_2 = sub2ind(size(square_2), square_2_X, square_2_y);
square_2(idx_2) = 1;
square_2_fft = fft2(square_2);
figure;
subplot(1,2,1);imshow(square_2);title("square 2");
subplot(1,2,2);imshow(abs(dip_fftshift(square_2_fft)));title("square 2 fft2");
figure;
subplot(1,2,1);imshow(abs(dip_fftshift(square_fft)));title("square1 fft2")
subplot(1,2,2);imshow(abs(dip_fftshift(square_2_fft)));title("square2 fft2");
sum(square_fft == square_2_fft, 'all')


square_3 = zeros(128, 128);
[square_3_X, square_3_y] = meshgrid(24:103, 54:73);
idx_3 = sub2ind(size(square_3), square_3_X, square_3_y);
square_3(idx_3) = 1;
square_3_fft = fft2(square_3);
figure;
subplot(1,2,1);imshow(square_3);title("square 3");
subplot(1,2,2);imshow(abs(dip_fftshift(square_3_fft)));title("square 3 fft2");

figure;
subplot(1,3,1);imshow(abs(dip_fftshift(square_fft)));title("square1 fft2")
subplot(1,3,2);imshow(abs(dip_fftshift(square_2_fft)));title("square2 fft2");
subplot(1,3,3);imshow(abs(dip_fftshift(square_3_fft)));title("square 3 fft2");

vec_80_1 = [zeros(24,1); ones(80,1); zeros(24,1)];
vec_1_20 = [zeros(1,54), ones(1,20), zeros(1,54)];
square_80_20 = vec_80_1 * vec_1_20;
figure;sgtitle("the 2 vectors");
subplot(1,2,1);imshow(vec_80_1);title('80x1 padded to 128x1');
subplot(1,2,2);imshow(vec_1_20);title('1x20 padded to 1x128');

figure;imshow(square_80_20);title("80x20 square by multiplying vectors");

square_3_sep_fft = sep_fft2(vec_80_1, vec_1_20);
figure;
subplot(1,2,1);imshow(abs(dip_fftshift(square_3_fft)));title("square3 fft2")
subplot(1,2,2);imshow(abs(dip_fftshift(square_3_sep_fft)));title("square3 fft2 by separating");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 2
beatles = imread("beatles.png");
beatles_norm = double(rgb2gray(beatles))/255;
beatles_dct2 = dct2(beatles_norm);
[M_beatles, N_beatles] = size(beatles_norm);
figure;imagesc(log(abs(beatles_dct2)));colormap(jet(64));colorbar;title('dct of beatles image')
half_ones_rand = [ones(1,M_beatles*N_beatles/2) zeros(1,M_beatles*N_beatles/2)];
for i = 1:7 %shuffle few times
    half_ones_rand = half_ones_rand(randperm(length(half_ones_rand))); 
end
half_ones_rand = reshape(half_ones_rand, [M_beatles, N_beatles]);
beatles_dct2_half_rand = half_ones_rand .* beatles_dct2;
beatles_idct2_half_rand = idct2(beatles_dct2_half_rand);
figure;sgtitle("half values removed randomly from dct domain");
subplot(1,2,1);imshow(beatles_norm);title("original image");
subplot(1,2,2);imshow(beatles_idct2_half_rand);title("50% values removed randomly");

beatles_dct2_half_lowest = beatles_dct2;
beatles_dct2_abs_median = median(abs(beatles_dct2_half_lowest), 'all');
beatles_dct2_half_lowest(beatles_dct2_half_lowest < beatles_dct2_abs_median & beatles_dct2_half_lowest > -beatles_dct2_abs_median) = 0;
beatles_idct2_half_lowest = idct2(beatles_dct2_half_lowest);
figure;sgtitle("half absolute lowest values removed from dct domain (values lower than the median)");
subplot(1,2,1);imshow(beatles_norm);title("original image")
subplot(1,2,2);imshow(beatles_idct2_half_lowest);title("50% lowest values removed")

a = 0.3;
figure;sgtitle("values from (-a,a) removed from dct domain");
subplot(2,3,1);imshow(beatles_norm);title('original image');
i=2;
for a = [0.02 0.05 0.07 0.09 0.15]
    beatles_dct2_no_a = beatles_dct2;
    beatles_dct2_no_a(beatles_dct2_no_a < a & beatles_dct2_no_a > -a) = 0;
    beatles_idct2_no_a = idct2(beatles_dct2_no_a);
    perc = sum(beatles_dct2_no_a == 0, 'all') / numel(beatles_dct2);
    subplot(2,3,i);imshow(beatles_idct2_no_a);title(['a = ' num2str(a) ', percentage = ' num2str(perc)])
    i = i+1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 3

beetle = imread("beetle.jpg");
beetle_norm = double(rgb2gray(beetle))/255;
n=5;
[c_beetle,s_beetle]=wavedec2(beetle_norm,n,'haar');


for i = 1:n
[h_beetle,v_beetle,detail_beetle] = detcoef2('all',c_beetle,s_beetle,i);
cfs2 = appcoef2(c_beetle,s_beetle,'haar',i);
V_img = wcodemat(v_beetle,255,'mat',i);
H_img = wcodemat(h_beetle,255,'mat',i);
D_img = wcodemat(detail_beetle,255,'mat',i);
A_img = wcodemat(cfs2,255,'mat',i);
figure;colormap('gray');
subplot(2,2,1);imagesc(A_img);title(['Approximation Coef. of Level '  int2str(i)]);
subplot(2,2,2);imagesc(H_img);title(['Horizontal Detail Coef. of Level '  int2str(i)]);
subplot(2,2,3);imagesc(V_img);title(['Vertical Detail Coef. of Level '  int2str(i)]);
subplot(2,2,4);imagesc(D_img);title(['Diagonal Detail Coef. of Level ' int2str(i)]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% functions
function [img_fft2] = dip_fft2(I)
    [M,N] = size(I);
    M_vec = (0:M-1);
    N_vec = (0:N-1);
    M_mat = exp(-2*pi*1i * (M_vec.' * M_vec)/M);
    N_mat = exp(-2*pi*1i * (N_vec.' * N_vec)/N);
    img_fft2 = M_mat * I * N_mat;
end


function [img_ifft2] = dip_ifft2(F)
    [M,N] = size(F);
    M_vec = (0:M-1);
    N_vec = (0:N-1);
    M_mat = exp(2*pi*1i * M_vec.' * M_vec/M);
    N_mat = exp(2*pi*1i * N_vec.' * N_vec/N);
    img_ifft2 = M_mat * F * N_mat;
    img_ifft2 = img_ifft2/(M*N);
end

function [shifted_fft2] = dip_fftshift(F)
    [M,N] = size(F);
    quad_1 = F(1:M/2,1:N/2);
    quad_2 = F(M/2+1:end, 1:N/2);
    quad_3 = F(M/2+1:end, N/2+1:end);
    quad_4 = F(1:M/2, N/2+1:end);
    shifted_fft2 = [quad_3 quad_2; quad_4 quad_1];
end

function [freed_willy] = Free_Willy(Willy)
    freed_willy = fft2(Willy);
    freed_willy(1, 1+10) = 0;
    freed_willy(1, 1+end-10) = 0;
    freed_willy = real(ifft2(freed_willy));
    figure;imshow(freed_willy);title("freed willy");
end

function [img_fft2] = sep_fft2(v1, v2)
    img_fft2 = fft(v1) * fft(v2);
end