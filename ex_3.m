close all
clc
disp('Elias Assaf 315284729 - Jameel Nassar 206985152')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1

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
%% Section 2
close all;
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
figure;imshow(abs((prison_fft)));title("prison fft2")
Free_Willy(imprisoned_willy);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 3

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