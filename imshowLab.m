function imshowLab(L,a,b)
%Usage: imshowLab(L,a,b)
%This function is equivilent to imshow but displays the L*a*b as a color image.
%The inputs are:
%L - an NxM matrix of type double of the Luminence channel in the range [0,1]
%a - an NxM matrix of type double of the a channel in the range [0,1]
%b - an NxM matrix of type double of the b channel in the range [0,1]

%The function does not open a new figure
%
%Author: Assaf Arbelle {arbellea@post.bgu.ac.il}

RGB = lab2rgb(cat(3,L,a,b));
imshow(RGB,[])
end
