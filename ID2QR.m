function ID2QR(ID)
if length(ID)~=9 
    fprintf('Please insert a string with 9 values [0,9]')
    return
end
QR = zeros(6);

for i = 1:9
    binstr = dec2bin(str2double(ID(i)),4);
    for j = 1:4
        QR(j+(i-1)*4) = logical(str2double(binstr(j)));
    end
end
QR = imresize(QR,[256,256],'nearest');
B = zeros(size(QR)+2);
B(2:end-1,2:end-1)=QR;
imshow(B)