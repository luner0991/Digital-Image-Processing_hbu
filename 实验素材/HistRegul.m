I1=imread('sjtp阴天时的玉米苗.JPG');
G1=rgb2gray(I1);
subplot(3,2,1), imhist(G1);
subplot(3,2,2), imshow(G1)
I2=imread('sjtp阴天时的玉米苗修改.jpg');
G2=rgb2gray(I2);
subplot(3,2,3), imhist(G2);
subplot(3,2,4), imshow(G2)
hist = imhist(G2);
J = histeq(G1, hist);
subplot(3,2,5), imhist(J);
subplot(3,2,6), imshow(J);