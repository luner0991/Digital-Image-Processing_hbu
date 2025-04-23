I = imread('tire.tif');
figure(1);
subplot(1, 2, 1); imshow(I); title('原始图像');
hgram=ones(1, 256);        %规定的直方图
J = histeq(I, hgram);        %完成直方图规定化
subplot(1, 2, 2); imshow(J); title('均衡后的图像');
figure(2);
subplot(1, 2, 1); imhist(I); title('原始图像的直方图');
subplot(1, 2, 2); imhist(J); title('均衡后图像的直方图');