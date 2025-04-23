%利用“打开文件对话框”选择需要打开的图片
[file,path] = uigetfile({'*.jpg'; '*.bmp'});
%如果未选择文件，选择了“Cancel”按钮，则退出程序
if isequal(file,0)
   disp('User selected Cancel');
   return;
end
%产生带路径的文件名
fileFullFileName = fullfile(path, file);
%读入图片
I = imread(file);
%显示图片（按2行3列显示，先显示第一幅图像）
subplot(2,3,1);
imshow(I);
title('Original Color Image');

%彩色图像灰度化
%首先判断当前图像是否为3通道的彩色图像
image_size=size(I);
dimension=numel(image_size);
if dimension==3 %如果是三维矩阵，则是彩色图像
    %将彩色图像转换为灰度图像
    G = rgb2gray(I);
    %显示灰度图像
    subplot(2,3,2);
    imshow(G);
    title('Gray Image');
    %显示灰度直方图
    subplot(2,3,3);
    imhist(G);
    title('Image Histogram');
else
    G = I;
end

%固定阈值分割
BW1 = im2bw(G, 220/256);
%显示图片（按2行3列显示，再显示第4幅图像）
subplot(2,3,4);
imshow(BW1);
title('Fixed threshold');

%自动阈值分割
T = graythresh(G);
BW2 = im2bw(G, T);
%显示图片（按2行3列显示，再显示第5幅图像）
subplot(2,3,5);
imshow(BW2);
title('Auto threshold');

%边缘检测
E1 = edge(G, 'sobel');
%在新窗口显示图片（按2行2列显示，显示第1幅图像）
figure, subplot(2,2,1);
imshow(I);
title('Original Color Image');
subplot(2,2,2);
imshow(E1);
title('Sobel Edge');

E2 = edge(G, 'roberts');
%在新窗口显示图片（按2行2列显示，显示第1幅图像）
subplot(2,2,3);
imshow(E1);
title('Roberts Edge');

E3 = edge(G, 'log');
%在新窗口显示图片（按2行2列显示，显示第1幅图像）
subplot(2,2,4);
imshow(E1);
title('LOG Edge');
