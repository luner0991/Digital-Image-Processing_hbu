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
    %进行图像取反并显示
    subplot(2,3,4);
    imshow(255-G);
    title('Reversed Gray Image');
end

%添加高斯噪声
GI = imnoise(I, 'gaussian');
%显示带有高斯噪声的图片
%显示图片（按2行3列显示，再显示第5幅图像）
subplot(2,3,5);
imshow(GI);
title('Gaussina Noise');

%利用高斯滤波器去除高斯噪声
sigma1 = 10;   %高斯正态分布标准差
GFilter = fspecial('gaussian',[5 5],sigma1);    %创建高斯滤波器
DeGaussianI = imfilter(GI,GFilter,'replicate'); %进行高斯滤波
%显示图片（按2行3列显示，再显示第6幅图像）
subplot(2,3,6);
imshow(DeGaussianI);
title('Gaussian Denoise');

