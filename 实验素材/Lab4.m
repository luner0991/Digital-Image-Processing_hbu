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
subplot(2,3,1), imshow(I);
title('Original Color Image');

%彩色图像灰度化
%首先判断当前图像是否为3通道的彩色图像
image_size=size(I);
dimension=numel(image_size);
if dimension==3 %如果是三维矩阵，则是彩色图像
    %将彩色图像转换为灰度图像
    G = rgb2gray(I);
else
    G = I;
end

%显示灰度图像
subplot(2,3,2), imshow(G);
title('Gray Image');

%自动阈值分割
T = graythresh(G);
BW = im2bw(G, T);
%显示图片（按2行3列显示，再显示第3幅图像）
subplot(2,3,3), imshow(BW);
title('Auto threshold');
BW = not(BW);
%显示图片（按2行3列显示，再显示第3幅图像）
subplot(2,3,4), imshow(BW);
title('Reversed');

%形态学处理
se = strel('disk',5);%这里是创建一个半径为5的平坦型圆盘结构元素
BWd = imdilate(BW, se);
subplot(2,3,5),imshow(BW);
title('膨胀后的图像');

BWe = imerode(BWd, se);
subplot(2,3,6),imshow(BWe);
title('腐蚀后的图像');


%对分割出来的区域进行标记
[L,N] = bwlabel(BWe);

%对标记出来的目标进行特征统计
stats = regionprops(L,'all');
for k = 1:N
    %取出各个目标的长轴
    MajorAL = stats(k).MajorAxisLength;
    %取出各个目标的短轴
    MinorAL = stats(k).MinorAxisLength;
    %计算长宽比
    Ratio = MajorAL/MinorAL;
    if Ratio < 2
        disp('Orange');
    else
        disp('Banana');
    end
end
