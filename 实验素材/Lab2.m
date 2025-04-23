%���á����ļ��Ի���ѡ����Ҫ�򿪵�ͼƬ
[file,path] = uigetfile({'*.jpg'; '*.bmp'});
%���δѡ���ļ���ѡ���ˡ�Cancel����ť�����˳�����
if isequal(file,0)
   disp('User selected Cancel');
   return;
end
%������·�����ļ���
fileFullFileName = fullfile(path, file);
%����ͼƬ
I = imread(file);
%��ʾͼƬ����2��3����ʾ������ʾ��һ��ͼ��
subplot(2,3,1);
imshow(I);
title('Original Color Image');

%��ɫͼ��ҶȻ�
%�����жϵ�ǰͼ���Ƿ�Ϊ3ͨ���Ĳ�ɫͼ��
image_size=size(I);
dimension=numel(image_size);
if dimension==3 %�������ά�������ǲ�ɫͼ��
    %����ɫͼ��ת��Ϊ�Ҷ�ͼ��
    G = rgb2gray(I);
    %��ʾ�Ҷ�ͼ��
    subplot(2,3,2);
    imshow(G);
    title('Gray Image');
    %��ʾ�Ҷ�ֱ��ͼ
    subplot(2,3,3);
    imhist(G);
    title('Image Histogram');
    %����ͼ��ȡ������ʾ
    subplot(2,3,4);
    imshow(255-G);
    title('Reversed Gray Image');
end

%��Ӹ�˹����
GI = imnoise(I, 'gaussian');
%��ʾ���и�˹������ͼƬ
%��ʾͼƬ����2��3����ʾ������ʾ��5��ͼ��
subplot(2,3,5);
imshow(GI);
title('Gaussina Noise');

%���ø�˹�˲���ȥ����˹����
sigma1 = 10;   %��˹��̬�ֲ���׼��
GFilter = fspecial('gaussian',[5 5],sigma1);    %������˹�˲���
DeGaussianI = imfilter(GI,GFilter,'replicate'); %���и�˹�˲�
%��ʾͼƬ����2��3����ʾ������ʾ��6��ͼ��
subplot(2,3,6);
imshow(DeGaussianI);
title('Gaussian Denoise');

