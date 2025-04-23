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
else
    G = I;
end

%�̶���ֵ�ָ�
BW1 = im2bw(G, 220/256);
%��ʾͼƬ����2��3����ʾ������ʾ��4��ͼ��
subplot(2,3,4);
imshow(BW1);
title('Fixed threshold');

%�Զ���ֵ�ָ�
T = graythresh(G);
BW2 = im2bw(G, T);
%��ʾͼƬ����2��3����ʾ������ʾ��5��ͼ��
subplot(2,3,5);
imshow(BW2);
title('Auto threshold');

%��Ե���
E1 = edge(G, 'sobel');
%���´�����ʾͼƬ����2��2����ʾ����ʾ��1��ͼ��
figure, subplot(2,2,1);
imshow(I);
title('Original Color Image');
subplot(2,2,2);
imshow(E1);
title('Sobel Edge');

E2 = edge(G, 'roberts');
%���´�����ʾͼƬ����2��2����ʾ����ʾ��1��ͼ��
subplot(2,2,3);
imshow(E1);
title('Roberts Edge');

E3 = edge(G, 'log');
%���´�����ʾͼƬ����2��2����ʾ����ʾ��1��ͼ��
subplot(2,2,4);
imshow(E1);
title('LOG Edge');
