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
subplot(2,3,1), imshow(I);
title('Original Color Image');

%��ɫͼ��ҶȻ�
%�����жϵ�ǰͼ���Ƿ�Ϊ3ͨ���Ĳ�ɫͼ��
image_size=size(I);
dimension=numel(image_size);
if dimension==3 %�������ά�������ǲ�ɫͼ��
    %����ɫͼ��ת��Ϊ�Ҷ�ͼ��
    G = rgb2gray(I);
else
    G = I;
end

%��ʾ�Ҷ�ͼ��
subplot(2,3,2), imshow(G);
title('Gray Image');

%�Զ���ֵ�ָ�
T = graythresh(G);
BW = im2bw(G, T);
%��ʾͼƬ����2��3����ʾ������ʾ��3��ͼ��
subplot(2,3,3), imshow(BW);
title('Auto threshold');
BW = not(BW);
%��ʾͼƬ����2��3����ʾ������ʾ��3��ͼ��
subplot(2,3,4), imshow(BW);
title('Reversed');

%��̬ѧ����
se = strel('disk',5);%�����Ǵ���һ���뾶Ϊ5��ƽ̹��Բ�̽ṹԪ��
BWd = imdilate(BW, se);
subplot(2,3,5),imshow(BW);
title('���ͺ��ͼ��');

BWe = imerode(BWd, se);
subplot(2,3,6),imshow(BWe);
title('��ʴ���ͼ��');


%�Էָ������������б��
[L,N] = bwlabel(BWe);

%�Ա�ǳ�����Ŀ���������ͳ��
stats = regionprops(L,'all');
for k = 1:N
    %ȡ������Ŀ��ĳ���
    MajorAL = stats(k).MajorAxisLength;
    %ȡ������Ŀ��Ķ���
    MinorAL = stats(k).MinorAxisLength;
    %���㳤���
    Ratio = MajorAL/MinorAL;
    if Ratio < 2
        disp('Orange');
    else
        disp('Banana');
    end
end
