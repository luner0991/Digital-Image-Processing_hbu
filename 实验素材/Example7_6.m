I = imread('tire.tif');
figure(1);
subplot(1, 2, 1); imshow(I); title('ԭʼͼ��');
hgram=ones(1, 256);        %�涨��ֱ��ͼ
J = histeq(I, hgram);        %���ֱ��ͼ�涨��
subplot(1, 2, 2); imshow(J); title('������ͼ��');
figure(2);
subplot(1, 2, 1); imhist(I); title('ԭʼͼ���ֱ��ͼ');
subplot(1, 2, 2); imhist(J); title('�����ͼ���ֱ��ͼ');