w=16;  %���ڵĿ��
I=imread('cameraman.tif');  %256*256
S=size(I);
subplot(2,2,1);   
imshow(I);
J=I(S(1)/2-w/2:S(1)/2+w/2-1,S(2)/2-w/2:S(2)/2+w/2-1) ;
 %ȡͼ���������ͼ�񣬴�СΪw*w
subplot(2,2,2);    
imshow(J);
K=I(2*w:S(1)-w,2*w:S(2)-5*w); 
%�ü�����2w����2w����11w����15w
subplot(2,2,3);    
imshow(K);
