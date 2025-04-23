w=16;  %窗口的宽度
I=imread('cameraman.tif');  %256*256
S=size(I);
subplot(2,2,1);   
imshow(I);
J=I(S(1)/2-w/2:S(1)/2+w/2-1,S(2)/2-w/2:S(2)/2+w/2-1) ;
 %取图像中央的子图像，大小为w*w
subplot(2,2,2);    
imshow(J);
K=I(2*w:S(1)-w,2*w:S(2)-5*w); 
%裁剪：左2w，上2w，下11w，右15w
subplot(2,2,3);    
imshow(K);
