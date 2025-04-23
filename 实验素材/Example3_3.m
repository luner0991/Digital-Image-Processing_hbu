I1=imread('cameraman.tif');  %256*256
subplot(2,3,1);   
imshow(I1), title('8 bit');
I2=(I1/4)*4;
subplot(2,3,2); 
imshow(I2),title('6 bit');
I3=(I1/16)*16;
subplot(2,3,3);    
imshow(I3),title('4 bit');
I3=(I1/32)*32;
subplot(2,3,4);    
imshow(I3),title('3 bit');
I3=(I1/64)*64;
subplot(2,3,5);    
imshow(I3),title('2 bit');
I3=(I1/128)*128;
subplot(2,3,5);    
imshow(I3),title('1 bit');