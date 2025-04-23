%利用“打开文件对话框”选择需要打开的图片
[file,path] = uigetfile( {'*.jpg'; '*.bmp'} );
%如果未选择文件，选择了“Cancel”按钮，则退出程序
if isequal(file,0)
   disp('User selected Cancel');
   return;
end
%产生带路径的文件名
fileFullFileName = fullfile(path, file);
%读入图片
I = imread(file);
%显示图片
imshow(I);

