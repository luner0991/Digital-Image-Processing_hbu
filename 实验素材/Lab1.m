%���á����ļ��Ի���ѡ����Ҫ�򿪵�ͼƬ
[file,path] = uigetfile( {'*.jpg'; '*.bmp'} );
%���δѡ���ļ���ѡ���ˡ�Cancel����ť�����˳�����
if isequal(file,0)
   disp('User selected Cancel');
   return;
end
%������·�����ļ���
fileFullFileName = fullfile(path, file);
%����ͼƬ
I = imread(file);
%��ʾͼƬ
imshow(I);

