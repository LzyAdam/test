clear%%千万不能没有要不然，有些莫名其妙的错
 clc
% [R,G,B]=uigetfile('*.jpg','选择图片');%%能自己选择图片的命令
[fn,pn,fi]=uigetfile('*.jpg','选择图片');

RGB=imread([pn fn ]);
%figure('NumberTitle', 'off', 'Name', '原图');%%figure改名字

% RGB=imread('timg.jpg');
%  RGB=im2double(RGB);%%不影响
R= RGB(:, :, 1); 
G= RGB(:, :, 2); 
B= RGB(:, :, 3); 
tic
HSV=rgb2hsv(RGB);%转成HSV
 toc                                                              % % figure
                                                                            % % imshow(YUV);


%%%分三通道%%%%%%%%%%
H=HSV(:,:,1);%为Y分量矩阵* 2 * pi
S=HSV(:,:,2);%为U分量矩阵
V=HSV(:,:,3);%为V分量矩阵
RGB = hsv2rgb(HSV) ;
% imshow(RGB);%%%figure1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%合成测试，看与上面的YUV显示图一样否%%%%%%%%%
hsv = cat(3, H, S, V);  
%%%%%%显示图像的八股文%%%%%%%%
% figure('NumberTitle', 'off', 'Name', '1=>HSV图像，H通道，S通道，V通道 分别显示');%%figure改名字
% subplot(221),imshow(hsv),title('HSV');
% subplot(222),imshow(H,[]),title('H');
%  subplot(223),imshow(S),title('S');
%  subplot(224),imshow(V),title('V');
%  %%%%合成测试，看与上面的YUV显示图一样否%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%以下进行傅里叶变换%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  F=fft2(V);          %傅里叶变换
  F1=real(log(abs(F)+1));   %取模并进行缩放 !!!!!这个加上图像会变成黑红色
  Fs=fftshift(F);%% 我曹不能取模，取模他妈的出倒影，也对绝对值负的变正
  %Fs=real(fftshift(F));      %将频谱图中零频率成分移动至频谱图中心
                                        %%注意Fs=F1,如果fs=f,就出问题了
                                        %%%但是Fs=F后面6个滤波器滤出的图可见，用Fs=F1不可见                         
%   S=log(abs(Fs)+1);    %取模并进行缩放
%   FFt= real(fftshift(F1));   %YUV后，Y通道的傅里叶频谱图，YUV图，RGB图
%  V=FFt;别过这些东西直接弄
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%还原测试YUV2RGB%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V=ifft2(ifftshift(Fs));
 hsv= cat(3, H ,S , V); 
 RGB_ = hsv2rgb(hsv);%转成RGB
%  figure('NumberTitle', 'off', 'Name', '2=>YUV图像2RGB_');
%  imshow(RGB_);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%经测试可以完美还原RGB图像%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%dispaly the fft result%%%%%%%%%%%
% figure('NumberTitle', 'off', 'Name', '3=>HSV后，V通道的傅里叶频谱图，HSV图，RGB图');
% subplot(131),imshow(V,[])  ,title('V通道频谱图');
% subplot(132),imshow(hsv,[])  ,title('HSV=>FFT三通道图');
% subplot(133),imshow(RGB_,[])  ,title('RGB图');
% %%%%%%%dispaly the fft result%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%下面代码低通滤波器巴特沃斯滤波%%%%%%%%%%
%  %%%%%%%%%%%low pass filter butterworth%%%%%%%%%%%%
%  
 n4=2;%step1%滤波器的阶数2%%%%%%
 %%step2%%%%%6个低通滤波器的截止频率%%%%%%%%%%%%%%%
D0=10;D1=20;   D2=40;D3=60;  D4=80;D5=255;      
%%%step3%%%%%%%6个低通滤波器的截止频率%%%%%%%%%%%%%%%
 [M,N]=size(F);%%%%%滤波器大小适配与图片%%%%%%%
m=fix(M/2);
n=fix(N/2);
for i=1:M
   for j=1:N
        d=sqrt((i-m)^2+(j-n)^2);%%%%算点到图像中心距离%%%%%%%  
        
        %%%%%%%%%%%%巴特沃斯低通滤波器%%%%%%%%% 
        h0=1/(1+0.414*(d/D0)^(2*n4)); %计算D0=10;低通滤波器传递函数
               s0(i,j)=h0*Fs(i,j);                   %%D0=10;滤波器处理过的图像
               T0(i, j) = h0;                           %%D0=10;滤波器的样子
               
        h1=1/(1+0.414*(d/D1)^(2*n4));%计算低通滤波器传递函数
              s1(i,j)=h1*Fs(i,j);                   %%%%D1=20;滤波器处理过的图像
              T1(i, j) = h1;                          %%%%D1=20;滤波器的样子
       
        h2=1/(1+0.414*(d/D2)^(2*n4));%计算D2=40;低通滤波器传递函数
               s2(i,j)=h2*Fs(i,j);                  %%D2=40;滤波器处理过的图像
               T2(i, j) = h2;                         %%%%D2=40;滤波器的样子
               
        h3=1/(1+0.414*(d/D3)^(2*n4)); %计算D3=60;低通滤波器传递函数
                s3(i,j)=h3*Fs(i,j);                  %D3=60;滤波器处理过的图像
                T3(i, j) = h3;                         %%;D3=60;滤波器的样子
                
        h4=1/(1+0.414*(d/D4)^(2*n4)); %计算D4=80;低通滤波器传递函数
                s4(i,j)=h4*Fs(i,j);                  %%D4=80;滤波器处理过的图像
                T4(i, j) = h4;                          %%D4=80;滤波器的样子
                
        h5=1/(1+0.414*(d/D5)^(2*n4)); %计算D5=255;低通滤波器传递函数
                s5(i,j)=h5*Fs(i,j);                  %%D5=255;滤波器处理过的图像
                T5(i, j) = h5;                         %%D5=255;滤波器的样子
       
   end
end

fr0=real(ifft2(ifftshift(s0)));  %频率域反变换到空间域，并取实部
fr1=real(ifft2(ifftshift(s1)));
fr10=fr1-fr0;
fr2=real(ifft2(ifftshift(s2)));
fr21=fr2-fr1;
fr3=real(ifft2(ifftshift(s3)));
fr32=fr3-fr2;
fr4=real(ifft2(ifftshift(s4)));
fr43=fr4-fr3;
fr5=real(ifft2(ifftshift(s5)));
fr54=fr5-fr4;
% figure('NumberTitle', 'off', 'Name', '4=>6个不同频段滤波器样子及处理后图像');
% subplot(3,4,1);imshow(fr0,[]);title('D0=10的效果图');%%D0=10;滤波器处理过的图像
% subplot(3,4,2);imshow(T0);title('低通滤波器D0=10');%%D0=10;滤波器样子
% subplot(3,4,3);imshow(fr10,[]);title('D1=20 D1-D0');%%D1-D0;滤波器处理过的图像
% subplot(3,4,4);imshow(T1-T0),title('低通滤波器D1=20 D1-D0');
% subplot(3,4,5);imshow(fr21,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(T2-T1);title('低通滤波器D2=40 D2-D1');
% subplot(3,4,7);imshow(fr32,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(T3-T2),title('低通滤波器D3=60 D3-D2');
% subplot(3,4,9);imshow(fr43,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(T4-T3),title('低通滤波器D4=60 D4-D3 ');
% subplot(3,4,11);imshow(fr54,[]);title('D5=80 D5-D4');
% subplot(3,4,12);imshow(T5-T4),title('低通滤波器D5=255 D5-D4');
% 
hsv0 = cat(3,H,S,fr0);  
hsv1= cat(3, H,S,fr10);  
hsv2 = cat(3, H,S,fr21);  
hsv3 = cat(3, H,S,fr32);  
hsv4 = cat(3,  H,S,fr43);  
hsv5 = cat(3, H,S,fr54);  
RGB_0= hsv2rgb(hsv0);%转成RGB
 RGB_1= hsv2rgb(hsv1);
 RGB_2= hsv2rgb(hsv2);
 RGB_3= hsv2rgb(hsv3);
 RGB_4= hsv2rgb(hsv4);
  RGB_5= hsv2rgb(hsv5);
% figure('NumberTitle', 'off', 'Name', '5=>HSV复合6个不同频段滤波器样子及处理后图像');
% subplot(3,4,1);imshow(hsv0,[]);title('D0=10的效果图');
% subplot(3,4,2);imshow(T0,[]);title('低通滤波器D0=10');
% subplot(3,4,3);imshow(hsv1,[]);title('D1=20 D1-D0');
% subplot(3,4,4);imshow(T1-T0,[]),title('低通滤波器D1=20 D1-D0');
% subplot(3,4,5);imshow(hsv2,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(T2-T1);title('低通滤波器D2=40 D2-D1');
% subplot(3,4,7);imshow(hsv3,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(T3-T2),title('低通滤波器D3=60 D3-D2');
% subplot(3,4,9);imshow(hsv4,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(T4-T3),title('低通滤波器D4=60 D4-D3 ');
% subplot(3,4,11);imshow(hsv5,[]);title('D5=80 D5-D4');
% subplot(3,4,12);imshow(T5-T4,[]),title('低通滤波器D5=255 D5-D4');

RGB_01=imadd(1*RGB_0,0.9*RGB_1);%I,J是读入的两幅图像
 RGB_012=imadd(1*RGB_01,0.8*RGB_2);
  RGB_0123=imadd(1*RGB_012,0.7*RGB_3);
   RGB_01234=imadd(0.9*RGB_0123,0.6*RGB_4);
      RGB_012345=imadd(1.2*RGB_01234,1*RGB_5);
       S=abs( RGB_012345)+1;    %取模并进行缩放
%  figure('NumberTitle', 'off', 'Name', '6=>HSV还原RGB不同6频段滤波器样子及处理后图像');
%   subplot(231),imshow(RGB_0,[])  ,title('HSV0频段复合');
%  subplot(232),imshow(RGB_01,[])  ,title('HSV01频段复合');
%  subplot(233),imshow( RGB_012,[])  ,title('HSV012频段复合');
%  subplot(234),imshow( RGB_0123,[])  ,title('HSV0123频段复合');
% subplot(235),imshow( RGB_01234,[])  ,title('HSV01234频段复合');
%  subplot(236),imshow(RGB_012345,[])  ,title('HSV012345频段复合');
 P0=sumsqr(F1);
%  P0=sumsqr(RGB);%%%%RGB 是原始的图像
 A0= log(abs(s0)+1);A1= log(abs(s1)+1);A2= log(abs(s2)+1);
 A3= log(abs(s3)+1);A4= log(abs(s4)+1);A5= log(abs(s5)+1);
 P0=sumsqr(F1);%%%%RGB 是原始的图像
P1=sumsqr(A0)/P0;P2=sumsqr(A1)/P0;P3=sumsqr(A2)/P0;
P4=sumsqr(A3)/P0;P5=sumsqr(A4)/P0;P6=sumsqr(A5)/P0;
 





      

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%第二张图像%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % % % % % % % % [R,G,B]=uigetfile('*.jpg','选择图片');%%能自己选择图片的命令

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%这里开始%%%%%%%%%%%%%%%%%%%
[fn,pn,fi]=uigetfile('*.jpg','选择图片');

 RGB2=imread([pn fn ]);figure('NumberTitle', 'off', 'Name', '风格图');%%figure改名字

% RGB2=imread('altlas11.jpg');
 RGB2=im2double(RGB2);
R2= RGB2(:, :, 1); 
G2= RGB2(:, :, 2); 
B2= RGB2(:, :, 3); 
tic
HSV2=rgb2hsv(RGB2);%转成HSV
 toc                                                             
%  figure
%                                                                            imshow(HSV2);
% 
% 
% %%%分三通道%%%%%%%%%%
H2=HSV2(:,:,1);%为Y分量矩阵* 2 * pi
S2=HSV2(:,:,2);%为U分量矩阵
V2=HSV2(:,:,3);%为V分量矩阵
RGB2 = hsv2rgb(HSV2) ;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%合成测试，看与上面的YUV显示图一样否%%%%%%%%%
% hsv2 = cat(3, H2, S2, V2);  
% %%%%%%显示图像的八股文%%%%%%%%
% figure('NumberTitle', 'off', 'Name', '7=>HSV图像，H通道，S通道，V通道 分别显示');%%figure改名字
% subplot(221),imshow(HSV2),title('HSV2');
% subplot(222),imshow(H2,[]),title('H2');
%  subplot(223),imshow(S2),title('S2');
%  subplot(224),imshow(V2),title('V2');
%  %%%%合成测试，看与上面的YUV显示图一样否%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%以下进行傅里叶变换%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  F2=fft2(V2);          %傅里叶变换
   F1t=real(log(abs(F2)+1));   %取模并进行缩放 !!!!!这个加上图像会变成黑红色
  Fs2=fftshift(F2);%% 我曹不能取模，取模他妈的出倒影，也对绝对值负的变正
  %Fs=real(fftshift(F));      %将频谱图中零频率成分移动至频谱图中心
%                                         %%注意Fs=F1,如果fs=f,就出问题了
%                                         %%%但是Fs=F后面6个滤波器滤出的图可见，用Fs=F1不可见                         
% %   S=log(abs(Fs)+1);    %取模并进行缩放
% %   FFt= real(fftshift(F1));   %YUV后，Y通道的傅里叶频谱图，YUV图，RGB图
% %  V=FFt;别过这些东西直接弄
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%还原测试YUV2RGB%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V2=ifft2(ifftshift(Fs2));
 hsv2= cat(3, H2 ,S2 , V2); 
 RGB2= hsv2rgb(hsv2);%转成RGB
%  figure('NumberTitle', 'off', 'Name', '8=>HSV图像2RGB_');
%  imshow(RGB2);
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%经测试可以完美还原RGB图像%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%dispaly the fft result%%%%%%%%%%%
% figure('NumberTitle', 'off', 'Name', '9=>HSV后，V通道的傅里叶频谱图，HSV图，RGB图');
% subplot(131),imshow(V2,[])  ,title('V通道频谱图');
% subplot(132),imshow(hsv2,[])  ,title('HSV=>FFT三通道图');
% subplot(133),imshow(RGB2,[])  ,title('RGB图');
% % %%%%%%%dispaly the fft result%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%下面代码低通滤波器巴特沃斯滤波%%%%%%%%%%
% %  %%%%%%%%%%%low pass filter butterworth%%%%%%%%%%%%
% %  
 n4=2;%step1%滤波器的阶数2%%%%%%
 %%step2%%%%%6个低通滤波器的截止频率%%%%%%%%%%%%%%%
D0=10;D1=20;   D2=40;D3=60;  D4=80;D5=255;      
%%%step3%%%%%%%6个低通滤波器的截止频率%%%%%%%%%%%%%%%
 [M2,N2]=size(F2);%%%%%滤波器大小适配与图片%%%%%%%
m2=fix(M2/2);
n2=fix(N2/2);
for i=1:M2
   for j=1:N2
        d=sqrt((i-m2)^2+(j-n2)^2);%%%%算点到图像中心距离%%%%%%%  
        
        %%%%%%%%%%%%巴特沃斯低通滤波器%%%%%%%%% 
        h6=1/(1+0.414*(d/D0)^(2*n4)); %计算D0=10;低通滤波器传递函数
               s6(i,j)=h6*Fs2(i,j);                   %%D0=10;滤波器处理过的图像
               T6(i, j) = h6;                           %%D0=10;滤波器的样子
               
        h7=1/(1+0.414*(d/D1)^(2*n4));%计算低通滤波器传递函数
              s7(i,j)=h7*Fs2(i,j);                   %%%%D1=20;滤波器处理过的图像
              T7(i, j) = h7;                          %%%%D1=20;滤波器的样子
       
        h8=1/(1+0.414*(d/D2)^(2*n4));%计算D2=40;低通滤波器传递函数
               s8(i,j)=h8*Fs2(i,j);                  %%D2=40;滤波器处理过的图像
               T8(i, j) = h8;                         %%%%D2=40;滤波器的样子
               
        h9=1/(1+0.414*(d/D3)^(2*n4)); %计算D3=60;低通滤波器传递函数
                s9(i,j)=h9*Fs2(i,j);                  %D3=60;滤波器处理过的图像
                T9(i, j) = h9;                         %%;D3=60;滤波器的样子
                
        h10=1/(1+0.414*(d/D4)^(2*n4)); %计算D4=80;低通滤波器传递函数
                s10(i,j)=h10*Fs2(i,j);                  %%D4=80;滤波器处理过的图像
                T10(i, j) = h10;                          %%D4=80;滤波器的样子
                
        h11=1/(1+0.414*(d/D5)^(2*n4)); %计算D5=255;低通滤波器传递函数
                s11(i,j)=h11*Fs2(i,j);                  %%D5=255;滤波器处理过的图像
                T11(i, j) = h11;                         %%D5=255;滤波器的样子
       
   end
end
fr6=real(ifft2(ifftshift(s6)));  %频率域反变换到空间域，并取实部
fr7=real(ifft2(ifftshift(s7)));
fr76=fr7-fr6;
fr8=real(ifft2(ifftshift(s8)));
fr87=fr8-fr7;
fr9=real(ifft2(ifftshift(s9)));
fr98=fr9-fr8;
fr10=real(ifft2(ifftshift(s10)));
fr109=fr10-fr9;
fr11=real(ifft2(ifftshift(s11)));
fr1110=fr11-fr10;
% figure('NumberTitle', 'off', 'Name', '10=>6个不同频段滤波器样子及处理后图像');
% subplot(3,4,1);imshow(fr6,[]);title('D0=10的效果图');%%D0=10;滤波器处理过的图像
% subplot(3,4,2);imshow(T6);title('低通滤波器D0=10');%%D0=10;滤波器样子
% subplot(3,4,3);imshow(fr76,[]);title('D1=20 D1-D0');%%D1-D0;滤波器处理过的图像
% subplot(3,4,4);imshow(T7-T6),title('低通滤波器D1=20 D1-D0');
% subplot(3,4,5);imshow(fr87,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(T8-T7);title('低通滤波器D2=40 D2-D1');
% subplot(3,4,7);imshow(fr98,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(T9-T8),title('低通滤波器D3=60 D3-D2');
% subplot(3,4,9);imshow(fr109,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(T10-T9),title('低通滤波器D4=60 D4-D3 ');
% subplot(3,4,11);imshow(fr1110,[]);title('D5=80 D5-D4');
% subplot(3,4,12);imshow(T11-T10),title('低通滤波器D5=255 D5-D4');

hsv6 = cat(3,H2,S2,fr6);  
hsv7= cat(3, H2,S2,fr76);  
hsv8 = cat(3, H2,S2,fr87);  
hsv9 = cat(3, H2,S2,fr98);  
hsv10 = cat(3,  H2,S2,fr109);  
hsv11= cat(3, H2,S2,fr1110);  
RGB_6= hsv2rgb(hsv6);%转成RGB
 RGB_7= hsv2rgb(hsv7);
 RGB_8= hsv2rgb(hsv8);
 RGB_9= hsv2rgb(hsv9);
 RGB_10= hsv2rgb(hsv10);
  RGB_11= hsv2rgb(hsv11);
% figure('NumberTitle', 'off', 'Name', '11=>复合6个不同频段滤波器样子及处理后图像');
% subplot(3,4,1);imshow(hsv6,[]);title('D0=10的效果图');
% subplot(3,4,2);imshow(T6,[]);title('低通滤波器D0=10');
% subplot(3,4,3);imshow(hsv7,[]);title('D1=20 D1-D0');
% subplot(3,4,4);imshow(T7-T6,[]),title('低通滤波器D1=20 D1-D0');
% subplot(3,4,5);imshow(hsv8,[]);title('D2=40 D2-D1');
% subplot(3,4,6);imshow(T8-T7);title('低通滤波器D2=40 D2-D1');
% subplot(3,4,7);imshow(hsv9,[]);title('D3=60 D3-D2');
% subplot(3,4,8);imshow(T9-T8),title('低通滤波器D3=60 D3-D2');
% subplot(3,4,9);imshow(hsv10,[]);title('D4=80 D4-D3');
% subplot(3,4,10);imshow(T10-T9),title('低通滤波器D4=60 D4-D3 ');
% subplot(3,4,11);imshow(hsv11,[]);title('D5=80 D5-D4');
% subplot(3,4,12);imshow(T11-T10,[]),title('低通滤波器D5=255 D5-D4');


RGB_67=imadd(1*RGB_6,0.9*RGB_7);%I,J是读入的两幅图像
 RGB_678=imadd(1*RGB_67,0.8*RGB_8);
  RGB_6789=imadd(1*RGB_678,0.7*RGB_9);
   RGB_678910=imadd(0.9*RGB_6789,0.6*RGB_10);
      RGB_67891011=imadd(1.2*RGB_678910,1*RGB_11);
  
%  figure('NumberTitle', 'off', 'Name', '12=>=>HSV还原RGB不同6频段滤波器样子及处理后图像');
%  subplot(231),imshow(RGB_6,[])  ,title('HSV6频段复合');
%  subplot(232),imshow(RGB_67,[])  ,title('HSV67频段复合');
%  subplot(233),imshow( RGB_678,[])  ,title('HSV678频段复合');
%  subplot(234),imshow( RGB_6789,[])  ,title('HSV6789频段复合');
% subplot(235),imshow( RGB_678910,[])  ,title('HSV678910频段复合');
%  subplot(236),imshow(RGB_67891011,[])  ,title('HSV67891011频段复合');
%  
 Pt=sumsqr(F1t);
%  P0=sumsqr(RGB);%%%%RGB 是原始的图像
 A6= log(abs(s6)+1);A7= log(abs(s7)+1);A8= log(abs(s8)+1);
 A9= log(abs(s9)+1);A10= log(abs(s10)+1);A11= log(abs(s11)+1);
% % %  计算能量百分比
P7=sumsqr(A6)/Pt;P8=sumsqr(A7)/Pt;P9=sumsqr(A8)/Pt;
P10=sumsqr(A9)/Pt;P11=sumsqr(A10)/Pt;P12=sumsqr(A11)/Pt;
%风格增益的计算
% G1=sqrt(P1/((P7+P1)+1));G2=sqrt(P2/((P8+P2)+1));G3=sqrt(P3/((P9+P3)+1));
%   G4=sqrt(P4/((P10+P4)+1));G5=sqrt(P5/((P11+P5)+1));G6=sqrt(P6/((P12+P6)+1));
%  G1=sqrt(P1/((P7+P1)*10));G2=sqrt(P2/((P8+P2)*10));G3=sqrt(P3/((P9+P3)*10));
%   G4=sqrt(P4/((P10+P4)*10));G5=sqrt(P5/((P11+P5)*10));G6=sqrt(P6/((P12+P6)*10));
% % % ok分子平方atlas
  G1=sqrt((P1^2)/((P7+P1)));G2=sqrt((P2^2)/((P8+P2)));G3=sqrt((P3^2)/((P9+P3)));
  G4=sqrt((P4^2)/((P10+P4)));G5=sqrt((P5^2)/((P11+P5)));G6=sqrt((P6^2)/((P12+P6)));
% % % lily
% %  G1=sqrt((P1)/((P7+0.3)));G2=sqrt((P2)/((P8+0.3)));G3=sqrt((P3)/((P9+0.3)));
% %   G4=sqrt((P4)/((P10+0.3)));G5=sqrt((P5)/((P11+0.3)));G6=sqrt((P6)/((P12+0.3)));
% G1=sqrt((P1^2)/((P7)));G2=sqrt((P2^2)/((P8)));G3=sqrt((P3^2)/((P9)));
%   G4=sqrt((P4^2)/((P10)));G5=sqrt((P5^2)/((P11)));G6=sqrt((P6^2)/((P12)));
%   G1=sqrt((P1^2)/((P7+P1)));G2=sqrt((P2)/((P8+P2)));G3=sqrt((P3)/((P9+P3)));
%   G4=sqrt((P4)/((P10+P4)));G5=sqrt((P5)/((P11+P5)));G6=sqrt((P6)/((P12+P6)));
% G1=sqrt((P1^2)/((P7+P1)));G2=sqrt((P2)/((P8+P2)));G3=sqrt((P3)/((P9+P3)));
%   G4=sqrt((P4)/((P10+P4)));G5=sqrt((P5)/((P11+P5)));G6=sqrt((P6)/((P12+P6)));
  %%%%%%%按照风格增益合成图像
%   RGB_06=imadd(0.9*RGB_0,G1*RGB_6);
%   RGB_17=imadd(0.8*RGB_1,G2*RGB_7);
%   RGB_28=imadd(0.7*RGB_2,G3*RGB_8);
%   RGB_39=imadd(0.6*RGB_3,G4*RGB_9);
%   RGB_410=imadd(0.5*RGB_4,G5*RGB_10);
%   RGB_511=imadd(0.5*RGB_5,G6*RGB_11);
%%test原来
  RGB_06=imadd(RGB_0,G1*RGB_6);
  RGB_17=imadd(RGB_1,G2*RGB_7);
  RGB_28=imadd(RGB_2,G3*RGB_8);
  RGB_39=imadd(RGB_3,G4*RGB_9);
  RGB_410=imadd(RGB_4,G5*RGB_10);
  RGB_511=imadd(RGB_5,G6*RGB_11);
%  RGB_06=imadd(0.8*RGB_0,G1*RGB_6);
%   RGB_17=imadd(0.8*RGB_1,G2*RGB_7);
%   RGB_28=imadd(0.8*RGB_2,G3*RGB_8);
%   RGB_39=imadd(0.8*RGB_3,G4*RGB_9);
%   RGB_410=imadd(0.8*RGB_4,G5*RGB_10);
%   RGB_511=imadd(0.8*RGB_5,G6*RGB_11);
%   figure('NumberTitle', 'off', 'Name', '李海洋风格增益效果');
% %   imshow(RGB_06);
% %    figure('NumberTitle', 'off', 'Name', '12=>=>HSV还原RGB不同6频段滤波器样子及处理后图像');
%  subplot(231),imshow(RGB_06,[])  ,title('HSV6频段复合');
%  subplot(232),imshow(RGB_17,[])  ,title('HSV67频段复合');
%  subplot(233),imshow( RGB_28,[])  ,title('HSV678频段复合');
%  subplot(234),imshow( RGB_39,[])  ,title('HSV6789频段复合');
% subplot(235),imshow( RGB_410,[])  ,title('HSV678910频段复合');
%  subplot(236),imshow(RGB_511,[])  ,title('HSV67891011频段复合');
 
%   figure('NumberTitle', 'off', 'Name', '李海洋风格增益效果');
%   imshow(RGB_06,[])  ,title('HSV6频段复合');
  RGB_Q=imlincomb(1,RGB_06, 0.9,RGB_17, 0.8,RGB_28,0.7,RGB_39,0.6,RGB_410,0.5,RGB_511);
%   RGB_E=imlincomb(1,RGB_06, 1,RGB_17, 1,RGB_28,1,RGB_39,1,RGB_410,1,RGB_511);
  RGB_E=imlincomb(1,RGB_06, 2,RGB_17, 3,RGB_28,4,RGB_39,5,RGB_410,6,RGB_511);
%   RGB_E=imlincomb(1,RGB_06, 2,RGB_17, 5,RGB_28,4,RGB_39,2,RGB_410,1,RGB_511);
%     RGB_E=imlincomb(1,RGB_06, 5,RGB_17, 2,RGB_28,3,RGB_39,52,RGB_410,9,RGB_511);
%       RGB_Y=imlincomb(1,RGB_06, 9,RGB_17, 4,RGB_28,3,RGB_39,2,RGB_410,1,RGB_511);
%   %D0=10;D1=20;   D2=40;D3=60;  D4=80;D5=255;     
%    subplot(231),imshow(RGB_Q,[])  ,title('RGB06');

 figure('NumberTitle', 'off', 'Name', '逐级递减的复合比例');
imshow(RGB_Q,[])  ,title('1:0.9:0.8:0.7:0.6:0.5');
figure('NumberTitle', 'off', 'Name', '相同比例');
RGB_W=imlincomb(1,RGB_06, 1,RGB_17, 1,RGB_28,1,RGB_39,1,RGB_410,1,RGB_511);
imshow(RGB_W,[])  ,title('1:1:1:1:1:1');
figure('NumberTitle', 'off', 'Name', '123432');
RGB_F=imlincomb(1,RGB_06, 2,RGB_17, 3,RGB_28,4,RGB_39,3,RGB_410,2,RGB_511);
imshow(RGB_F,[])  ,title('1:2:3:4:3:2');
figure('NumberTitle', 'off', 'Name', '增益百分比倒数');
RGB_T=imlincomb(0.12/G1,RGB_06, 0.12/G2,RGB_17, 0.12/G3,RGB_28,0.12/G4,RGB_39,0.12/G5,RGB_410,0.12/G6,RGB_511);
imshow(RGB_T,[])  ,title('0.1/(G1:G2:G3:G4:G5:G6)');
figure('NumberTitle', 'off', 'Name', '增益比递增');
RGB_T=imlincomb(1,RGB_06, 2,RGB_17, 3,RGB_28,4,RGB_39,5,RGB_410,6,RGB_511);
imshow(RGB_T,[])  ,title('1:2:3:4:5:6');
figure('NumberTitle', 'off', 'Name', '9G');
RGB_P=imlincomb(9*G1,RGB_06, 9*G2,RGB_17, 9*G3,RGB_28,9*G4,RGB_39,9*G5,RGB_410,9*G6,RGB_511);
imshow(RGB_P,[])  ,title('9*G');
figure('NumberTitle', 'off', 'Name', '8G');
RGB_B=imlincomb(8*G1,RGB_06, 8*G2,RGB_17, 8*G3,RGB_28,8*G4,RGB_39,8*G5,RGB_410,8*G6,RGB_511);
imshow(RGB_B,[])  ,title('8*G');
figure('NumberTitle', 'off', 'Name', '7G');
RGB_L=imlincomb(7*G1,RGB_06, 7*G2,RGB_17, 7*G3,RGB_28,7*G4,RGB_39,7*G5,RGB_410,7*G6,RGB_511);
imshow(RGB_L,[])  ,title('7*G');
% FUCK=saveas(RGB_L,'D:\matlabresult.jpg');
% saveas(gcf,['D:\matlabresult\','test1.jpg']);
% [fileName,pathName]=uiputfile({'*.jpg;*.tif;*.png*.gif'});
% RGB_L=[pathName fileName];
% imwrite(k,RGB_L);
% [fn,pn]=uiputfile('*.jpg','保存图片');
% % [pathName fileName]=RGB_L;
% % imwrite(RGB_L,'111111.jpg');
% imwrite(RGB_L,[pn,fn]);
% [imgfilename,imgpathname]=uiputfile('*.jpg','保存图片');
% % [pathName fileName]=RGB_L;
% % imwrite(RGB_L,'111111.jpg');
% imwrite(RGB_L,[imgpathname,imgfilename]);
[imgfilename,imgpathname,fi]=uigetfile('*.jpg','选择图片');

 RGB=imread([imgpathname imgfilename ]);figure('NumberTitle', 'off', 'Name', '原图');%%figure改名字


% [fn,pn,fi]=uigetfile('*.jpg','选择图片');
% 
% RGB=imread([pn fn ]);

% [file,path] = uiputfile('*.jpg','Save file name');


% imwrite(imgrgb,'flower.bmp','.bmp');%.jpg格式转换为bmp格式
% imggray=rgb2gray(imgrgb);
% imwrite(imggray,'flower_grayscale.bmp','bmp');%存储为灰度图像
% [imgind,map]=rgb2ind(imgrgb,256);%转换为256色的索引图像
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 



%  subplot(233),imshow( RGB_E,[])  ,title('1:1:1:1:1:1:1');
%  subplot(234),imshow( RGB_R,[])  ,title('1:2:3:4:5:6');
% subplot(235),imshow( RGB_T,[])  ,title('1:2:5:4:2:1');
%  subplot(236),imshow(RGB_Y,[])  ,title('152351');
%  figure('NumberTitle', 'off', 'Name', '本文提出风格增益效果按照1:1:1:1:1:1:1合成');
%   imshow( RGB_E,[])  ,title('本文提出风格增益按照1:1:1:1:1:1:1混合');
%   figure('NumberTitle', 'off', 'Name', '本文风格增益，分子平方');
%  %imshow( RGB_E,[])  ,title('全部分子平方');
% % imshow( RGB_E,[])  ,title('2345分子平方');
% %   subplot(233),imshow( RGB_E,[])  ,title('1345分子平方');
% %     subplot(234),imshow( RGB_E,[])  ,title('1245分子平方');
% %       subplot(235),imshow( RGB_E,[])  ,title('1235分子平方');
%    imshow( RGB_E,[])  ,title('分子平方只有G1做');
     %imshow( RGB_E,[])  ,title('分母x10');
%  figure('NumberTitle', 'off', 'Name', '15=>139不同频段强化对照组');
%  RGB_D=imlincomb(1,RGB_06, 0.5,RGB_17, 0.5,RGB_28,0.5,RGB_39,0.6,RGB_410,0.5,RGB_511);
%   RGB_I=imlincomb(1,RGB_06, 9,RGB_17, 3,RGB_28,3,RGB_39,3,RGB_410,3,RGB_511);
%   RGB_C=imlincomb(1,RGB_06, 3,RGB_17, 9,RGB_28,3,RGB_39,3,RGB_410,3,RGB_511);
%   RGB_V=imlincomb(1,RGB_06, 2,RGB_17, 3,RGB_28,9,RGB_39,2,RGB_410,3,RGB_511);
%     RGB_N=imlincomb(1,RGB_06, 3,RGB_17,3,RGB_28,3,RGB_39,9,RGB_410,3,RGB_511);
%       RGB_M=imlincomb(1,RGB_06, 3,RGB_17, 3,RGB_28,3,RGB_39,2,RGB_410,9,RGB_511);

%   subplot(231),imshow(RGB_D,[])  ,title('1:0.5:0.5:0.5:0.5');
%  subplot(232),imshow(RGB_I,[])  ,title('1:933333');
%  subplot(233),imshow( RGB_C,[])  ,title('1:39333');
%  subplot(234),imshow( RGB_V,[])  ,title('1:33933');
% subplot(235),imshow( RGB_N,[])  ,title('13333393');
%  subplot(236),imshow(RGB_M,[])  ,title('1333339');
%  figure('NumberTitle', 'off', 'Name', '16=>124不同频段强化对照组');
%  RGB_D=imlincomb(1,RGB_06, 4,RGB_17, 2,RGB_28,2,RGB_39,2,RGB_410,2,RGB_511);
%   RGB_X=imlincomb(1,RGB_06, 2,RGB_17, 4,RGB_28,2,RGB_39,2,RGB_410,2,RGB_511);
%   RGB_C=imlincomb(1,RGB_06, 2,RGB_17, 2,RGB_28,4,RGB_39,2,RGB_410,2,RGB_511);
%   RGB_V=imlincomb(1,RGB_06, 2,RGB_17, 2,RGB_28,2,RGB_39,4,RGB_410,2,RGB_511);
%     RGB_N=imlincomb(1,RGB_06, 2,RGB_17,2,RGB_28,2,RGB_39,2,RGB_410,4,RGB_511);
%       RGB_M=imlincomb(0.85,RGB_06, 2,RGB_17, 2,RGB_28,2,RGB_39,2,RGB_410,4,RGB_511);

%   subplot(231),imshow(RGB_D,[])  ,title('142222');
%  subplot(232),imshow(RGB_X,[])  ,title('RGBI=>124444');
%  subplot(233),imshow( RGB_C,[])  ,title('=>122422');
%  subplot(234),imshow( RGB_V,[])  ,title('122242');
% subplot(235),imshow( RGB_N,[])  ,title('122224');
%  subplot(236),imshow(RGB_M,[])  ,title('1333339');
%  figure('NumberTitle', 'off', 'Name', '17=>124不同频段强化对照组');
%  RGB_Z=imlincomb(1,RGB_06, 1.5,RGB_17, 3,RGB_28,1.5,RGB_39,1.5,RGB_410,1.5,RGB_511);
%   RGB_X=imlincomb(1,RGB_06, 3,RGB_17, 1.5,RGB_28,1.5,RGB_39,2,RGB_410,2,RGB_511);
%   RGB_C=imlincomb(1,RGB_06, 2,RGB_17, 2,RGB_28,3,RGB_39,2,RGB_410,2,RGB_511);
%   RGB_V=imlincomb(1,RGB_06, 3,RGB_17, 4,RGB_28,4,RGB_39,5,RGB_410,2,RGB_511);
%     RGB_N=imlincomb(1,RGB_06, 2,RGB_17,2,RGB_28,2,RGB_39,2,RGB_410,3,RGB_511);
%       RGB_M=imlincomb(0.85,RGB_06, 1.5,RGB_17, 4,RGB_28,5,RGB_39,3,RGB_410,1,RGB_511);
% 
%   subplot(231),imshow(RGB_Z,[])  ,title('1;1.5;3');
%  subplot(232),imshow(RGB_X,[])  ,title('RGBI=>124444');
%  subplot(233),imshow( RGB_C,[])  ,title('=>122422');
%  subplot(234),imshow( RGB_V,[])  ,title('122242');
% subplot(235),imshow( RGB_N,[])  ,title('122224');
%  subplot(236),imshow(RGB_M,[])  ,title('1;1.5;4;5;3;1');
%  
 figure('NumberTitle', 'off', 'Name', '两张原始图像');
  subplot(121),imshow(RGB,[])  ,title('人脸图像');
 subplot(122),imshow(RGB2,[])  ,title('风格迁移图像');
%  figure('NumberTitle', 'off', 'Name', '两张原始图像与迁移图像对比');
%   subplot(131),imshow(RGB,[])  ,title('人脸图像');
%  subplot(132),imshow(RGB2,[])  ,title('风格迁移图像');
%   subplot(133),imshow(RGB_V,[])  ,title('效果图');