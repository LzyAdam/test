function varargout = GUIfinalpaper(varargin)
% GUIFINALPAPER MATLAB code for GUIfinalpaper.fig
%      GUIFINALPAPER, by itself, creates a new GUIFINALPAPER or raises the existing
%      singleton*.
%
%      H = GUIFINALPAPER returns the handle to a new GUIFINALPAPER or the handle to
%      the existing singleton*.
%
%      GUIFINALPAPER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUIFINALPAPER.M with the given input arguments.
%
%      GUIFINALPAPER('Property','Value',...) creates a new GUIFINALPAPER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUIfinalpaper_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUIfinalpaper_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUIfinalpaper

% Last Modified by GUIDE v2.5 27-Apr-2020 13:11:42

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUIfinalpaper_OpeningFcn, ...
                   'gui_OutputFcn',  @GUIfinalpaper_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GUIfinalpaper is made visible.
function GUIfinalpaper_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUIfinalpaper (see VARARGIN)

% Choose default command line output for GUIfinalpaper
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GUIfinalpaper wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GUIfinalpaper_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[fileName,pathName] = uigetfile({'*.jpg;*.tif;*.png*.gif'},'选图');
global RGB;
global P1;global P2;global P3;global P4;global P5;global P0;
global RGB_1;global RGB_2;global RGB_3;global RGB_4;global RGB_5;global RGB_0;
RGB=[pathName,fileName];
RGB=imread(RGB);
R= RGB(:, :, 1); 
G= RGB(:, :, 2); 
B= RGB(:, :, 3); 
HSV=rgb2hsv(RGB);
H=HSV(:,:,1);%为Y分量矩阵* 2 * pi
S=HSV(:,:,2);%为U分量矩阵
V=HSV(:,:,3);%为V分量矩阵
RGB = hsv2rgb(HSV) ;
 F=fft2(V);          %傅里叶变换
  F1=real(log(abs(F)+1));   %取模并进行缩放 !!!!!这个加上图像会变成黑红色
  Fs=fftshift(F);%% 我曹不能取模，取模他妈的出倒影，也对绝对值负的变正
   
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


RGB_01=imadd(1*RGB_0,0.9*RGB_1);%I,J是读入的两幅图像
 RGB_012=imadd(1*RGB_01,0.8*RGB_2);
  RGB_0123=imadd(1*RGB_012,0.7*RGB_3);
   RGB_01234=imadd(0.9*RGB_0123,0.6*RGB_4);
      RGB_012345=imadd(1.2*RGB_01234,1*RGB_5);
      
 P0=sumsqr(F1);

 A0= log(abs(s0)+1);A1= log(abs(s1)+1);A2= log(abs(s2)+1);
 A3= log(abs(s3)+1);A4= log(abs(s4)+1);A5= log(abs(s5)+1);
 P0=sumsqr(F1);%%%%RGB 是原始的图像
P1=sumsqr(A0)/P0;P2=sumsqr(A1)/P0;P3=sumsqr(A2)/P0;
P4=sumsqr(A3)/P0;P5=sumsqr(A4)/P0;P6=sumsqr(A5)/P0;

b=size(RGB);%%%%no
if numel(b)>2%%%%no
   axes(handles.axes1)%%%%no
    subimage(RGB);%%%%no
%     set(handles.gray,'Enable','on');
%     set(handles.erzhi,'Enable','on');
    axis off;%%%%no
end

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[fileName,pathName] = uigetfile({'*.jpg;*.tif;*.png*.gif'},'选图');
global RGB2;
global P7;global P8;global P9;global P10;global P11;global P12;
global RGB_6;global RGB_7;global RGB_8;global RGB_9;global RGB_10;global RGB_11;
% RGB_6= hsv2rgb(hsv6);%转成RGB
%  RGB_7= hsv2rgb(hsv7);
%  RGB_8= hsv2rgb(hsv8);
%  RGB_9= hsv2rgb(hsv9);
%  RGB_10= hsv2rgb(hsv10);
%   RGB_11= hsv2rgb(hsv11);
RGB2=[pathName,fileName];
RGB2=imread(RGB2);
% % % % % % % % % A6= log(abs(s6)+1);A7= log(abs(s7)+1);A8= log(abs(s8)+1);
% % % % % % % % %  A9= log(abs(s9)+1);A10= log(abs(s10)+1);A11= log(abs(s11)+1);
% % % % % % % % % % % %  计算能量百分比
% % % % % % % % % P7=sumsqr(A6)/Pt;P8=sumsqr(A7)/Pt;P9=sumsqr(A8)/Pt;
% % % % % % % % % P10=sumsqr(A9)/Pt;P11=sumsqr(A10)/Pt;P12=sumsqr(A11)/Pt;
% fn,pn,fi]=uigetfile('*.jpg','选择图片');
% 
%  RGB2=imread([pn fn ]);figure('NumberTitle', 'off', 'Name', '风格图');%%figure改名字


 RGB2=im2double(RGB2);
R2= RGB2(:, :, 1); 
G2= RGB2(:, :, 2); 
B2= RGB2(:, :, 3); 
HSV2=rgb2hsv(RGB2);
H2=HSV2(:,:,1);%为Y分量矩阵* 2 * pi
S2=HSV2(:,:,2);%为U分量矩阵
V2=HSV2(:,:,3);%为V分量矩阵
RGB2 = hsv2rgb(HSV2) ;
  F2=fft2(V2);          %傅里叶变换
   F1t=real(log(abs(F2)+1)); 
  Fs2=fftshift(F2);%% 
  V2=ifft2(ifftshift(Fs2));
 hsv2= cat(3, H2 ,S2 , V2); 
 RGB2= hsv2rgb(hsv2);%转成RGB

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



RGB_67=imadd(1*RGB_6,0.9*RGB_7);%I,J是读入的两幅图像
 RGB_678=imadd(1*RGB_67,0.8*RGB_8);
  RGB_6789=imadd(1*RGB_678,0.7*RGB_9);
   RGB_678910=imadd(0.9*RGB_6789,0.6*RGB_10);
      RGB_67891011=imadd(1.2*RGB_678910,1*RGB_11);
      Pt=sumsqr(F1t);
%  P0=sumsqr(RGB);%%%%RGB 是原始的图像
 A6= log(abs(s6)+1);A7= log(abs(s7)+1);A8= log(abs(s8)+1);
 A9= log(abs(s9)+1);A10= log(abs(s10)+1);A11= log(abs(s11)+1);
% % %  计算能量百分比
P7=sumsqr(A6)/Pt;P8=sumsqr(A7)/Pt;P9=sumsqr(A8)/Pt;
P10=sumsqr(A9)/Pt;P11=sumsqr(A10)/Pt;P12=sumsqr(A11)/Pt;

b=size(RGB2);%%%%no
if numel(b)>2%%%%no
   axes(handles.axes2)%%%%no
    subimage(RGB2);%%%%no
%     set(handles.gray,'Enable','on');
%     set(handles.erzhi,'Enable','on');
    axis off;%%%%no
% [fn,pn,fi]=uigetfile('*.jpg','选择图片');
% 
%  RGB2=imread([pn fn ]);figure('NumberTitle', 'off', 'Name', '风格图');%%figure改名字




%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%以下进行傅里叶变换%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 


 



  

%  
 
end

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global RGB;
global P1;global P2;global P3;global P4;global P5;global P0;
global RGB_1;global RGB_2;global RGB_3;global RGB_4;global RGB_5;global RGB_0;
global RGB2;
global P7;global P8;global P9;global P10;global P11;global P6;global P12;
global RGB_6;global RGB_7;global RGB_8;global RGB_9;global RGB_10;global RGB_11;
G1=sqrt((P1^2)/((P7+P1)));G2=sqrt((P2^2)/((P8+P2)));G3=sqrt((P3^2)/((P9+P3)));
  G4=sqrt((P4^2)/((P10+P4)));G5=sqrt((P5^2)/((P11+P5)));G6=sqrt((P6^2)/((P12+P6)));
  
  RGB_06=imadd(RGB_0,G1*RGB_6);
  RGB_17=imadd(RGB_1,G2*RGB_7);
  RGB_28=imadd(RGB_2,G3*RGB_8);
  RGB_39=imadd(RGB_3,G4*RGB_9);
  RGB_410=imadd(RGB_4,G5*RGB_10);
  RGB_511=imadd(RGB_5,0.46*RGB_11);
%   figure('NumberTitle', 'off', 'Name', '8G');
RGB_B=imlincomb(8*G1,RGB_06, 8*G2,RGB_17, 8*G3,RGB_28,8*G4,RGB_39,8*G5,RGB_410,8*0.46,RGB_511);
% imshow(RGB_B,[])  ;
% axes(handles.axes3);
b=size(RGB_B);%%%%no
if numel(b)>2%%%%no
   axes(handles.axes3)%%%%no
    subimage(RGB_B);%%%%no
%     set(handles.gray,'Enable','on');
%     set(handles.erzhi,'Enable','on');
    axis off;%%%%no

 end


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global RGB;
global P1;global P2;global P3;global P4;global P5;global P0;
global RGB_1;global RGB_2;global RGB_3;global RGB_4;global RGB_5;global RGB_0;
global RGB2;
global P7;global P8;global P9;global P10;global P11;global P6;global P12;
global RGB_6;global RGB_7;global RGB_8;global RGB_9;global RGB_10;global RGB_11;
G1=sqrt((P1^2)/((P7+P1)));G2=sqrt((P2^2)/((P8+P2)));G3=sqrt((P3^2)/((P9+P3)));
  G4=sqrt((P4^2)/((P10+P4)));G5=sqrt((P5^2)/((P11+P5)));G6=sqrt((P6^2)/((P12+P6)));
  
  RGB_06=imadd(RGB_0,G1*RGB_6);
  RGB_17=imadd(RGB_1,G2*RGB_7);
  RGB_28=imadd(RGB_2,G3*RGB_8);
  RGB_39=imadd(RGB_3,G4*RGB_9);
  RGB_410=imadd(RGB_4,G5*RGB_10);
  RGB_511=imadd(RGB_5,0.46*RGB_11);
%   figure('NumberTitle', 'off', 'Name', '8G');
RGB_B=imlincomb(8*G1,RGB_06, 6*G2,RGB_17, 6*G3,RGB_28,6*G4,RGB_39,4*G5,RGB_410,4*0.46,RGB_511);
% imshow(RGB_B,[])  ;
% axes(handles.axes3);
b=size(RGB_B);%%%%no
if numel(b)>2%%%%no
   axes(handles.axes3)%%%%no
    subimage(RGB_B);%%%%no
%     set(handles.gray,'Enable','on');
%     set(handles.erzhi,'Enable','on');
    axis off;%%%%no

 end

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global RGB;
global P1;global P2;global P3;global P4;global P5;global P0;
global RGB_1;global RGB_2;global RGB_3;global RGB_4;global RGB_5;global RGB_0;
global RGB2;
global P7;global P8;global P9;global P10;global P11;global P6;global P12;
global RGB_6;global RGB_7;global RGB_8;global RGB_9;global RGB_10;global RGB_11;
G1=sqrt((P1^2)/((P7+P1)));G2=sqrt((P2^2)/((P8+P2)));G3=sqrt((P3^2)/((P9+P3)));
  G4=sqrt((P4^2)/((P10+P4)));G5=sqrt((P5^2)/((P11+P5)));G6=sqrt((P6^2)/((P12+P6)));
  
  RGB_06=imadd(RGB_0,G1*RGB_6);
  RGB_17=imadd(RGB_1,G2*RGB_7);
  RGB_28=imadd(RGB_2,G3*RGB_8);
  RGB_39=imadd(RGB_3,G4*RGB_9);
  RGB_410=imadd(RGB_4,G5*RGB_10);
  RGB_511=imadd(RGB_5,0.46*RGB_11);
%   figure('NumberTitle', 'off', 'Name', '8G');
RGB_B=imlincomb(8*G1,RGB_06, 8*G2,RGB_17, 6*G3,RGB_28,4*G4,RGB_39,2*G5,RGB_410,1*0.46,RGB_511);
% imshow(RGB_B,[])  ;
% axes(handles.axes3);
b=size(RGB_B);%%%%no
if numel(b)>2%%%%no
   axes(handles.axes3)%%%%no
    subimage(RGB_B);%%%%no
%     set(handles.gray,'Enable','on');
%     set(handles.erzhi,'Enable','on');
    axis off;%%%%no

 end
