#ifndef BMP_PROPERTY_H
#define BMP_PROPERTY_H 

typedef struct 
{   //unsigned short    bfType;//说明文件的类型，该值必需是0x4D42（十六进制ASCII码4D代表“B”，42代表“M”），也就是字符'BM',2字节. 
	//注销bfType是为了是结构体字节对齐（4字节对齐），否则在最后存入头文件时由于字节对齐的原因会使结构体多出两个字节。
    unsigned long    bfSize; // 位图文件的大小，以字节为单位,4字节
    unsigned short    bfReserved1;  // 位图文件保留字，必须为0,2字节
    unsigned short    bfReserved2;  // 位图文件保留字，必须为0,2字节
    unsigned long    bfOffBits;  // 位图数据的起始位置，以相对于位图,4字节
} BitMapFileHeader;						//12个字节				

typedef struct  
{  
    unsigned long  biSize;   // 说明BITMAPINFOHEADER结构所需要的字数,4字节。
    long   biWidth;   // 位图的宽度，以像素为单位，4字节
    long   biHeight;  // 位图的高度，以像素为单位，4字节
    unsigned short   biPlanes;   // 目标设备的级别，必须为1，2字节
    unsigned short   biBitCount;  // 每个像素所需的位数，必须是1(双色)，4(16色)，8(256色)或24(真彩色)之一，2字节
    unsigned long  biCompression;   // 位图压缩类型，必须是 0(不压缩)，1(BI_RLE8压缩类型)或2(BI_RLE4压缩类型)之一，4字节
    unsigned long  biSizeImage;   // 位图数据的大小，以字节为单位，4字节
    long   biXPelsPerMeter;   // 位图水平分辨率，每米像素数，4字节
    long   biYPelsPerMeter;   // 位图垂直分辨率，每米像素数，4字节
    unsigned long   biClrUsed;   // 位图实际使用的颜色表中的颜色数，4字节
    unsigned long   biClrImportant;   // 位图显示过程中重要的颜色数，4字节
} BitMapInfoHeader;						//40个字节

typedef struct   
{  
    unsigned char rgbBlue; //该颜色的蓝色分量  
    unsigned char rgbGreen; //该颜色的绿色分量  
    unsigned char rgbRed; //该颜色的红色分量  
    unsigned char rgbReserved; //保留值,为0  
} RgbQuad;  //4字节

unsigned short Bmtype = 0x4d42;
BitMapFileHeader bitMapFileHeader;
BitMapInfoHeader bitMapInfoHeader;

#endif