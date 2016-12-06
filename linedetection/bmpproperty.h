#ifndef BMP_PROPERTY_H
#define BMP_PROPERTY_H 

typedef struct 
{   //unsigned short    bfType;//˵���ļ������ͣ���ֵ������0x4D42��ʮ������ASCII��4D����B����42����M������Ҳ�����ַ�'BM',2�ֽ�. 
	//ע��bfType��Ϊ���ǽṹ���ֽڶ��루4�ֽڶ��룩��������������ͷ�ļ�ʱ�����ֽڶ����ԭ���ʹ�ṹ���������ֽڡ�
    unsigned long    bfSize; // λͼ�ļ��Ĵ�С�����ֽ�Ϊ��λ,4�ֽ�
    unsigned short    bfReserved1;  // λͼ�ļ������֣�����Ϊ0,2�ֽ�
    unsigned short    bfReserved2;  // λͼ�ļ������֣�����Ϊ0,2�ֽ�
    unsigned long    bfOffBits;  // λͼ���ݵ���ʼλ�ã��������λͼ,4�ֽ�
} BitMapFileHeader;						//12���ֽ�				

typedef struct  
{  
    unsigned long  biSize;   // ˵��BITMAPINFOHEADER�ṹ����Ҫ������,4�ֽڡ�
    long   biWidth;   // λͼ�Ŀ�ȣ�������Ϊ��λ��4�ֽ�
    long   biHeight;  // λͼ�ĸ߶ȣ�������Ϊ��λ��4�ֽ�
    unsigned short   biPlanes;   // Ŀ���豸�ļ��𣬱���Ϊ1��2�ֽ�
    unsigned short   biBitCount;  // ÿ�����������λ����������1(˫ɫ)��4(16ɫ)��8(256ɫ)��24(���ɫ)֮һ��2�ֽ�
    unsigned long  biCompression;   // λͼѹ�����ͣ������� 0(��ѹ��)��1(BI_RLE8ѹ������)��2(BI_RLE4ѹ������)֮һ��4�ֽ�
    unsigned long  biSizeImage;   // λͼ���ݵĴ�С�����ֽ�Ϊ��λ��4�ֽ�
    long   biXPelsPerMeter;   // λͼˮƽ�ֱ��ʣ�ÿ����������4�ֽ�
    long   biYPelsPerMeter;   // λͼ��ֱ�ֱ��ʣ�ÿ����������4�ֽ�
    unsigned long   biClrUsed;   // λͼʵ��ʹ�õ���ɫ���е���ɫ����4�ֽ�
    unsigned long   biClrImportant;   // λͼ��ʾ��������Ҫ����ɫ����4�ֽ�
} BitMapInfoHeader;						//40���ֽ�

typedef struct   
{  
    unsigned char rgbBlue; //����ɫ����ɫ����  
    unsigned char rgbGreen; //����ɫ����ɫ����  
    unsigned char rgbRed; //����ɫ�ĺ�ɫ����  
    unsigned char rgbReserved; //����ֵ,Ϊ0  
} RgbQuad;  //4�ֽ�

unsigned short Bmtype = 0x4d42;
BitMapFileHeader bitMapFileHeader;
BitMapInfoHeader bitMapInfoHeader;

#endif