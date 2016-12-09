#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "bmpproperty.h"
#include <time.h>
#include <vector>
using namespace std;
using namespace cv;
struct myLine
{
	double p;
	double alt;
	Point point;
	int maxx;
	int minx;
	int maxy;
	int miny;
	int count = 0;
};
uint minwidth=5,maxwidth=400,errorp=50 ;
Mat original;
bool comparel(const myLine & v1, const myLine & v2)
{
	return v1.point.x < v2.point.x;
}
bool comparev(const myLine & v1, const myLine & v2)
{
	return v1.point.y < v2.point.y;
}


void adaptiveThreshold2(InputArray _src, OutputArray _dst, double maxValue,
	int method, int type, int blockSize, double delta)
{
	Mat src = _src.getMat();
	CV_Assert(src.type() == CV_8UC1);
	CV_Assert(blockSize % 2 == 1 && blockSize > 1);  //blockSize����������ֵ�����������С,����������: 3, 5, 7, ...  
	Size size = src.size();

	_dst.create(size, src.type());
	Mat dst = _dst.getMat();

	if (maxValue < 0)
	{
		dst = Scalar(0);
		return;
	}

	Mat mean;

	if (src.data != dst.data)
		mean = dst;

	if (method == ADAPTIVE_THRESH_MEAN_C)
		boxFilter(src, mean, src.type(), Size(blockSize, blockSize),
		Point(-1, -1), true, BORDER_REPLICATE);
	else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
		GaussianBlur(src, mean, Size(blockSize, blockSize), 0, 0, BORDER_REPLICATE); //��˹ƽ��  
	else
		CV_Error(CV_StsBadFlag, "Unknown/unsupported adaptive threshold method");

	int i, j;
	uchar imaxval = saturate_cast<uchar>(maxValue);
	int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);
	uchar tab[768];

	if (type == CV_THRESH_BINARY)
	for (i = 0; i < 768; i++)
		tab[i] = (uchar)(i - 255 > -idelta ? imaxval : 0);
	else if (type == CV_THRESH_BINARY_INV)
	for (i = 0; i < 768; i++)
		tab[i] = (uchar)(i - 255 <= -idelta + 45 || i - 255 >= -idelta + 115 ? imaxval : 0);
	else
		CV_Error(CV_StsBadFlag, "Unknown/unsupported threshold type");

	if (src.isContinuous() && mean.isContinuous() && dst.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
	}

	for (i = 0; i < size.height; i++)
	{
		const uchar* sdata = src.data + src.step*i;
		const uchar* mdata = mean.data + mean.step*i;
		uchar* ddata = dst.data + dst.step*i;

		for (j = 0; j < size.width; j++)
			ddata[j] = tab[sdata[j] - mdata[j] + 255];
	}
}

void denoisy(const Mat src,Mat* dst)
{
	//�������
	uchar* sur = src.data;
	uint col = src.cols;
	uint row = src.rows;


    //���һ����ʱ�����鿽��
	uchar* res = new uchar[row*col];
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			res[i*col + j] = sur[i*col + j];


    //

		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
			{
				if (sur[i*col + j] == 255)
				{
					int k = j;
					while (sur[i*col + j] == 255 && j < col)
						j++;
					if (j - k > maxwidth || j - k < minwidth)
						for (int l = k; l < j; l++)
							res[i*col + l] = 0;
				}
			}
			for (int j = 0; j < col; j++)
			for (int i = 0; i < row; i++)
			{
				if (sur[i*col + j] == 255)
				{
					int k = i;
					while (sur[i*col + j] == 255 && i < row)
						i++;
					if (i - k > maxwidth || i - k < minwidth)
					for (int l = k; l < i; l++)
						res[l*col + j] = 0;
				}
			}


	
	*dst = Mat(row, col, CV_8U, (void*)res);
}
void AreaDetect(Mat src,int mode)
{
	//�������
	uchar* sur = src.data;
	uint col = src.cols;
	uint row = src.rows;

	//�������
	uint *im_label=new uint[col*row];  
	memset(im_label, 0, sizeof(uint)*col*row);

	uint *im_label3 = new uint[col*row];
	memset(im_label3, 0, sizeof(uint)*col*row);

	//��ջ����
	uint *st = new uint[65526];         
	memset(st, 0, sizeof(uint)*65536);    
	
	int label = 1;

	/*�ҳ�ÿ�����������򲢽��б��*/
	if (sur[0])
	{
		im_label[0] = label;
		st[label] = label;
		label++;
	}

	//���еĴ���                      
	for (int j = 1; j<col; j++)
	  if (sur[j])
		if (im_label[j - 1]) 
			im_label[j] = im_label[j - 1];
		else
		{
			im_label[j] = label;
			st[label] = label;
			label++;
		}

     
	 for (int i = 1; i < row; i++)
	  {

		 //ÿ�еĵ�һ����
		  if (sur[i*col]) 
		  if (sur[(i-1)*col])
			  im_label[i*col] = im_label[(i-1)*col];
		  else
		  {
			  im_label[i*col] = label;
			  st[label] = label;
			  label++;
		  }

		
		  for (int j = 1; j < col; j++)
		  if (sur[i*col + j])
		  {
			  if (!sur[i*col + j - 1] && !sur[(i - 1)*col + j])
			  {
				  im_label[i*col+j] = label;
				  st[label] = label;
				  label++;
			  }
			 else if (!sur[i*col + j - 1] && sur[(i - 1)*col + j])
				  im_label[i*col + j] = im_label[(i - 1)*col+j];
			 else if (sur[i*col + j - 1] && !sur[(i - 1)*col + j])
				  im_label[i*col + j] = im_label[i*col + j - 1];
			 else if (sur[i*col + j - 1] && sur[(i - 1)*col + j])
			  {
				   im_label[i*col + j] = im_label[i*col+j-1];
				   st[im_label[(i - 1)*col + j]] = im_label[i*col + j-1];			  
			  }
		  }		
	  }

	 for (int i = 0; i < row; i++)
	 for (int j = 0; j < col; j++)
	 {
		 int k;
		 k = st[im_label[i*col + j]];
		 while (st[k] != k)
			 k = st[st[k]];
		 im_label[i*col + j] = k;
	 }
	 
	 uchar *im_label2 = new uchar[col*row];
	 memset(im_label2, 0, sizeof(uchar)*col*row);
	 for (int i = 0; i < row; i++ )
	 for (int j = 0; j < col; j++)
	 if (sur[i*col + j]) im_label2[i*col + j] = im_label[i*col + j]%256 ;
	 Mat a = Mat(row, col, CV_8UC1, (uchar*)im_label2);
	 imshow("xs.jpg", a);
	 cout << label;

	 /*�ڶ����֣�ͼ������¹���*/





	uint *maxx = new uint[label];    memset(maxx, 0, label*sizeof(uint));
	uint *minx = new uint[label];    memset(minx, -1, label*sizeof(uint));
	uint *maxy = new uint[label];    memset(maxy, 0, label*sizeof(uint));
	uint *miny = new uint[label];    memset(miny, -1, label*sizeof(uint));





	int *u11 = new int[label];      memset(u11, 0, label*sizeof(int));
	int *u20 = new int[label];      memset(u20, 0, label*sizeof(int));
	int *u02 = new int[label];      memset(u02, 0, label*sizeof(int));
	double *tanu = new double[label];
	double *alt = new double[label]; memset(alt, 0, label*sizeof(double));
	double *p = new double[label];   memset(p, 0, label*sizeof(double));

	
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
		if (im_label[i*col + j]>0)
		{
			if (i > maxx[im_label[i*col + j]]) maxx[im_label[i*col + j]] = i;
			if (i < minx[im_label[i*col + j]]) minx[im_label[i*col + j]] = i;
			if (j > maxy[im_label[i*col + j]]) maxy[im_label[i*col + j]] = j;
			if (j < miny[im_label[i*col + j]]) miny[im_label[i*col + j]] = j;
		}

		for (int i = 1; i < label;i++)
		if (maxx[i] - minx[i]<row/2 &&  maxy[i] - miny[i]<col/2)
		if (maxx[i] - minx[i]>row/20 && maxy[i] - miny[i]>col/20)
		{

			int x1 = maxx[i], x2 = minx[i], y1 = maxy[i], y2 = miny[i];


			while (x1 > minx[i])
			{
				int count = 0;
				for (int j = miny[i]; j < maxy[i];j++)
					if (im_label[x1*col + j] == i) count++;
				if (count / (double)(maxy[i] - miny[i]) < 0.2)
					x1--;
				else break;
			}
			while (x2 <maxx[i])
			{
				int count = 0;
				for (int j = miny[i]; j < maxy[i]; j++)
				if (im_label[x2*col + j] == i) count++;
				if (count / (double)(maxy[i] - miny[i]) < 0.2)
					x2++;
				else break;
			}

			while (y1 > miny[i])
			{
				int count = 0;
				for (int j = minx[i]; j < maxx[i]; j++)
				if (im_label[j*col + y1] == i) count++;
				if (count / (double)(maxx[i] - minx[i]) < 0.2)
					y1--;
				else break;
			}

			while (y2 < maxy[i])
			{
				int count = 0;
				for (int j = minx[i]; j < maxx[i]; j++)
				if (im_label[j*col + y2] == i) count++;
				if (count / (double)(maxx[i] - minx[i]) < 0.2)
					y2++;
				else break;
			}


			for (int j = x2; j < x1; j++)
			for (int k = y2; k < y1; k++)
				im_label3[j*col + k] = i;
		}



				for (int i = 0; i < row; i++)
				for (int j = 0; j < col; j++)
				if (im_label3[i*col + j])
					im_label2[i*col + j] = 255; else im_label2[i*col + j] = 0;
				Mat b = Mat(row, col, CV_8UC1, (uchar*)im_label2);
			//	imshow("xts.jpg", b);
			//	imwrite("xts.jpg", b);

	/*���������ҵ�ÿ����������ĺ�����*/
         

	int *x = new int[label];        memset(x, 0, label*sizeof(int));
	int *y = new int[label];        memset(y, 0, label*sizeof(int));
	int *count = new int[label];    memset(count, 0, label*sizeof(int));

	for (int i = 0; i < row;i++)
	for (int j = 0; j < col;j++)
	if (im_label3[i*col + j]>0)
	{
		x[im_label3[i*col + j]] += i;
		y[im_label3[i*col + j]] += j;
		count[im_label3[i*col + j]]++;
	}



	for (int k = 0; k < label;k++)
		if (count[k])
		{
			x[k] = x[k] / count[k];
			y[k] = y[k] / count[k];
			Point p=Point(y[k],x[k]) ;
		//	circle(b, p, 2, Scalar(0, 0, 0),2);

		}
	

	
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
		if (im_label3[i*col + j] != 0)
		{
			u11[im_label3[i*col + j]] += (i - x[im_label3[i*col + j]])*(j - y[im_label3[i*col + j]]);
			u02[im_label3[i*col + j]] += (j - y[im_label3[i*col + j]])*(j - y[im_label3[i*col + j]]);
			u20[im_label3[i*col + j]] += (i - x[im_label3[i*col + j]])*(i - x[im_label3[i*col + j]]);
		}

	vector<myLine> vec;
	for (int i = 0; i < label; i++ )
	if (u11[i] != 0 || u20[i] != 0 || u02[i] != 0)
	{
		alt[i] = atan(2 * u11[i] / (double)(u20[i] - u02[i])) / 2;
		//Point p1, p2;
		//p1.x = y[i] + 50*cos(alt[i]);
		//p1.y = x[i] + 50*sin(alt[i]);
		//p2.x = y[i] - 50*cos(alt[i]);
		//p2.y = x[i] - 50*sin(alt[i]);
		p[i] = (y[i] * sin(alt[i]) + x[i] * cos(alt[i]));
		cout << i << ":" << p[i] << " " << alt[i] << endl;
		vector<myLine>::iterator j = vec.begin();
		int flag = 0;
		for (; j != vec.end();j++)
		if (abs((*j).p - p[i]) < errorp && abs((*j).alt - alt[i])<0.5)
		  {
			   (*j).p = ((*j).p*(*j).count + p[i]) / ((*j).count + 1);
			   (*j).alt = ((*j).alt*(*j).count + alt[i]) / ((*j).count + 1);
			   (*j).point.x = ((*j).point.x*(*j).count + x[i]) / ((*j).count + 1);
			   (*j).point.y = ((*j).point.y*(*j).count + y[i]) / ((*j).count + 1);
			   (*j).maxx = max((*j).maxx,(int)maxx[i]);
			   (*j).minx = min((*j).minx,(int)minx[i]);
			   (*j).maxy = max((*j).maxy, (int)maxy[i]);
			   (*j).miny = min((*j).miny, (int)miny[i]);
			   (*j).count++;
			   flag = 1;
			   break;
		  }
		   if (!flag)
		{
			myLine m;
			m.p = p[i];
			m.alt = alt[i];
			m.point.x = x[i];
			m.point.y = y[i];
			m.maxx = maxx[i];
			m.minx = minx[i];
			m.maxy = maxy[i];
			m.miny = miny[i];
			m.count = 1;
		//	cout << "OK" << m.point.x<<" " << m.point.y<<endl;
			vec.push_back(m);
		}


	//	line(b, p1, p2, Scalar(127, 127, 127));
		p[i] = (y[i] * cos(alt[i]) + x[i] * sin(alt[i]));
		alt[i] = alt[i] + CV_PI / 2;


		j = vec.begin();
		flag = 0;
		for (; j != vec.end(); j++)
		if (abs((*j).p - p[i]) < errorp && abs((*j).alt - alt[i])<0.5)
		{
			(*j).p = ((*j).p*(*j).count + p[i]) / ((*j).count + 1);
			(*j).alt = ((*j).alt*(*j).count + alt[i]) / ((*j).count + 1);
			(*j).point.x = ((*j).point.x*(*j).count + x[i]) / ((*j).count + 1);
			(*j).point.y = ((*j).point.y*(*j).count + y[i]) / ((*j).count + 1);
			(*j).maxx = max((*j).maxx, (int)maxx[i]);
			(*j).minx = min((*j).minx, (int)minx[i]);
			(*j).maxy = max((*j).maxy, (int)maxy[i]);
			(*j).miny = min((*j).miny, (int)miny[i]);
			(*j).count++;
			flag = 1;
			break;
		}
		if (!flag)
		{
			myLine m;
			m.p = p[i];
			m.alt = alt[i];
			m.point.x = x[i];
			m.point.y = y[i];
			m.maxx = maxx[i];
			m.minx = minx[i];
			m.maxy = maxy[i];
			m.miny = miny[i];
			m.count = 1;
	//		cout << "OK" << m.point.x << " " << m.point.y << endl;
			vec.push_back(m);
		}

		//p1.x = y[i] + 50 * cos(alt[i]);
		//p1.y = x[i] + 50 * sin(alt[i]);
		//p2.x = y[i] - 50 * cos(alt[i]);
		//p2.y = x[i] - 50 * sin(alt[i]);
		//line(b, p1, p2, Scalar(127, 127, 127));

		

	
	}
	for (vector<myLine>::iterator j = vec.begin(); j != vec.end(); j++)
	{
		Point p1, p2;
		p1.x = (*j).point.y + 500 * cos((*j).alt);
		p1.y = (*j).point.x + 500 * sin((*j).alt);
		p2.x = (*j).point.y - 500 * cos((*j).alt);
		p2.y = (*j).point.x - 500 * sin((*j).alt);
		//line(b, p1, p2, Scalar(127, 127, 127),2);
	}
//	imshow("xts2.jpg", b);
//	imwrite("xts2.jpg", b);

	vector<vector<myLine>> vline;
	for (vector<myLine>::iterator j = vec.begin(); j != vec.end(); j++)
	{
		
		int flag = 0;
		for (vector<vector<myLine>>::iterator k = vline.begin(); k != vline.end(); k++)
		{
			vector<myLine>::iterator l = (*k).begin();
			if (abs((*l).alt - (*j).alt) < 0.1)
			{
				(*k).push_back(*j);
				flag = 1;
			}
		}
		if (!flag)
		{
			vector<myLine> l;
			l.push_back(*j);
			vline.push_back(l);
		//	cout << "lk" << endl;
		}
	}
	
	for (vector<vector<myLine>>::iterator k = vline.begin(); k != vline.end(); k++)
	{
		sort((*k).begin(), (*k).end(), comparel);
		for (vector<myLine>::iterator l = (*k).begin(); l!= (*k).end(); l++)
		//	cout << ',' << (*l).point.x;
		cout << endl;
	}
	vector<myLine>::iterator it = (*(vline.begin())).begin();
	it++;
	for (; it != (*vline.begin()).end(); it++)
	{
		vector<myLine>::iterator l = --it;
		it++;
		Point p1, p2;
		p1.x = ((*it).point.y + (*l).point.y) / 2 + 700 * cos(((*it).alt + (*l).alt) / 2);
		p1.y = ((*it).point.x + (*l).point.x) / 2 + 700 * sin(((*it).alt + (*l).alt) / 2);
		p2.x = ((*it).point.y + (*l).point.y) / 2 - 700 * cos(((*it).alt + (*l).alt) / 2);
		p2.y = ((*it).point.x + (*l).point.x) / 2 - 700 * sin(((*it).alt + (*l).alt) / 2);
		line(b, p1, p2, Scalar(127, 127, 127),2);
		line(original, p1, p2, Scalar(0, 0, 255),3);
	}
	

	for (vector<vector<myLine>>::iterator k = vline.begin(); k != vline.end(); k++)
	{
		sort((*k).begin(), (*k).end(), comparev);
		for (vector<myLine>::iterator l = (*k).begin(); l != (*k).end(); l++)
	//		cout << ',' << (*l).point.y;
		cout << endl;
	}
	it = (*(++vline.begin())).begin();
	it++;
	for (; it != (*(++vline.begin())).end(); it++)
	{
		vector<myLine>::iterator l = --it;
		it++;
		Point p1, p2;
		p1.x = ((*it).point.y + (*l).point.y) / 2 + 700 * cos(((*it).alt + (*l).alt) / 2);
		p1.y = ((*it).point.x + (*l).point.x) / 2 + 700 * sin(((*it).alt + (*l).alt) / 2);
		p2.x = ((*it).point.y + (*l).point.y) / 2 - 700 * cos(((*it).alt + (*l).alt) / 2);
		p2.y = ((*it).point.x + (*l).point.x) / 2 - 700 * sin(((*it).alt + (*l).alt) / 2);
		line(b, p1, p2, Scalar(127, 127, 127),2);
		line(original, p1, p2, Scalar(0, 0, 255),3);
	}



	//Point p1 = Point(1, 2), p2 = Point(600, 500);
	//line(b, p1, p2, Scalar(0, 0, 255));
	//imshow("xtes.jpg", b);
	//imwrite("xtes.jpg", b);
	//imshow("result.jpg", original);
	imwrite("result.jpg", original);

/*	for (int i = 0; i < label; i++)
		if (u11[i] != 0 && count[i]>30)
		{
			tanu[i] = tan(atan(2 * u11[i] / (double)(u20[i] - u02[i])) / 2);
		    tanu[i] = tan(atan(tanu[i]) - 3.1415926 / 2);
			alt[i] = atan(tanu[i]) - 3.1415926 / 2;
			p[i] = x[i] * cos(alt[i]) + y[i] * sin(alt[i]);
			cout <<"("<<alt[i]<<","<<p[i]<<")"<< endl;
			for (int j = -100; j < 100; j++)
			if (tanu[i] < 1)
			{
				if (x[i] + j >= 0 && x[i] + j<row && y[i] + (int)(j*tanu[i])>=0 && y[i] + (int)(j*tanu[i])<col)
					im_label[(x[i] + j) * col + y[i] + (int)(j*tanu[i])] = 183;
			}
			else
			if (x[i] + (int)(j*(1 / tanu[i])) >= 0 && x[i] + (int)(j*(1 / tanu[i]))<row && y[i] + j >= 0 && y[i] + j<col)
				im_label[(x[i] + (int)(j*(1 / tanu[i]))) *col + y[i] + j] = 183;
		}
		Mat b = Mat(row, col, CV_8U, (void*)im_label);
		imwrite("30.bmp", b);
        /*detect the line which is collineation*/
/*		struct Mline
		{
			int avg;
			int x;
			int y;
			int count;
			double gra;
			Point2f a;
			Point2f b;
			int minx;
			int maxx;
			int miny;
			int maxy;
		};

		Mline *al = new Mline[label];    memset(al, 0, label*sizeof(Mline));
		int *la = new int[label];    memset(la, 0, label*sizeof(int));
		int k = -1;
		for (int i = 0; i < label; i++)
		{
			if (p[i])
			{
				int j = 0;
				while (j <= k)
				{
					if (abs(al[j].avg/al[j].count - p[i]) < 3)
					{
						la[i] = j;
						al[j].maxx = max(al[k].maxx, (int)maxx[i]);
						al[j].maxy = max(al[k].maxy, (int)maxy[i]);
						al[k].minx = min(al[k].minx, (int)minx[i]);
						al[k].miny = min(al[k].miny, (int)miny[i]);
						al[j].avg += p[i];
						al[j].gra += alt[i];
						al[j].x += x[i];
						al[j].y += y[i];
						al[j].count++;
						break;
					}
					j++;
				}
				if (j > k) 
				{
					k++;
					al[k].maxx=maxx[i];
					al[k].maxy=maxy[i];
					al[k].minx=minx[i];
					al[k].miny=miny[i];
					al[k].avg = p[i];
					al[k].x = x[i];
					al[k].y = y[i];
					al[k].gra = alt[i];
					al[k].count = 1;
					la[i] = k;
				}
			}
		}
		for (int i = 0; i < label; i++)
		if (p[i])
			cout << "(" << p[i] << "," << la[i]<<","<<alt[i] << ")"<<endl;
		for (int i = 0; i <= k; i++)
		{
			al[i].x = al[i].x/al[i].count;
			al[i].y = al[i].y/al[i].count;
			if (!mode)
			{
				al[i].gra = al[i].gra / al[i].count;
				al[i].a.x = al[i].y - sin(al[i].gra) * 100;
				al[i].a.y = al[i].x - cos(al[i].gra) * 100;
				al[i].b.x = al[i].y + sin(al[i].gra) * 100;
				al[i].b.y = al[i].x + cos(al[i].gra) * 100;
			}
			else
			{
				al[i].gra = al[i].gra / al[i].count + 3.1415926 / 2;
				al[i].a.x = al[i].miny;
				al[i].a.y = al[i].x - cos(al[i].gra) * (al[i].y - al[i].miny) / sin(al[i].gra);
				al[i].b.x = al[i].maxy;
				al[i].b.y = al[i].x + cos(al[i].gra) * (al[i].maxy - al[i].y) / sin(al[i].gra);
			}
			cout << "(" << al[i].minx << "," << al[i].maxx << ")" << endl;
			line(original, al[i].b, al[i].a, Scalar(0, 135, 0),minwidth);
		}
		imshow("1", original);*/
            		 
}
int main()
{
	original=imread("29.jpg", 1);
	Mat original = imread("29.jpg", 0);
	Mat imthreshold,imdenoisyx,imdenoisyy;
	int blockSize =99;
	int constValue =45; 
	//Mat local;
	adaptiveThreshold2(original, imthreshold, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, blockSize, constValue);
	//
	clock_t start,finish;
	start = clock();
	//imshow("Threshold.bmp", imthreshold);
	denoisy(imthreshold,&imdenoisyx);
	denoisy(imdenoisyx, &imdenoisyx);
	//imshow("afterdn.jpg", imdenoisyx);
	denoisy(imthreshold, &imdenoisyy);
	//imshow("close3.jpg", imdenoisyy);
	//morphologyEx(imthreshold, imthreshold, MORPH_CLOSE, Mat(5, 5, CV_8U), Point(-1, -1), 1);
	//morphologyEx(imdenoisyx, imdenoisyx, MORPH_OPEN, Mat(5, 5, CV_8U), Point(-1, -1), 1);
	morphologyEx(imdenoisyx, imdenoisyx, MORPH_CLOSE, Mat(1,1, CV_8U), Point(-1, -1), 1);
	//morphologyEx(imdenoisyx, imdenoisyx, MORPH_CLOSE, Mat(5, 5, CV_8U), Point(-1, -1), 1);
	//imshow("close.jpg", imdenoisyx);
	AreaDetect(imdenoisyx, 0);
	finish = clock();
	double k = (double)(start - finish);
	cout << "clock" << k << endl;;
	//imwrite("1.bmp", imdenoisyx);
	//denoisy(imthreshold, &imdenoisyy,1);
	//imwrite("0.bmp", imdenoisyy);
	//double startTime=clock();
	//AreaDetect(imdenoisyy,1);
	//double endTime = clock();
	//cout <<endTime-startTime << "ms" << endl;
	waitKey(0);
	return 0;
}