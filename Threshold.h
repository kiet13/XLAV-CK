#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
using namespace std;
using namespace cv;

//phân ngưỡng tĩnh
class StaticThreshold
{
	//ngưỡng dưới
	int _lowThreshold;
	//ngưỡng trên
	int _highThreshold;
public:
	/*
	Hàm áp dụng phân ngưỡng tĩnh
	- srcImage: ảnh input
	- dstImage: ảnh kết quả	
	Hàm trả về
	1: nếu phân ngưỡng thành công
	0: nếu phân ngưỡng không thành công
	*/
	int Apply(const Mat& srcImage, Mat &dstImage);


	StaticThreshold();
	StaticThreshold(int lowThreshold, int highThreshold);
	~StaticThreshold();
};

StaticThreshold::StaticThreshold()
{
	_lowThreshold = _highThreshold = 0;
}

StaticThreshold::StaticThreshold(int lowThreshold, int highThreshold)
{
	if (lowThreshold < 0 || lowThreshold > 255)
		_lowThreshold = 0;

	if (highThreshold < 0 || highThreshold > 255)
		_highThreshold = 255;

	if (highThreshold < lowThreshold)
		_lowThreshold = _highThreshold = 0;

	_lowThreshold = lowThreshold;
	_highThreshold = highThreshold;
}

StaticThreshold::~StaticThreshold()
{}

int StaticThreshold::Apply(const Mat& srcImage, Mat& dstImage)
{
	if (srcImage.empty() == true || srcImage.isContinuous() == false)
		return 0;

	// Ảnh màu cần chuyển đổi về ảnh xám trước khi phân đoạn
	if (srcImage.channels() != 1)
		return 0;

	dstImage = srcImage.clone();
	for (int r = 0; r < srcImage.rows; r++)
	{
		for (int c = 0; c < srcImage.cols; c++)
		{
			int value = srcImage.at<uchar>(r, c);
			if (value > _lowThreshold && value < _highThreshold)
				dstImage.at<uchar>(r, c) = 255;
			else
				dstImage.at<uchar>(r, c) = 0;
		}
	}
	return 1;
}


//phân ngưỡng cục bộ dựa vào trung bình
class AverageLocalThreshold
{
	//hệ số C
	int _C;

public:
	/*
	Hàm áp dụng phân ngưỡng cục bộ theo trung bình
	- srcImage: ảnh input
	- dstImage: ảnh kết quả
	- winSize: kích thước lân cận
	Hàm trả về
		1: nếu phân ngưỡng thành công
		0: nếu phân ngưỡng không thành công
	*/

	int Apply(const Mat& srcImage, Mat &dstImage, Size winSize);


	AverageLocalThreshold();
	AverageLocalThreshold(int C);
	~AverageLocalThreshold();
};
AverageLocalThreshold::AverageLocalThreshold()
{
	_C = 0;
}

AverageLocalThreshold::AverageLocalThreshold(int C)
{
	_C = C;
}

AverageLocalThreshold::~AverageLocalThreshold()
{}

int AverageLocalThreshold::Apply(const Mat& srcImage, Mat &dstImage, Size winSize)
{
	if (srcImage.empty() == true || srcImage.isContinuous() == false)
		return 0;

	// Ảnh màu cần chuyển đổi về ảnh xám trước khi phân đoạn
	if (srcImage.channels() != 1)
		return 0;

	dstImage = srcImage.clone();
	for (int r = 0; r < srcImage.rows; r++)
	{
		for (int c = 0; c < srcImage.cols; c++)
		{
			int value = srcImage.at<uchar>(r, c);
			vector<uchar> neightborValues;

			int halfWidth = winSize.width >> 1;
			int halfHeight = winSize.height >> 1;
			float mean = 0;

			// Calcualte mean of neightbor values
			for (int i = -halfHeight; i <= halfHeight; i++)
			{
				for (int j = -halfWidth; j <= halfWidth; j++)
				{
					if (r + i >= 0 && r + i < srcImage.rows
						&& c + j >= 0 && c + j < srcImage.cols)
					{
						uchar neightbor = srcImage.at<uchar>(r + i, c + j);
						mean += neightbor;
						neightborValues.push_back(neightbor);
					}
				}
			}
			size_t size = neightborValues.size();
			mean /= size;

			float myThreshold = mean - _C;
			if (myThreshold < 0)
				myThreshold = 0;

			if (myThreshold > 255)
				myThreshold = 255;
			if ((float)value > myThreshold)
				dstImage.at<uchar>(r, c) = 255;
			else
				dstImage.at<uchar>(r, c) = 0;
		}
	}
	return 1;
}

//phân ngưỡng cục bộ dựa vào trung vị
class MedianLocalThreshold
{
	//hệ số C
	int _C;

public:
	/*
	Hàm áp dụng phân ngưỡng cục bộ dựa vào trung vị
	- srcImage: ảnh input
	- dstImage: ảnh kết quả
	- winSize: kích thước lân cận
	Hàm trả về
	1: nếu phân ngưỡng thành công
	0: nếu phân ngưỡng không thành công
	*/

	int Apply(const Mat& srcImage, Mat &dstImage, Size winSize);


	MedianLocalThreshold();
	~MedianLocalThreshold();
};

//phân ngưỡng cục bộ dựa vào thuật toán Sauvola
class SauvolaLocalThreshold
{
	//hệ số r
	int _r;
	//hệ số k
	float _k;
public:
	/*
	Hàm áp dụng thuật toán Sauvola để phân ngưỡng
	- srcImage: ảnh input
	- dstImage: ảnh kết quả
	- winSize: kích thước lân cận
	Hàm trả về
	1: nếu phân ngưỡng thành công
	0: nếu phân ngưỡng không thành công
	*/

	int Apply(const Mat& srcImage, Mat &dstImage, Size winSize);


	SauvolaLocalThreshold();
	SauvolaLocalThreshold(int r, float k);
	~SauvolaLocalThreshold();
};

SauvolaLocalThreshold::SauvolaLocalThreshold()
{}

SauvolaLocalThreshold::SauvolaLocalThreshold(int r, float k)
{
	if (r < 0 || r > 255)
		r = 128;

	if (k < 0 || k > 1)
		k = 0.5;

	_r = r;
	_k = k;
}

SauvolaLocalThreshold::~SauvolaLocalThreshold()
{}

int SauvolaLocalThreshold::Apply(const Mat& srcImage, Mat &dstImage, Size winSize)
{
	if (srcImage.empty() == true || srcImage.isContinuous() == false)
		return 0;

	// Ảnh màu cần chuyển đổi về ảnh xám trước khi phân đoạn
	if (srcImage.channels() != 1)
		return 0;

	dstImage = srcImage.clone();
	for (int r = 0; r < srcImage.rows; r++)
	{
		for (int c = 0; c < srcImage.cols; c++)
		{
			int value = srcImage.at<uchar>(r, c);
			vector<uchar> neightborValues;

			int halfWidth = winSize.width >> 1;
			int halfHeight = winSize.height >> 1;
			float mean = 0;

			// Calcualte mean of neightbor values
			for (int i = -halfHeight; i <= halfHeight; i++)
			{
				for (int j = -halfWidth; j <= halfWidth; j++)
				{
					if (r + i >= 0 && r + i < srcImage.rows
						&& c + j >= 0 && c + j < srcImage.cols)
					{
						uchar neightbor = srcImage.at<uchar>(r + i, c + j);
						mean += neightbor;
						neightborValues.push_back(neightbor);
					}
				}
			}
			size_t size = neightborValues.size();
			mean /= size;

			float std = 0;
			// Calculate std of neightbor values
			for (int i = 0; i < size; i++)
			{
				uchar neightborValue = neightborValues[i];
				std += (neightborValue - mean) * (neightborValue - mean);
			}
			std = sqrtf(std / (size - 1));

			// Determine value of binary image
			float myThreshold = mean * (1 + _k * (std / _r - 1));
			if ((float)value > myThreshold)
				dstImage.at<uchar>(r, c) = 255;
			else
				dstImage.at<uchar>(r, c) = 0;
		}
	}

	//uchar* pData = (uchar*)srcImage.data;
	//size_t widthstep = srcImage.step[0];
	//uchar* pRow;
	//for (int r = 0; r < srcImage.rows; r++)
	//{
	//	pRow = pData + r * widthstep;
	//	for (int c = 0; c < srcImage.cols; c++)
	//	{
	//		pRow += c;
	//		uchar value = *pRow;
	//		vector<uchar> neightborValues;

	//		int halfWidth = winSize.width >> 1;
	//		int halfHeight = winSize.height >> 1;
	//		float mean = 0;

	//		// Calcualte mean of neightbor values
	//		for (int i = -halfHeight; i <= halfHeight; i++)
	//		{
	//			for (int j = -halfWidth; j <= halfWidth; j++)
	//			{
	//				if (r + i >= 0 && r + i < srcImage.rows
	//					&& c + j >= 0 && c + j < srcImage.cols)
	//				{
	//					uchar neightbor = pRow[i * widthstep + j];
	//					mean += neightbor;
	//					neightborValues.push_back(neightbor);
	//				}
	//			}
	//		}

	//		size_t size = neightborValues.size();
	//		mean /= size;

	//		float std = 0;
	//		// Calculate std of neightbor values
	//		for (int i = 0; i < size; i++)
	//		{
	//			uchar neightborValue = neightborValues[i];
	//			std += (neightborValue - mean) * (neightborValue - mean);
	//		}
	//		std = sqrtf(std / (size - 1));

	//		// Determine value of binary image
	//		float myThreshold = mean * (1 + _k * (std / _r - 1));
	//		if ((float)value > myThreshold)
	//			dstImage.at<uchar>(r, c) = 255;
	//		else
	//			dstImage.at<uchar>(r, c) = 0;
	//	}
	//	// pData += widthstep;
	//}
	return 1;
}

