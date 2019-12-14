#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
# define M_PI           3.14159265358979323846
#include <vector>

class CannyEdgeDetector
{
	//ngưỡng dưới
	int _lowThreshold;
	//ngưỡng trên
	int _highThreshold;

public:
	/*
		Hàm áp dụng thuật toán Canny để phát hiện biên cạnh
		- srcImage: ảnh input
		- dstImage: ảnh kết quả
		Hàm trả về
			1: nếu detect thành công
			0: nếu detect không thành công
	*/


	int Apply(const Mat& srcImage, Mat &dstImage);

	CannyEdgeDetector();
	CannyEdgeDetector(int low, int high);
	~CannyEdgeDetector();
};

CannyEdgeDetector::CannyEdgeDetector()
{
	_lowThreshold = 0;
	_highThreshold = 255;
}

CannyEdgeDetector::CannyEdgeDetector(int low, int high)
{
	if (low < 0)
		_lowThreshold = 0;
	else if (low > 255)
		_lowThreshold = 255;
	else _lowThreshold = low;

	if (high < 0)
		_highThreshold = 0;
	else if (high > 255)
		_highThreshold = 255;
	else _highThreshold = high;
	
}

CannyEdgeDetector::~CannyEdgeDetector() {};

vector<vector<double>> createFilter(int row, int column, double sigmaIn)
{
	vector<vector<double>> filter;

	for (int i = 0; i < row; i++)
	{
		vector<double> col;
		for (int j = 0; j < column; j++)
		{
			col.push_back(-1);
		}
		filter.push_back(col);
	}

	float coordSum = 0;
	float constant = 2.0 * sigmaIn * sigmaIn;
	float sum = 0.0;

	for (int x = -row / 2; x <= row / 2; x++)
	{
		for (int y = -column / 2; y <= column / 2; y++)
		{
			coordSum = (x * x + y * y);
			filter[x + row / 2][y + column / 2] = (exp(-(coordSum) / constant)) / (M_PI * constant);
			sum += filter[x + row / 2][y + column / 2];
		}
	}

	for (int i = 0; i < row; i++)
		for (int j = 0; j < column; j++)
			filter[i][j] /= sum;

	return filter;

}
Mat useFilter(Mat srcImage, vector<vector<double>> filterIn)
{
	int size = (int)filterIn.size() / 2;
	Mat filteredImg = Mat(srcImage.rows - 2 * size, srcImage.cols - 2 * size, CV_8UC1);
	for (int i = size; i < srcImage.rows - size; i++)
	{
		for (int j = size; j < srcImage.cols - size; j++)
		{
			double sum = 0;

			for (int x = 0; x < filterIn.size(); x++)
				for (int y = 0; y < filterIn.size(); y++)
				{
					sum += filterIn[x][y] * (double)(srcImage.at<uchar>(i + x - size, j + y - size));
				}

			filteredImg.at<uchar>(i - size, j - size) = sum;
		}

	}
	return filteredImg;
}

int CannyEdgeDetector::Apply(const Mat& srcImage, Mat& dstImage)
{
	if (srcImage.empty() == true || srcImage.isContinuous() == false)
		return 0;

	//Grayscale
		dstImage = Mat(srcImage.rows, srcImage.cols, CV_8UC1);
		for (int i = 0; i < srcImage.rows; i++)
			for (int j = 0; j < srcImage.cols; j++)
			{
				int b = srcImage.at<Vec3b>(i, j)[0];
				int g = srcImage.at<Vec3b>(i, j)[1];
				int r = srcImage.at<Vec3b>(i, j)[3];
					
				double newValue = (r * 0.2126 + g * 0.7152 + b * 0.0722);
				dstImage.at<uchar>(i, j) = newValue;
			}

	//Gaussian filter (3x3)
		vector<vector<double>> filter = createFilter(3, 3, 1);
		useFilter(dstImage, filter);
		

	Mat angles;
	//Sobel filter
	{
		//Sobel X Filter
		double x1[] = { -1.0, 0, 1.0 };
		double x2[] = { -2.0, 0, 2.0 };
		double x3[] = { -1.0, 0, 1.0 };

		vector<vector<double>> xFilter(3);
		xFilter[0].assign(x1, x1 + 3);
		xFilter[1].assign(x2, x2 + 3);
		xFilter[2].assign(x3, x3 + 3);

		//Sobel Y Filter
		double y1[] = { 1.0, 2.0, 1.0 };
		double y2[] = { 0, 0, 0 };
		double y3[] = { -1.0, -2.0, -1.0 };
	
		vector<vector<double>> yFilter(3);
		yFilter[0].assign(y1, y1 + 3);
		yFilter[1].assign(y2, y2 + 3);
		yFilter[2].assign(y3, y3 + 3);

		//Limit Size
		int size = (int)xFilter.size() / 2;
		
		Mat sobel = Mat(dstImage.rows - 2 * size, dstImage.cols - 2 * size, CV_8UC1);

		angles = Mat(dstImage.rows - 2 * size, dstImage.cols - 2 * size, CV_32FC1); //AngleMap
		
		for (int i = size; i < dstImage.rows - size; i++)
		{
			for (int j = size; j < dstImage.cols - size; j++)
			{
				double sumx = 0, sumy = 0;
				for (int x = 0; x < xFilter.size(); x++)
					for (int y = 0; y < yFilter.size(); y++)
					{
						sumx += xFilter[x][y] * (double)(dstImage.at<uchar>(i + x - size, j + y - size)); //Sobel_X Filter Value
						sumx += yFilter[x][y] * (double)(dstImage.at<uchar>(i + x - size, j + y - size)); //Sobel_X Filter Value
					}

				double sumxsq = sumx * sumx;
				double sumysq = sumy * sumy;

				double sq2 = sqrt(sumxsq + sumysq);
				if (sq2 > 255)
					sq2 = 255;
				sobel.at<uchar>(i - size, j - size) = sq2;

				if (sumx == 0)
					angles.at<float>(i - size, j - size) = 90;
				else
					angles.at<float>(i - size, j - size) = atan(sumy / sumx)*180/M_PI;
			}
		}
		dstImage = sobel;
	}

	//Non-maximum suppression
	{
		Mat nonMaxSupp = Mat(dstImage.rows - 2, dstImage.cols - 2, CV_8UC1);
		for (int i =1;i<dstImage.rows-1;i++)
			for (int j = 1; j < dstImage.cols - 1; j++)
			{
				float Tangent = angles.at<float>(i, j);

				nonMaxSupp.at<uchar>(i - 1, j - 1) = dstImage.at<uchar>(i, j);
				//Horizontal Edge
				if (((-22.5 < Tangent) && (Tangent <= 22.5)) || ((157.5 < Tangent) && (Tangent <= -157.5)))
				{
					if ((dstImage.at<uchar>(i, j) < dstImage.at<uchar>(i, j + 1)) || (dstImage.at<uchar>(i, j) < dstImage.at<uchar>(i, j - 1)))
						nonMaxSupp.at<uchar>(i - 1, j - 1) = 0;
				}

				//Vertical Edge
				if (((-112.5 < Tangent) && (Tangent <= -67.5)) || ((67.5 < Tangent) && (Tangent <= 112.5)))
				{
					if ((dstImage.at<uchar>(i, j) < dstImage.at<uchar>(i + 1, j)) || (dstImage.at<uchar>(i, j) < dstImage.at<uchar>(i - 1, j)))
						nonMaxSupp.at<uchar>(i - 1, j - 1) = 0;
				}

				//-45 Degree Edge
				if (((-67.5 < Tangent) && (Tangent <= -22.5)) || ((112.5 < Tangent) && (Tangent <= 157.5)))
				{
					if ((dstImage.at<uchar>(i, j) < dstImage.at<uchar>(i - 1, j + 1)) || (dstImage.at<uchar>(i, j) < dstImage.at<uchar>(i + 1, j - 1)))
						nonMaxSupp.at<uchar>(i - 1, j - 1) = 0;
				}

				//45 Degree Edge
				if (((-157.5 < Tangent) && (Tangent <= -112.5)) || ((22.5 < Tangent) && (Tangent <= 67.5)))
				{
					if ((dstImage.at<uchar>(i, j) < dstImage.at<uchar>(i + 1, j + 1)) || (dstImage.at<uchar>(i, j) < dstImage.at<uchar>(i - 1, j - 1)))
						nonMaxSupp.at<uchar>(i - 1, j - 1) = 0;
				}
			}
		dstImage = nonMaxSupp;
	}

	//Double threshold & Hysteresis
	{
		if (_lowThreshold > 255)
			_lowThreshold = 255;
		if (_highThreshold > 255)
			_highThreshold = 255;

		for (int i = 0; i < dstImage.rows; i++)
		{
			for (int j = 0; j < dstImage.cols; j++)
			{
				if (dstImage.at<uchar>(i, j) > _highThreshold)
					dstImage.at<uchar>(i, j) = 255;
				else if (dstImage.at<uchar>(i, j) < _lowThreshold)
					dstImage.at<uchar>(i, j) = 0;
				else
				{
					bool anyHigh = false;
					bool anyBetween = false;
					for (int x = i - 1; x < i + 2; x++)
					{
						for (int y = j - 1; y < j + 2; y++)
						{
							if (x <= 0 || y <= 0 || x > dstImage.rows || y > dstImage.cols) //Out of bounds
								continue;
							else
							{
								if (dstImage.at<uchar>(x, y) > _highThreshold)
								{
									dstImage.at<uchar>(i, j) = 255;
									anyHigh = true;
									break;
								}
								else if (dstImage.at<uchar>(x, y) <= _highThreshold && dstImage.at<uchar>(x, y) >= _lowThreshold)
									anyBetween = true;
							}
						}
						if (anyHigh)
							break;
					}
					if (!anyHigh && anyBetween)
						for (int x = i - 2; x < i + 3; x++)
						{
							for (int y = j - 2; y < j + 3; y++)
							{
								if (x < 0 || y < 0 || x > dstImage.rows || y > dstImage.cols) //Out of bounds
									continue;
								else
								{
									if (dstImage.at<uchar>(x, y) > _highThreshold)
									{
										dstImage.at<uchar>(i, j) = 255;
										anyHigh = true;
										break;
									}
								}
							}
							if (anyHigh)
								break;
						}
					if (!anyHigh)
						dstImage.at<uchar>(i, j) = 0;
				}
			}
		}
	}


	return 1;
}
