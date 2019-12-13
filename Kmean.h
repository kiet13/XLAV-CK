#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
using namespace std;
using namespace cv;


float CalcEuclidDistance(vector<float> vct1, vector<float> vct2)
{
	float result = 0;
	for (int i = 0; i < vct1.size(); i++)
		result += (vct1[i] - vct2[i]) * (vct1[i] - vct2[i]);
	result = sqrtf(result);
	return result;
}

class Kmean
{
	//số cụm K
	int _numClusters;	
	vector<vector<float>> _centroids;
	bool _hasCoverged;
	float _threshold;
	int _maxIter;

	// Hàm khởi tạo các centroid với N là số pixel trên một ảnh (N = width * height)
	int InitCentroids(uchar* pData, int N, int channels);

	// Hàm gán cụm cho các điểm ảnh
	void AssignLabels(uchar* pData, int N, int channels, uchar* labels);

	// Hàm cập nhật lại các centroids
	void UpdateCentroids(uchar* pData, int N, int channels, uchar* labels);

public:
	/*
	Hàm áp dụng thuật toán Kmeans để phân đoạn ảnh
	- srcImage: ảnh input
	- dstImage: ảnh kết quả
	Hàm trả về
	1: nếu phân đoạn thành công
	0: nếu phân đoạn không thành công
	*/

	int Apply(const Mat& srcImage, Mat &dstImage);

	Kmean();
	Kmean(int k);
	~Kmean();
};

Kmean::Kmean()
{
	_numClusters = 2;
	_hasCoverged = false;
	_threshold = (float)0.001;
	_maxIter = 300;
}

Kmean::Kmean(int k)
{
	k = (k > 2) ? k : 2;
	_numClusters = k;
	_hasCoverged = false;
	_threshold = (float)0.001;
	_maxIter = 300;
}


Kmean::~Kmean()
{}

bool isContain(vector<int> list, int x)
{
	for (int i = 0; i < list.size(); i++)
		if (x == list[i])
			return true;
	return false;
}

int Kmean::InitCentroids(uchar* pData, int N, int channels)
{
	if (N <= 0)
		return 0;

	int centroidIndex;
	vector<int> listCentroidIdx;
	for (int i = 0; i < _numClusters; i++)
	{
		do
		{
			// generate a centroidIndex without duplicate
			centroidIndex = rand() % N;
		} while (isContain(listCentroidIdx, N));

		listCentroidIdx.push_back(centroidIndex);
		vector<float> centroid;
		for (int i = 0; i < channels; i++)
			centroid.push_back(pData[channels * centroidIndex + i]);
		_centroids.push_back(centroid);
	}
	return 1;
}

void Kmean::AssignLabels(uchar* pData, int N, int channels, uchar* labels)
{
	for (int i = 0; i < N; i++)
	{
		int idxMinDist = 0;
		
		vector<float> pixel = { (float)pData[i * channels], 
			(float)pData[i * channels + 1], (float)pData[i * channels + 2] };

		float dist = CalcEuclidDistance(pixel, _centroids[0]);
		float minDist = dist;
		for (int j = 1; j < _numClusters; j++)
		{
			dist = CalcEuclidDistance(pixel, _centroids[j]);
			if (dist < minDist)
			{
				minDist = dist;
				idxMinDist = j;
			}
		}

		labels[i] = idxMinDist;
	}
}

void Kmean::UpdateCentroids(uchar* pData, int N, int channels, uchar* labels)
{
	_hasCoverged = true;
	
	// Initialize newCentroids with K vector (0, 0, ...)
	vector<vector<float>> newCentroids;
	for (int i = 0; i < _numClusters; i++)
	{
		vector<float> vct;
		for (int i = 0; i < channels; i++)
			vct.push_back(0);
		newCentroids.push_back(vct);
	}

	// Calc new centroids each cluster
	int* nEachCluster = new int [_numClusters] {0};
	for (int i = 0; i < N; i++)
	{
		nEachCluster[labels[i]]++;
		for (int j = 0; j < channels; j++)
			newCentroids[labels[i]][j] += 
			(pData[i*channels + j] - newCentroids[labels[i]][j]) / nEachCluster[labels[i]];
	}

	vector<float> vec0 = { 0.0f, 0.0f, 0.0f };
	float d;
	d = CalcEuclidDistance(newCentroids[0], _centroids[0]) / CalcEuclidDistance(_centroids[0], vec0);

	if (d < _threshold)
	{
		_centroids = newCentroids;
		_hasCoverged = false;
	}
}



int Kmean::Apply(const Mat& srcImage, Mat &dstImage)
{
	if (srcImage.empty() == true || srcImage.isContinuous() == false)
		return 0;

	dstImage = srcImage.clone();
	int nChannels = srcImage.channels();
	int nc = srcImage.cols * nChannels;
	uchar* pData = (uchar*)srcImage.data;
	int N = srcImage.rows * srcImage.cols;
	uchar* pDst = (uchar*)dstImage.data;
	

	uchar* labels = new uchar[N];
	// Initialize k centroids
	InitCentroids(pData, N, nChannels);

	for (int i = 0; i < _maxIter; i++)
	{
		AssignLabels(pData, N, nChannels, labels);
		UpdateCentroids(pData, N, nChannels, labels);
		if (_hasCoverged == true)
			break;
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < nChannels; j++)
			pDst[i * nChannels + j] = (uchar)_centroids[labels[i]][j];
	}

	return 1;
}
