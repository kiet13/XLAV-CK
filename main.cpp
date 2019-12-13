#include "Threshold.h"
#include "Kmean.h"


int main(int argc, char *argv[])
{
	try
	{
		// Get input path
		char* fileinput = argv[2];
		Mat original = imread(fileinput, CV_LOAD_IMAGE_ANYCOLOR); // Original image

		// Chuyển ảnh ban đầu về ảnh xám
		Mat grayOriginal;
		cvtColor(original, grayOriginal, CV_BGR2GRAY);
		Mat dstImage;

		if (strcmp(argv[1], "--static") == 0)
		{
			int lowThreshold = atoi(argv[3]);
			int highThreshold = atoi(argv[4]);
			StaticThreshold staticThreshold(lowThreshold, highThreshold);
			
			staticThreshold.Apply(grayOriginal, dstImage);
			
		}
		else if (strcmp(argv[1], "--mean") == 0)
		{
			int size = atoi(argv[3]);
			if (size <= 0)
			{
				cout << "Invalid command!!!" << endl;
				return 0;
			}

			if (size == 1) size = 3;
			if (size % 2 == 0) size++;

			Size win_size;
			win_size.height = win_size.width = size;

			int C = atoi(argv[4]);

			AverageLocalThreshold mean(C);
			mean.Apply(grayOriginal, dstImage, win_size);
		}
		else if (strcmp(argv[1], "--sauvola") == 0)
		{
			int size = atoi(argv[3]);
			if (size <= 0)
			{
				cout << "Invalid command!!!" << endl;
				return 0;
			}

			if (size == 1) size = 3;
			if (size % 2 == 0) size++;

			Size win_size;
			win_size.height = win_size.width = size;

			float k = (float)atof(argv[4]);
			int r = atoi(argv[5]);
			
			SauvolaLocalThreshold sauvola(r, k);
			sauvola.Apply(grayOriginal, dstImage, win_size);
		}
		else if (strcmp(argv[1], "--kmean") == 0)
		{
			int K = atoi(argv[3]);

			Kmean kmean(K);

			kmean.Apply(original, dstImage);
		}
		imshow("Original", original);
		imshow("Processed", dstImage);
		waitKey(0);
	}
	catch (cv::Exception &e)
	{
		cerr << e.msg << endl;
	}
	return 0;
}