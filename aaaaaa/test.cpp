#include "opencv2\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>

#include <opencv2/opencv.hpp>
#include <vector>
#include <assert.h>



using namespace cv;
using namespace std;
struct Blob
{
	cv::Size matContainSize;
	cv::Rect boundingRect;
	std::vector<cv::Point2i> points;

	Blob(Blob* b) {};
	Blob() {};
};

std::vector<Blob> FindBlobs(Mat& matBinary, cv::Size minSize, cv::Size maxSize)
{
	assert(matBinary.channels() == 1, "Image is not binary");


	std::vector<Blob> blobs;

	// Fill the label_image with the blobs
	// 0  - background
	// 1  - unlabelled foreground
	// 2+ - labelled foreground

	uchar label_count = 2; // starts at 2 because 0,1 are used already

	for (int y = 0; y < matBinary.rows; y++)
	{
		for (int x = 0; x < matBinary.cols; x++)
		{
			uchar val = matBinary.at<uchar>(y, x);
			if (val != 255)
				continue;

			cv::Rect rect;
			cv::floodFill(matBinary, cv::Point(x, y), ++label_count, &rect);
			if (label_count == 255)
				label_count = 2;

			if (minSize.area() > 0 && (rect.width < minSize.width || rect.height < minSize.height))
				continue;

			if (maxSize.area() > 0 && (rect.width > maxSize.width || rect.height > maxSize.height))
				continue;

			Blob blob;

			for (int i = rect.y; i < (rect.y + rect.height); i++)
			{
				for (int j = rect.x; j < (rect.x + rect.width); j++)
				{
					if (matBinary.at<uchar>(i, j) == label_count)
						blob.points.push_back(cv::Point2i(j, i));
				}
			}



			if (blob.points.size() == 0)
				continue;

			blob.boundingRect = rect;
			blob.matContainSize = matBinary.size();
			blobs.push_back(blob);

		}
	}
	return blobs;
}

int main()
{
	CascadeClassifier cascade = cv::CascadeClassifier("cascade.xml");

	//load ảnh và chuyển thành ảnh xám
	cv::Mat matGray = imread("11.jpg", IMREAD_GRAYSCALE);

	//detect
	std::vector<cv::Rect> rects;
	cascade.detectMultiScale(matGray, rects, 1.1, 3, CV_HAAR_SCALE_IMAGE);
	
	//in ra số lượng đối tượng phát hiện được
	std::cout << "Detected " << rects.size() << " objects";

	Mat img_color = imread("11.jpg", IMREAD_GRAYSCALE);
	Mat face_roi;
	Mat BlackWhite;
	Mat Blur;
	//Cắt biển số 
	for (int n = 0; n < rects.size(); n++) {
		rectangle(img_color, rects[n], Scalar(255, 0, 0), 0);
		matGray(rects[n]).copyTo(face_roi);
	}
	imshow("face_roi", face_roi);
	blur(face_roi, Blur, Size(5,5), Point(0,0), 0);//Làm mờ ảnh
	imshow("Blur", Blur);
	


	adaptiveThreshold(Blur, BlackWhite, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 15, 11);
	//									độ tương phản						đen trắng
	imshow("TrangDen", BlackWhite);
	vector<Blob> blob= FindBlobs(BlackWhite, Size(4,2), Size(100,100));
	//										   kích cỡ chữ =pixel


	//Hoán vị ảnh
	for (int i = 0; i < blob.size()-1; i++)
	{
		for (int j = i + 1; j < blob.size(); j++)
		{
			if ((blob[i].boundingRect.x > blob[j].boundingRect.x 
				&& blob[i].boundingRect.y>face_roi.size().height/2 
				&&blob[j].boundingRect.y> face_roi.size().height / 2) 
				|| 
				(blob[i].boundingRect.x > blob[j].boundingRect.x
				&& blob[i].boundingRect.y < face_roi.size().height / 2
				&& blob[j].boundingRect.y < face_roi.size().height / 2))
			{
				Blob tamp;
				tamp = blob[i];
				blob[i] = blob[j];
				blob[j] = tamp;
			}
		}
	}
	

	for (int i = 0; i < blob.size(); i++)
	{
		rectangle(BlackWhite, blob[i].boundingRect, Scalar(255,0,0 ), 1);
		Mat tachChuoi;
		face_roi(blob[i].boundingRect).copyTo(tachChuoi);

		char name[20];
		sprintf_s(name, "face_%d.png", i);
		
		imshow(name, tachChuoi);
		
	}
	imshow("BlackWhite", BlackWhite);
	/*imshow("BienSo", face_roi);*/
	imshow("image", matGray);
	imshow("VJ Face Detector", img_color);

	
	waitKey(0);
	return 0;
	
}