#define _USE_MATH_DEFINES

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "util.h"
#include "FAST.h"
#include "BRIEF.h"

int main()
{
	std::string dir_0 = "data/ap_0.jpg";
	std::string dir_1 = "data/ap_1.jpg";

	cv::Mat img_0 = imread(dir_0, cv::IMREAD_GRAYSCALE);
	cv::Mat img_1 = imread(dir_1, cv::IMREAD_GRAYSCALE);

	// resize, add blur and convert to CV_32F
	my::prepareImg(img_0, 6, 1);
	my::prepareImg(img_1, 6, 1);

	cv::Mat img_0_features(img_0.rows, img_0.cols, CV_8U, cv::Scalar(0));
	std::vector<cv::KeyPoint> img_0_kpts;

	my::FAST(img_0, img_0_features, img_0_kpts, 12, 40);
	cv::imshow("Image 1 FAST", img_0_features);

	// CONVERT TO UINT AFTER
	img_0_features.convertTo(img_0_features, CV_32F);
	my::HarrisMeasure(img_0_features, img_0_kpts, 0.04);

	// generate random pairs for BRIEF descriptor
	std::vector<cv::Point2i> p, q;
	my::randomPairs(p, q, 4);

	std::vector<std::vector<uint32_t>> desc;
	desc = my::BRIEF(img_0, img_0_kpts, p, q);

	cv::waitKey(0);
	return 0;
}