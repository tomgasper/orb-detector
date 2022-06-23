#define _USE_MATH_DEFINES

#include <iostream>
#include <chrono>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "util.h"
#include "FAST.h"
#include "BRIEF.h"

int main()
{
	// load img data
	std::string dir_0 = "data/ap_0.jpg";
	std::string dir_1 = "data/ap_1.jpg";

	cv::Mat img_0 = imread(dir_0, cv::IMREAD_GRAYSCALE);
	cv::Mat img_1 = imread(dir_1, cv::IMREAD_GRAYSCALE);

	// resize, add blur and convert to CV_32F
	my::prepareImg(img_0, 6, 1);
	my::prepareImg(img_1, 6, 1);

	// create containers
	cv::Mat img_0_features(img_0.rows, img_0.cols, CV_32F, cv::Scalar(0));
	std::vector<cv::KeyPoint> img_0_kpts_arr;

	cv::Mat img_1_features(img_1.rows, img_1.cols, CV_32F, cv::Scalar(0));
	std::vector<cv::KeyPoint> img_1_kpts_arr;

	const int patch_boundary = 8;
	const int patch_45_deg_offset = (int)ceil(sqrt(2 * powf((float)patch_boundary,2)));

	std::vector<std::vector<uint32_t>> desc0,desc1;
	std::vector<cv::DMatch> match_arr;
	cv::Mat matches_img(img_0.rows, img_0.cols, CV_8U);

	// generate random pairs for BRIEF descriptor
	std::vector<cv::Point2i> p, q;
	my::randomPairs(p, q, patch_boundary);

	// start looking for keypoints
	my::FAST(img_0, img_0_features, img_0_kpts_arr, 12, 40);
	my::FAST(img_1, img_1_features, img_1_kpts_arr, 12, 40);

	my::HarrisMeasure(img_0_features, img_0_kpts_arr, 0.04);
	my::HarrisMeasure(img_1_features, img_1_kpts_arr, 0.04);

	my::filterOutOfBounds(img_0_kpts_arr, img_0.rows, img_0.cols, patch_boundary+ patch_45_deg_offset);
	my::filterOutOfBounds(img_1_kpts_arr, img_1.rows, img_1.cols, patch_boundary+ patch_45_deg_offset);

	desc0 = my::BRIEF(img_0, img_0_kpts_arr, p, q,patch_boundary);
	desc1 = my::BRIEF(img_1, img_1_kpts_arr, p, q,patch_boundary);

	img_0.convertTo(img_0, CV_8U);
	img_1.convertTo(img_1, CV_8U);

	// show matches
	my::matchKeypoints(desc0, desc1, match_arr, 35);
	cv::drawMatches(img_0, img_0_kpts_arr, img_1, img_1_kpts_arr, match_arr,matches_img);

	cv::imshow("Matches", matches_img);

	cv::waitKey(0);
	return 0;
}