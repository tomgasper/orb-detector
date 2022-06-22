#pragma once
#define _USE_MATH_DEFINES
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace my
{
	void FAST(cv::Mat&, cv::Mat&, std::vector<cv::KeyPoint>&, int, float);
	void HarrisMeasure(cv::Mat& img, std::vector<cv::KeyPoint>& kpts, float k);
}