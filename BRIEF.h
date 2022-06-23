#pragma once
#include <opencv2/core.hpp>
#include <iostream>

namespace my
{
	std::vector<std::vector<uint32_t>> BRIEF(cv::Mat& img, std::vector<cv::KeyPoint>& kpts, std::vector<cv::Point2i>& p, std::vector<cv::Point2i>& q, const int&);
	void matchKeypoints(std::vector<std::vector<uint32_t>>& desc_1, std::vector<std::vector<uint32_t>>& desc_2, std::vector<cv::DMatch>& match_arr, const int& max_dist);
}