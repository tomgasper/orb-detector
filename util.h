#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <random>

namespace my
{
	void prepareImg(cv::Mat& img, int factor, int blur=1)
	{
		assert(!img.empty());

		cv::resize(img, img, img.size() / factor, 0, 0);
		cv::GaussianBlur(img, img, cv::Size(blur, blur), 0);
		img.convertTo(img, CV_32F);
	}


	void randomPairs(std::vector<cv::Point2i>& p, std::vector<cv::Point2i>& q)
	{
		std::random_device rd; // obtain random number
		std::mt19937 mt(rd()); // seed
		std::uniform_int_distribution<int> r1(-8, 8);

		for (int i = 0; i < 256; ++i)
		{
			p.push_back(cv::Point2i(r1(mt), r1(mt)));
			q.push_back(cv::Point2i(r1(mt), r1(mt)));
		}
	}
}
