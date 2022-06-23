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


	void randomPairs(std::vector<cv::Point2i>& p, std::vector<cv::Point2i>& q, int bound = 8)
	{
		std::random_device rd; // obtain random number
		std::mt19937 mt(rd()); // seed
		std::uniform_int_distribution<int> r1(-bound, bound);

		for (int i = 0; i < 256; ++i)
		{
			p.push_back(cv::Point2i(r1(mt), r1(mt)));
			q.push_back(cv::Point2i(r1(mt), r1(mt)));
		}
	}

	void filterOutOfBounds(std::vector<cv::KeyPoint>& all_kpts, const int& rows, const int& cols, const int& patch_boundary)
	{
		std::vector<cv::KeyPoint> kpts;

		for (auto& kpt : all_kpts)
		{
			if (kpt.pt.x < patch_boundary || kpt.pt.y < patch_boundary || kpt.pt.y > rows - patch_boundary || kpt.pt.x > cols - patch_boundary)
			{
				continue;
			}
			else {
				cv::KeyPoint new_kpt;
				new_kpt.pt = cv::Point2i(kpt.pt.x, kpt.pt.y);
				kpts.push_back(kpt);
			}
		}
		all_kpts = kpts;
	}
}
