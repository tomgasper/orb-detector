#include "FAST.h"

namespace my
{
	void FAST(cv::Mat& img, cv::Mat& dst, std::vector<cv::KeyPoint>& kpts, int n = 9, float t = 0)
	{
		for (int i = 3; i < img.rows-3; i++)
		{
			for (int j = 3; j < img.cols-3; j++)
			{
				float c = img.at<float>(i, j);

				// ring with radius 3
				std::vector<float> ring
				{ img.at<float>(i, j + 3), img.at<float>(i - 1, j + 3), img.at<float>(i - 2, j + 2), img.at<float>(i - 3, j + 1),
				img.at<float>(i - 3, j), img.at<float>(i - 3, j - 1), img.at<float>(i - 2, j - 2), img.at<float>(i - 1, j - 3), img.at<float>(i, j - 3),
				img.at<float>(i + 1, j - 3), img.at<float>(i + 2, j - 2), img.at<float>(i + 3, j - 1), img.at<float>(i + 3, j),
				img.at<float>(i + 3, j + 1), img.at<float>(i + 2, j + 2), img.at<float>(i + 1, j + 3)
				};

				int count_bright = 0;
				int count_dark = 0;

				// quick check to avoid unnecessary computation
				// to be added

				// quick check 1

				if (c - t <= img.at<float>(i - 3, j) && c + t >= img.at<float>(i - 3, j)
					&& c - t <= img.at<float>(i + 3, j) && c + t >= img.at<float>(i + 3, j)) {
					continue;
				}

				// quick check 2

				if (img.at<float>(i - 3, j) > c + t) count_bright++;
				else count_dark++;

				if (img.at<float>(i + 3, j) > c + t) count_bright++;
				else count_dark++;

				if (img.at<float>(i, j+3) > c + t) count_bright++;
				else count_dark++;

				if (img.at<float>(i, j - 3) > c + t) count_bright++;
				else count_dark++;

				if (count_bright >= 3 || count_dark >= 3)
				{
					count_bright = 0;
					count_dark = 0;

					for (int i = 0; i < ring.size(); i++)
					{
						if (c + t < ring[i]) { count_bright++; }
						else if (c - t > ring[i]) { count_dark++; }
					}

					if (count_bright > n || count_dark > n)
					{
						cv::KeyPoint kpt;
						kpt.pt = cv::Point2i(j, i);
						kpts.push_back(kpt);
						dst.at<float>(i, j) = c;
					}
				}

			}
		}
	}

	void HarrisMeasure(cv::Mat& img, std::vector<cv::KeyPoint>& kpts, float k = 0.04)
	{
		struct R_res {
			uint indx;
			float R;
		};

		std::vector<R_res> R_arr;

		float mean = 0.F;
		float o_sqr = 0.F;

		for (int p = 0; p < kpts.size(); p++)
		{
			cv::KeyPoint& kpt = kpts[p];

			if (kpt.pt.x < 1 || kpt.pt.y < 1 || kpt.pt.x > img.rows - 1 || kpt.pt.y > img.cols - 1) continue;

			// calculate mean and variance of the 3x3 patch
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					mean += img.at<float>(kpt.pt.y + i, kpt.pt.x + j);
				}
			}
			mean = mean / 9;

			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					o_sqr += powf(img.at<float>(kpt.pt.y + i, kpt.pt.x + j) - mean, 2);
				}
			}

			o_sqr = o_sqr / (9 - 1);

			// calculate autocorrelation matrix
			cv::Mat A(2, 2, CV_32F, cv::Scalar(0));
			cv::Mat M = cv::Mat(2, 2, CV_32F, cv::Scalar(0));

			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					// derivatives of image patch
					float G_x = img.at<float>(i + kpt.pt.y, j + kpt.pt.x+1) - img.at<float>(i+kpt.pt.y, j+kpt.pt.x-1);
					float G_y = img.at<float>(i+kpt.pt.y+1, j+kpt.pt.x) - img.at<float>(i+kpt.pt.y-1, j+kpt.pt.x);

					
					// weight, circular Gaussian
					float w = (exp(-(powf(i, 2) + powf(j, 2)) / (2.F * o_sqr))) / (2.F * M_PI * o_sqr);
					float G_x_y = G_x * G_y;

					/*float M_data[4]{ powf(G_x,2), G_x_y,
										G_x_y, powf(G_y,2)
					};*/

					M.at<float>(0, 0) = powf(G_x, 2);
					M.at<float>(0, 1) = G_x_y;
					M.at<float>(1, 0) = G_x_y;
					M.at<float>(1, 1) = powf(G_y, 2);
					
					M = w * M;
					A += M;
				}
			}

			cv::Mat R(1, 1, CV_32F);

			R = (float)cv::determinant(A) - k * (pow(cv::trace(A)[0], 2));

			// push to list for later comparison
			R_res res;
			res.indx = p;
			res.R = R.at<float>(0, 0);

			R_arr.push_back(res);
		}

		assert(!R_arr.empty());

		// sort array by descending order
		// bubble sort
		for (int i = 0; i < R_arr.size(); i++)
		{
			for (int j = 0; j < R_arr.size() - i - 1; j++)
			{
				if (R_arr[j].R < R_arr[j + 1].R)
				{
					std::swap(R_arr[j], R_arr[j + 1]);
				}
			}
		}

		cv::Mat test_img(img.rows, img.cols, CV_8U, cv::Scalar(0));
		std::vector<cv::KeyPoint> new_kpts;

		int size_limit = (int)(R_arr.size() / 5);

		for (int i = 0; i < size_limit; i++)
		{
			if (R_arr[i].R > 0)
			{
				cv::KeyPoint new_kpt;
				new_kpt.pt = kpts[R_arr[i].indx].pt;

				new_kpts.push_back(new_kpt);
			}
			// test_img.at<char>(new_kpt.pt.y, new_kpt.pt.x) = 255;
		}
		kpts = new_kpts;

		// cv::imshow("Harris Test black", test_img);
	}
}