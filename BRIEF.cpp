#include "BRIEF.h"

namespace my
{
	std::vector<std::vector<uint32_t>> BRIEF(cv::Mat& img, std::vector<cv::KeyPoint>& kpts, std::vector<cv::Point2i>& p, std::vector<cv::Point2i>& q, const int& patch_boundary = 8)
	{
		std::vector<std::vector<uint32_t>> desc_arr;

		// create descriptor for each good keypoint
		for (int i = 0; i < kpts.size(); i++)
		{
			cv::KeyPoint& kpt = kpts[i];

			// cutting 256 bit information into 8x32 bit
			std::vector<uint32_t> desc(8, 0);
			desc_arr.push_back(desc);

			for (int k = 0; k < p.size(); k++)
			{
				// calculate moment of given patch/image
				// 
				// centroid is -> [ M_10/M_00, M_01/M_00 }
				// and angle of the vector from [0,0] to the centroid is
				// atan(M_01/M_00/(M_10/M_00)) -> atan(M_01/M_10) = theta

				float m_10 = 0;
				float m_01 = 0;

				for (int l = -patch_boundary; l <= patch_boundary; l++)
				{
					for (int u = -patch_boundary; u <= patch_boundary; u++)
					{
						m_01 += u * img.at<float>(kpt.pt.y + l, kpt.pt.x + u);
						m_10 += l * img.at<float>(kpt.pt.y + l, kpt.pt.x + u);
					}
				}

				float ang = atan2f(m_01, m_10);

				int x_p = p[k].x * cos(ang) - p[k].y * sin(ang);
				int y_p = p[k].x * sin(ang) + p[k].y * cos(ang);

				int x_q = q[k].x * cos(ang) - q[k].y * sin(ang);
				int y_q = q[k].x * sin(ang) + q[k].y * cos(ang);

				if (img.at<float>(kpts[i].pt.y + x_p, kpts[i].pt.x + y_p) > img.at<float>(kpts[i].pt.y + x_q, kpts[i].pt.x + y_q))
				{
					// bit index; 0 to 31
					int idx = k % 32;
					// vector index; 0 to 7
					int r = k / 32;
					// switch the bit corresponding to the given index
					desc_arr[i][r] = desc_arr[i][r] | 1 << idx;
				}
			}
		}
		return desc_arr;
	}

	void matchKeypoints(std::vector<std::vector<uint32_t>>& desc_1, std::vector<std::vector<uint32_t>>& desc_2, std::vector<cv::DMatch>& match_arr, const int& max_dist=35)
	{
		for (int i = 0; i < desc_1.size(); ++i)
		{
			cv::DMatch match(i, 0, 256);
			for (int j = 0; j < desc_2.size(); ++j)
			{
				int dist = 0;
				for (int k = 0; k < 8; k++)
				{
					// bitwise XOR operator to calculate Hamming distance
					uint res = desc_1[i][k] ^ desc_2[j][k];

					// dist += _mm_popcnt_u32(desc_1[i][k] ^ desc_2[j][k]);
					while (res > 0)
					{
						dist += res & 1;
						res = res >> 1;
					}
				}
				if (dist < max_dist && dist < match.distance)
				{
					match.distance = dist;
					match.trainIdx = j;
				}
			}
			if (match.distance < max_dist)
			{
				match_arr.push_back(match);
			}
		}
	}
}
