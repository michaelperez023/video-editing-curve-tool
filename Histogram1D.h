#pragma once
// To create Histograms of gray-level images
class Histogram1D
{
private:
	int histSize[1];
	float hranges[2];
	const float* ranges[1];
	int channels[1];

public:
	Histogram1D()
	{
		histSize[0] = 256;
		hranges[0] = 0.0;
		hranges[1] = 256.00;
		ranges[0] = hranges;
		channels[0] = 0;
	}

	// Computes the 1D histogram.
	cv::Mat getHistogram(const cv::Mat& image) {

		cv::Mat hist;

		// Compute 1D histogram with calcHist
		cv::calcHist(&image,
			1,			// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used
			hist,		// the resulting histogram
			1,			// it is a 1D histogram
			histSize,	// number of bins
			ranges		// pixel value range
		);

		return hist;
	}

	// Computes the 1D histogram and returns an image of it.
	cv::Mat getHistogramImage(const cv::Mat& image, int zoom = 1) {

		// Compute histogram first
		cv::Mat hist = getHistogram(image);

		// Creates image
		return Histogram1D::getImageOfHistogram(hist, zoom);
	}

	// static methods

    // Create an image representing a histogram
	static cv::Mat getImageOfHistogram(const cv::Mat& hist, int zoom) {

		// Get min and max bin values
		double maxVal = 0;
		double minVal = 0;
		cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

		// get histogram size
		int histSize = hist.rows;

		// Square image on which to display histogram
		cv::Mat histImg(histSize * zoom, histSize * zoom, CV_8U, cv::Scalar(255));

		// set highest point at 90% of nbins (i.e. image height)
		int hpt = static_cast<int>(0.9 * histSize);

		// Draw vertical line for each bin
		for (int h = 0; h < histSize; h++) {

			float binVal = hist.at<float>(h);
			if (binVal > 0) {
				int intensity = static_cast<int>(binVal * hpt / maxVal);
				cv::line(histImg, cv::Point(h * zoom, histSize * zoom),
					cv::Point(h * zoom, (histSize - intensity) * zoom), cv::Scalar(0), zoom);
			}
		}

		return histImg;
	}

};