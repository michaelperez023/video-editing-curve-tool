// Curve.cpp
//
// Tone curve, lookup table and curve window drawing (see Curve.h).
//
// The curve is a piecewise cubic Hermite interpolant of the control points,
// evaluated directly as y(x). Its slopes at the control points follow Fritsch
// and Carlson (1980): each starts as the slope of the parabola through the
// point and its neighbours, it is zero where the curve has a peak or a
// valley, and the two slopes of each segment are reduced where needed so that
// the segment is monotone. The curve therefore passes through every control
// point, never overshoots between them, and is monotone wherever the control
// points are.

#include "Curve.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
	bool IsLevel(int value)
	{
		return value >= 0 && value <= ToneCurve::kMaxLevel;
	}

	// Sorts the points by x and checks the conditions of the ToneCurve
	// constructor. Returns an empty string, or a description of the problem.
	std::string SortAndCheck(std::vector<cv::Point>& points)
	{
		for (const cv::Point& p : points)
		{
			if (!IsLevel(p.x) || !IsLevel(p.y))
				return "the point " + std::to_string(p.x) + ":" + std::to_string(p.y) +
				       " is outside levels 0 to 255";
		}
		std::sort(points.begin(), points.end(),
		          [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; });
		for (std::size_t i = 1; i < points.size(); ++i)
		{
			if (points[i].x == points[i - 1].x)
				return "two points have x = " + std::to_string(points[i].x);
		}
		if (points.size() < 2)
			return "at least two points are needed";
		return "";
	}

	// Splits `text` at every `separator`, keeping empty parts
	std::vector<std::string> Split(const std::string& text, char separator)
	{
		std::vector<std::string> parts;
		std::size_t start = 0;
		while (true)
		{
			const std::size_t end = text.find(separator, start);
			parts.push_back(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
			if (end == std::string::npos)
				return parts;
			start = end + 1;
		}
	}

	std::string Trim(const std::string& text)
	{
		const char* const kSpace = " \t";
		const std::size_t first = text.find_first_not_of(kSpace);
		if (first == std::string::npos)
			return "";
		return text.substr(first, text.find_last_not_of(kSpace) - first + 1);
	}

	// Parses a level: an integer from 0 to 255, without a sign
	bool ParseLevel(const std::string& text, int& value)
	{
		const std::string digits = Trim(text);
		if (digits.empty() || digits.size() > 3)
			return false;
		for (const char c : digits)
		{
			if (c < '0' || c > '9')
				return false;
		}
		value = std::stoi(digits);
		return IsLevel(value);
	}

	// Drawing in the curve window uses fixed-point coordinates with this many
	// fractional bits, so that anti-aliased lines fall between pixels
	constexpr int kShift = 4;

	cv::Point Fixed(const cv::Point2d& p)
	{
		return cv::Point(cvRound(p.x * (1 << kShift)), cvRound(p.y * (1 << kShift)));
	}

	int RoundLevel(double y)
	{
		return std::clamp(static_cast<int>(std::floor(y + 0.5)), 0, ToneCurve::kMaxLevel);
	}
}

ToneCurve::ToneCurve()
	: ToneCurve(std::vector<cv::Point>{cv::Point(0, 0), cv::Point(kMaxLevel, kMaxLevel)})
{
}

ToneCurve::ToneCurve(std::vector<cv::Point> points)
	: points_(std::move(points))
{
	const std::string problem = SortAndCheck(points_);
	if (!problem.empty())
		throw std::invalid_argument("Invalid tone curve: " + problem);
	ComputeTangents();
}

bool ToneCurve::Parse(const std::string& text, ToneCurve& curve, std::string& error)
{
	if (Trim(text).empty())
	{
		error = "no points given";
		return false;
	}

	std::vector<cv::Point> points;
	for (const std::string& part : Split(text, ','))
	{
		const std::vector<std::string> levels = Split(part, ':');
		cv::Point p;
		if (levels.size() != 2 || !ParseLevel(levels[0], p.x) || !ParseLevel(levels[1], p.y))
		{
			error = "'" + Trim(part) + "' is not a point x:y with levels from 0 to 255";
			return false;
		}
		points.push_back(p);
	}

	const std::string problem = SortAndCheck(points);
	if (!problem.empty())
	{
		error = problem;
		return false;
	}
	curve = ToneCurve(std::move(points));
	return true;
}

std::string ToneCurve::ToString() const
{
	std::string text;
	for (const cv::Point& p : points_)
	{
		if (!text.empty())
			text += ',';
		text += std::to_string(p.x) + ":" + std::to_string(p.y);
	}
	return text;
}

void ToneCurve::ComputeTangents()
{
	const std::size_t n = points_.size();
	std::vector<double> width(n - 1);  // x distance from each point to the next
	std::vector<double> slope(n - 1);  // slope of the straight line to the next point
	for (std::size_t k = 0; k + 1 < n; ++k)
	{
		width[k] = points_[k + 1].x - points_[k].x;
		slope[k] = (points_[k + 1].y - points_[k].y) / width[k];
	}

	// Initial slopes: the slope of the parabola through each point and its two
	// neighbours, and at the two ends the slope of the parabola through the
	// end point and the next two (a straight line for two points). Inside,
	// the slope is zero at a peak, a valley or next to a flat segment.
	tangents_.assign(n, 0.0);
	if (n == 2)
	{
		tangents_[0] = slope[0];
		tangents_[1] = slope[0];
	}
	else
	{
		tangents_[0] = ((2.0 * width[0] + width[1]) * slope[0] - width[0] * slope[1]) / (width[0] + width[1]);
		tangents_[n - 1] = ((2.0 * width[n - 2] + width[n - 3]) * slope[n - 2] - width[n - 2] * slope[n - 3]) /
		                   (width[n - 2] + width[n - 3]);
	}
	for (std::size_t k = 1; k + 1 < n; ++k)
	{
		if (slope[k - 1] * slope[k] > 0.0)
			tangents_[k] = (width[k] * slope[k - 1] + width[k - 1] * slope[k]) / (width[k - 1] + width[k]);
	}

	// A segment with end slopes a*s and b*s, where s is the slope of its
	// chord, is monotone if a, b >= 0 and a^2 + b^2 <= 9 (Fritsch and Carlson's
	// sufficient condition). An end slope against the direction of its
	// segment becomes zero, and slopes that are too steep are scaled down;
	// both only make slopes smaller, which keeps segments that were already
	// checked monotone.
	for (std::size_t k = 0; k + 1 < n; ++k)
	{
		if (slope[k] == 0.0)
		{
			tangents_[k] = 0.0;
			tangents_[k + 1] = 0.0;
			continue;
		}
		double a = tangents_[k] / slope[k];
		double b = tangents_[k + 1] / slope[k];
		if (a < 0.0)
		{
			tangents_[k] = 0.0;
			a = 0.0;
		}
		if (b < 0.0)
		{
			tangents_[k + 1] = 0.0;
			b = 0.0;
		}
		const double radius2 = a * a + b * b;
		if (radius2 > 9.0)
		{
			const double scale = 3.0 / std::sqrt(radius2);
			tangents_[k] = scale * a * slope[k];
			tangents_[k + 1] = scale * b * slope[k];
		}
	}
}

double ToneCurve::Evaluate(double x) const
{
	if (x <= points_.front().x)
		return points_.front().y;
	if (x >= points_.back().x)
		return points_.back().y;

	// Segment k, with points_[k].x <= x < points_[k + 1].x
	std::size_t k = 0;
	while (x >= points_[k + 1].x)
		++k;

	const cv::Point& p0 = points_[k];
	const cv::Point& p1 = points_[k + 1];
	const double width = p1.x - p0.x;
	const double t = (x - p0.x) / width;
	const double t2 = t * t;
	const double t3 = t2 * t;

	// Cubic Hermite basis functions
	const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
	const double h10 = t3 - 2.0 * t2 + t;
	const double h01 = -2.0 * t3 + 3.0 * t2;
	const double h11 = t3 - t2;
	return h00 * p0.y + h10 * width * tangents_[k] + h01 * p1.y + h11 * width * tangents_[k + 1];
}

cv::Mat ToneCurve::Lut() const
{
	cv::Mat lut(1, kLevels, CV_8U);
	for (int level = 0; level < kLevels; ++level)
		lut.at<uchar>(0, level) = static_cast<uchar>(RoundLevel(Evaluate(level)));
	return lut;
}

int ToneCurve::Nearest(cv::Point2d p, double max_distance) const
{
	int nearest = -1;
	double nearest_distance = 0.0;
	for (std::size_t i = 0; i < points_.size(); ++i)
	{
		const double distance = std::hypot(points_[i].x - p.x, points_[i].y - p.y);
		if (distance <= max_distance && (nearest < 0 || distance < nearest_distance))
		{
			nearest = static_cast<int>(i);
			nearest_distance = distance;
		}
	}
	return nearest;
}

int ToneCurve::Insert(cv::Point p)
{
	p.x = std::clamp(p.x, 0, kMaxLevel);
	p.y = std::clamp(p.y, 0, kMaxLevel);
	auto it = std::lower_bound(points_.begin(), points_.end(), p.x,
	                           [](const cv::Point& point, int x) { return point.x < x; });
	if (it != points_.end() && it->x == p.x)
		it->y = p.y;
	else
		it = points_.insert(it, p);
	const int index = static_cast<int>(it - points_.begin());
	ComputeTangents();
	return index;
}

cv::Point ToneCurve::Move(int index, cv::Point p)
{
	const int count = static_cast<int>(points_.size());
	if (index < 0 || index >= count)
		throw std::out_of_range("ToneCurve::Move: no point " + std::to_string(index));

	const int lowest = index > 0 ? points_[index - 1].x + 1 : 0;
	const int highest = index + 1 < count ? points_[index + 1].x - 1 : kMaxLevel;
	cv::Point& point = points_[index];
	point.x = std::clamp(p.x, lowest, highest);
	point.y = std::clamp(p.y, 0, kMaxLevel);
	ComputeTangents();
	return point;
}

bool ToneCurve::Remove(int index)
{
	if (index < 0 || index >= static_cast<int>(points_.size()))
		throw std::out_of_range("ToneCurve::Remove: no point " + std::to_string(index));
	if (points_.size() <= 2)
		return false;
	points_.erase(points_.begin() + index);
	ComputeTangents();
	return true;
}

cv::Point2d CurvePixel(double x, double y)
{
	return cv::Point2d(kCurveMargin + kCurveScale * x, kCurveMargin + kCurveScale * (ToneCurve::kMaxLevel - y));
}

cv::Point CurveLevels(int px, int py)
{
	return cv::Point(RoundLevel(static_cast<double>(px - kCurveMargin) / kCurveScale),
	                 RoundLevel(ToneCurve::kMaxLevel - static_cast<double>(py - kCurveMargin) / kCurveScale));
}

cv::Mat DrawCurve(const ToneCurve& curve, const cv::Mat& histogram, int selected, cv::Point cursor)
{
	const cv::Scalar kBackground(255, 255, 255);
	const cv::Scalar kHistogram(218, 218, 218);
	const cv::Scalar kGrid(200, 200, 200);
	const cv::Scalar kBorder(130, 130, 130);
	const cv::Scalar kCurve(25, 25, 25);
	const cv::Scalar kCursor(60, 60, 220);  // red (BGR)
	constexpr int kMax = ToneCurve::kMaxLevel;

	cv::Mat image(kCurveImageSize, kCurveImageSize, CV_8UC3, kBackground);

	// Histogram of the frame behind the curve; the largest bin reaches 90% of
	// the height, as in Histogram1D
	if (!histogram.empty() && histogram.total() == static_cast<std::size_t>(ToneCurve::kLevels))
	{
		cv::Mat bins;
		histogram.reshape(1, 1).convertTo(bins, CV_64F);
		double largest = 0.0;
		cv::minMaxLoc(bins, nullptr, &largest);
		for (int level = 0; largest > 0.0 && level < ToneCurve::kLevels; ++level)
		{
			const double height = 0.9 * kMax * bins.at<double>(0, level) / largest;
			if (height <= 0.0)
				continue;
			const cv::Point2d bottom = CurvePixel(level, 0);
			const cv::Point2d top = CurvePixel(level, height);
			const int left = cvRound(bottom.x) - kCurveScale / 2;
			cv::rectangle(image, cv::Point(left, cvRound(top.y)),
			              cv::Point(left + kCurveScale - 1, cvRound(bottom.y)), kHistogram, cv::FILLED);
		}
	}

	// Grid at a quarter, half and three quarters of the range, the border of
	// the plot and the identity line
	for (const int level : {64, 128, 192})
	{
		cv::line(image, CurvePixel(level, 0), CurvePixel(level, kMax), kGrid, 1);
		cv::line(image, CurvePixel(0, level), CurvePixel(kMax, level), kGrid, 1);
	}
	cv::rectangle(image, CurvePixel(0, kMax), CurvePixel(kMax, 0), kBorder, 1);
	cv::line(image, Fixed(CurvePixel(0, 0)), Fixed(CurvePixel(kMax, kMax)), kGrid, 1, cv::LINE_AA, kShift);

	// The curve, sampled every quarter level
	std::vector<cv::Point> samples;
	for (int i = 0; i <= 4 * kMax; ++i)
	{
		const double x = i / 4.0;
		samples.push_back(Fixed(CurvePixel(x, curve.Evaluate(x))));
	}
	cv::polylines(image, samples, false, kCurve, 2, cv::LINE_AA, kShift);

	// Control points: hollow, or filled when selected
	const std::vector<cv::Point>& points = curve.Points();
	for (std::size_t i = 0; i < points.size(); ++i)
	{
		const cv::Point centre = Fixed(CurvePixel(points[i].x, points[i].y));
		const int radius = 5 << kShift;
		const bool filled = static_cast<int>(i) == selected;
		cv::circle(image, centre, radius, filled ? kCurve : kBackground, cv::FILLED, cv::LINE_AA, kShift);
		cv::circle(image, centre, radius, kCurve, 2, cv::LINE_AA, kShift);
	}

	// Lines through the mouse position, and the input level under it with
	// the output level the curve gives it
	if (cursor.x >= 0 && cursor.y >= 0 && cursor.x < image.cols && cursor.y < image.rows)
	{
		cv::line(image, cv::Point(0, cursor.y), cv::Point(image.cols - 1, cursor.y), kCursor, 1);
		cv::line(image, cv::Point(cursor.x, 0), cv::Point(cursor.x, image.rows - 1), kCursor, 1);
		const int input = CurveLevels(cursor.x, cursor.y).x;
		const std::string label = "in " + std::to_string(input) + "  out " +
		                          std::to_string(RoundLevel(curve.Evaluate(input)));
		cv::putText(image, label, cv::Point(kCurveMargin + 8, kCurveMargin + 22), cv::FONT_HERSHEY_SIMPLEX,
		            0.55, kCurve, 1, cv::LINE_AA);
	}
	return image;
}
