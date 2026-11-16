// Curve.h
//
// The tone curve of the video curve tool: control points that map input
// levels to output levels, a smooth curve through them and the 256-entry
// lookup table (LUT) that applies the curve to 8-bit images. Also the point
// editing used by the curve window, and the drawing of that window.
//
// Michael Perez, CAP 4410 Computer Vision, Assignment 1, Florida Polytechnic
// University (2019). Rewritten in 2026: the curve is a monotone cubic Hermite
// interpolant evaluated as a function of the input level, which replaces the
// 2019 Catmull-Rom spline code and its bisection search.
//
// Interpolation: F. N. Fritsch and R. E. Carlson, "Monotone Piecewise Cubic
// Interpolation", SIAM Journal on Numerical Analysis 17(2), 238-246, 1980.

#ifndef CURVE_H
#define CURVE_H

#include <opencv2/core.hpp>

#include <string>
#include <vector>

class ToneCurve
{
public:
	static constexpr int kLevels = 256;  // levels of an 8-bit channel
	static constexpr int kMaxLevel = kLevels - 1;

	// The identity curve: control points (0, 0) and (255, 255)
	ToneCurve();

	// A curve through `points`, where x is the input level and y the output
	// level, both from 0 to 255. The points are sorted by x; at least two are
	// needed and no two may have the same x. Throws std::invalid_argument if
	// the points do not meet these conditions.
	explicit ToneCurve(std::vector<cv::Point> points);

	// Parses control points written as "x:y,x:y,...", for example
	// "0:0,128:191,255:255". Returns false and describes the problem in
	// `error` if the text is not a valid curve; `curve` is then unchanged.
	static bool Parse(const std::string& text, ToneCurve& curve, std::string& error);

	// The control points in the format read by Parse()
	std::string ToString() const;

	// Control points, sorted by x
	const std::vector<cv::Point>& Points() const { return points_; }

	// Output level for the input level x, before rounding. The curve passes
	// through every control point, and it is constant before the first point
	// and after the last one. Between two points it is a cubic that stays
	// between their output levels.
	double Evaluate(double x) const;

	// 1x256 CV_8U lookup table for cv::LUT: the curve at levels 0 to 255,
	// rounded to the nearest level
	cv::Mat Lut() const;

	// Point editing, used by the curve window. Indices refer to Points().

	// Index of the control point nearest to `p` (in levels), or -1 if none is
	// within `max_distance`
	int Nearest(cv::Point2d p, double max_distance) const;

	// Adds a control point at `p` (clamped to levels 0 to 255) and returns
	// its index. If a point already has that x, it is moved to `p` instead.
	int Insert(cv::Point p);

	// Moves control point `index` towards `p` and returns where it ends up.
	// Points cannot pass their neighbours, so x stays strictly between the x
	// of the points on either side, and both levels stay within 0 to 255.
	cv::Point Move(int index, cv::Point p);

	// Removes control point `index`. Returns false and keeps the point if the
	// curve would be left with fewer than two points.
	bool Remove(int index);

private:
	void ComputeTangents();

	std::vector<cv::Point> points_;
	std::vector<double> tangents_;  // slope dy/dx of the curve at each control point
};

// The curve window: the plot covers levels 0 to 255 on both axes with
// kCurveScale pixels per level and a margin of kCurveMargin pixels.
constexpr int kCurveScale = 2;
constexpr int kCurveMargin = 8;
constexpr int kCurveImageSize = 2 * kCurveMargin + kCurveScale * ToneCurve::kMaxLevel + 1;

// Pixel position in the curve window of input level x and output level y
cv::Point2d CurvePixel(double x, double y);

// Levels at pixel (px, py) of the curve window, rounded and clamped to 0..255
cv::Point CurveLevels(int px, int py);

// Image of the curve window (kCurveImageSize x kCurveImageSize, BGR). The
// histogram (256 bins, any type, may be empty) is drawn behind the curve. The
// point `selected` (an index, or -1) is filled, and if `cursor` (a pixel
// position) is inside the image, lines through it and its levels are shown.
cv::Mat DrawCurve(const ToneCurve& curve, const cv::Mat& histogram, int selected = -1,
                  cv::Point cursor = cv::Point(-1, -1));

#endif
