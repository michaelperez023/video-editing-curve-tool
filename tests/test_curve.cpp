// Tests for the tone curve (Curve.h): the lookup table and the interpolation
// behind it, parsing of --curve, point editing and the curve window drawing.
//
// Usage: curve_tests [name]   runs one test, or all of them without a name.
// The exit status is 0 when every check passes.

#include "Curve.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
	int g_failures = 0;

	void Check(bool ok, const char* expression, const char* file, int line)
	{
		if (!ok)
		{
			std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expression);
			++g_failures;
		}
	}

#define CHECK(expression) Check((expression), #expression, __FILE__, __LINE__)

	ToneCurve Curve(const char* text)
	{
		ToneCurve curve;
		std::string error;
		if (!ToneCurve::Parse(text, curve, error))
		{
			std::fprintf(stderr, "test curve %s is not valid: %s\n", text, error.c_str());
			++g_failures;
		}
		return curve;
	}

	std::vector<int> LutValues(const ToneCurve& curve)
	{
		const cv::Mat lut = curve.Lut();
		CHECK(lut.rows == 1 && lut.cols == 256 && lut.type() == CV_8UC1);
		std::vector<int> values(256);
		for (int i = 0; i < 256; ++i)
			values[i] = lut.at<uchar>(0, i);
		return values;
	}

	// Curves used by several tests, with points as they would be placed by hand
	const char* const kCurves[] = {
		"0:0,128:191,255:255",               // brighter
		"0:0,128:64,255:255",                // darker
		"0:0,64:40,192:215,255:255",         // more contrast (an S curve)
		"0:40,64:80,192:175,255:215",        // less contrast
		"0:255,255:0",                       // inverted
		"0:0,20:55,30:215,230:225,255:255",  // uneven spacing
		"0:0,100:40,108:230,255:255",        // a steep rise between close points
		"0:180,128:100,255:60",              // decreasing
		"30:0,220:255",                      // flat below 30 and above 220
		"0:0,128:255,255:0",                 // a peak
		"0:0,1:255,2:0,3:255,255:128",       // neighbouring levels
		"0:0,50:100,51:100,200:101,255:255", // nearly flat in the middle
	};

	// Random control points: 2 to 16 points with distinct x. With `order` +1
	// or -1 the output levels are sorted up or down, with 0 they are random.
	std::vector<cv::Point> RandomPoints(cv::RNG& rng, int order)
	{
		const int count = rng.uniform(2, 17);
		std::vector<int> xs;
		while (static_cast<int>(xs.size()) < count)
		{
			const int x = rng.uniform(0, 256);
			if (std::find(xs.begin(), xs.end(), x) == xs.end())
				xs.push_back(x);
		}
		std::sort(xs.begin(), xs.end());
		std::vector<int> ys;
		for (int i = 0; i < count; ++i)
			ys.push_back(rng.uniform(0, 256));
		if (order > 0)
			std::sort(ys.begin(), ys.end());
		else if (order < 0)
			std::sort(ys.begin(), ys.end(), [](int a, int b) { return a > b; });
		std::vector<cv::Point> points;
		for (int i = 0; i < count; ++i)
			points.emplace_back(xs[i], ys[i]);
		return points;
	}

	// Number of control points whose level the lookup table does not give exactly
	int MissedPoints(const ToneCurve& curve)
	{
		const std::vector<int> lut = LutValues(curve);
		int missed = 0;
		for (const cv::Point& p : curve.Points())
		{
			if (lut[p.x] != p.y)
				++missed;
		}
		return missed;
	}

	bool NeverDecreases(const std::vector<int>& lut)
	{
		for (std::size_t i = 1; i < lut.size(); ++i)
		{
			if (lut[i] < lut[i - 1])
				return false;
		}
		return true;
	}

	bool NeverIncreases(const std::vector<int>& lut)
	{
		for (std::size_t i = 1; i < lut.size(); ++i)
		{
			if (lut[i] > lut[i - 1])
				return false;
		}
		return true;
	}

	// Whether the curve stays between the output levels of the two control
	// points around every x (sampled every eighth of a level)
	bool StaysBetweenPoints(const ToneCurve& curve)
	{
		const std::vector<cv::Point>& points = curve.Points();
		for (std::size_t k = 0; k + 1 < points.size(); ++k)
		{
			const double low = std::min(points[k].y, points[k + 1].y) - 1e-9;
			const double high = std::max(points[k].y, points[k + 1].y) + 1e-9;
			for (double x = points[k].x; x <= points[k + 1].x; x += 0.125)
			{
				const double y = curve.Evaluate(x);
				if (y < low || y > high)
					return false;
			}
		}
		return true;
	}

	void TestIdentity()
	{
		// The default curve, and curves whose points all lie on the diagonal
		const ToneCurve curves[] = {
			ToneCurve(),
			Curve("0:0,64:64,200:200,255:255"),
			Curve("0:0,1:1,2:2,128:128,254:254,255:255"),
		};
		for (const ToneCurve& curve : curves)
		{
			const std::vector<int> lut = LutValues(curve);
			int different = 0;
			double largest_error = 0.0;
			for (int i = 0; i < 256; ++i)
			{
				if (lut[i] != i)
					++different;
			}
			for (double x = 0.0; x <= 255.0; x += 0.125)
				largest_error = std::max(largest_error, std::abs(curve.Evaluate(x) - x));
			std::printf("%-40s %d levels changed, largest error %.2g\n", curve.ToString().c_str(), different,
			            largest_error);
			CHECK(different == 0);
			CHECK(largest_error < 1e-9);
		}

		// Applied to an image, it changes nothing
		cv::Mat image(37, 53, CV_8UC3);
		cv::RNG rng(1);
		rng.fill(image, cv::RNG::UNIFORM, 0, 256);
		cv::Mat result;
		cv::LUT(image, ToneCurve().Lut(), result);
		CHECK(cv::norm(image, result, cv::NORM_INF) == 0.0);
	}

	void TestControlPoints()
	{
		for (const char* text : kCurves)
		{
			const int missed = MissedPoints(Curve(text));
			std::printf("%-40s %d control points missed\n", text, missed);
			CHECK(missed == 0);
		}

		// Random curves, including uneven and very close spacing
		cv::RNG rng(2019);
		int missed = 0;
		for (int trial = 0; trial < 3000; ++trial)
			missed += MissedPoints(ToneCurve(RandomPoints(rng, trial % 3 - 1)));
		std::printf("3000 random curves: %d control points missed\n", missed);
		CHECK(missed == 0);
	}

	void TestUnevenSpacing()
	{
		// In 2019 this curve gave level 204 at input 20, where the point is at
		// 55, and jumped from 44 to 199 between inputs 18 and 19
		const ToneCurve curve = Curve("0:0,20:55,30:215,230:225,255:255");
		const std::vector<int> lut = LutValues(curve);
		std::printf("LUT[18..21] = %d %d %d %d, LUT[30] = %d, LUT[230] = %d\n", lut[18], lut[19], lut[20], lut[21],
		            lut[30], lut[230]);
		CHECK(lut[0] == 0 && lut[20] == 55 && lut[30] == 215 && lut[230] == 225 && lut[255] == 255);
		CHECK(NeverDecreases(lut));

		// The steepest step is in the 10 levels from 20 to 30, which rise by
		// 160 levels (16 per level); a monotone cubic is at most 1.5 times as
		// steep as its chord there
		int steepest = 0;
		for (int i = 1; i < 256; ++i)
			steepest = std::max(steepest, lut[i] - lut[i - 1]);
		std::printf("largest step between neighbouring levels: %d\n", steepest);
		CHECK(steepest <= 25);
	}

	void TestMonotone()
	{
		const char* const increasing[] = {
			"0:0,128:191,255:255",
			"0:0,64:40,192:215,255:255",
			"0:0,20:55,30:215,230:225,255:255",
			"0:0,100:40,108:230,255:255",
			"30:0,220:255",
			"0:0,50:100,51:100,200:101,255:255",
			"0:0,10:250,11:251,12:252,255:255",
		};
		const char* const decreasing[] = {
			"0:255,255:0",
			"0:180,128:100,255:60",
			"0:255,10:250,20:20,255:0",
		};
		for (const char* text : increasing)
		{
			const bool ok = NeverDecreases(LutValues(Curve(text)));
			std::printf("%-40s %s\n", text, ok ? "never decreases" : "DECREASES");
			CHECK(ok);
		}
		for (const char* text : decreasing)
		{
			const bool ok = NeverIncreases(LutValues(Curve(text)));
			std::printf("%-40s %s\n", text, ok ? "never increases" : "INCREASES");
			CHECK(ok);
		}

		// Random monotone curves give monotone tables, and no curve overshoots
		// the control points around it
		cv::RNG rng(4410);
		int wrong_direction = 0;
		int overshoots = 0;
		for (int trial = 0; trial < 3000; ++trial)
		{
			const int order = trial % 3 - 1;
			const ToneCurve curve(RandomPoints(rng, order));
			const std::vector<int> lut = LutValues(curve);
			if ((order > 0 && !NeverDecreases(lut)) || (order < 0 && !NeverIncreases(lut)))
				++wrong_direction;
			if (!StaysBetweenPoints(curve))
				++overshoots;
		}
		std::printf("3000 random curves: %d monotone curves with a table that is not, %d overshoots\n",
		            wrong_direction, overshoots);
		CHECK(wrong_direction == 0);
		CHECK(overshoots == 0);
		for (const char* text : kCurves)
			CHECK(StaysBetweenPoints(Curve(text)));
	}

	void TestFlatEnds()
	{
		// Levels before the first point and after the last one keep its output level
		const std::vector<int> lut = LutValues(Curve("30:0,220:255"));
		int wrong = 0;
		for (int i = 0; i <= 30; ++i)
			wrong += lut[i] != 0;
		for (int i = 220; i < 256; ++i)
			wrong += lut[i] != 255;
		std::printf("30:0,220:255: %d levels outside 30-220 changed; LUT[125] = %d\n", wrong, lut[125]);
		CHECK(wrong == 0);
		CHECK(lut[125] > 100 && lut[125] < 155);

		const std::vector<int> constant = LutValues(Curve("64:100,192:100"));
		CHECK(std::all_of(constant.begin(), constant.end(), [](int v) { return v == 100; }));
	}

	void TestSmooth()
	{
		// The slope of the curve is continuous at every control point, so the
		// curve has no corners (a polyline through the points would have)
		for (const char* text : kCurves)
		{
			const ToneCurve curve = Curve(text);
			const std::vector<cv::Point>& points = curve.Points();
			double largest_jump = 0.0;
			for (std::size_t k = 1; k + 1 < points.size(); ++k)
			{
				const double x = points[k].x;
				const double h = 1e-7;
				const double left = (curve.Evaluate(x) - curve.Evaluate(x - h)) / h;
				const double right = (curve.Evaluate(x + h) - curve.Evaluate(x)) / h;
				largest_jump = std::max(largest_jump, std::abs(right - left) / std::max(1.0, std::abs(left)));
			}
			std::printf("%-40s largest change of slope at a point: %.1e\n", text, largest_jump);
			CHECK(largest_jump < 1e-3);
		}

		// Between the points it is curved: the brighter curve lies above the
		// straight lines from (0, 0) to (128, 191) and on to (255, 255)
		const ToneCurve brighter = Curve("0:0,128:191,255:255");
		CHECK(brighter.Evaluate(64) > 95.5 + 1.0);
		CHECK(brighter.Evaluate(192) > 223.0 + 1.0);
	}

	void TestParse()
	{
		ToneCurve curve;
		std::string error;
		CHECK(ToneCurve::Parse("0:0,128:191,255:255", curve, error));
		CHECK(curve.Points() == std::vector<cv::Point>({{0, 0}, {128, 191}, {255, 255}}));
		CHECK(curve.ToString() == "0:0,128:191,255:255");

		// Spaces are allowed, and the points can be in any order
		CHECK(ToneCurve::Parse(" 255 : 255 , 0:0,128:191 ", curve, error));
		CHECK(curve.ToString() == "0:0,128:191,255:255");

		struct Bad
		{
			const char* text;
			const char* message;
		};
		const Bad bad[] = {
			{"", "no points given"},
			{"   ", "no points given"},
			{"0:0", "at least two points are needed"},
			{"0:0,0:10", "two points have x = 0"},
			{"0:0,128:5,128:6,255:255", "two points have x = 128"},
			{"0:0,256:255", "'256:255' is not a point"},
			{"0:-1,255:255", "'0:-1' is not a point"},
			{"+1:0,255:255", "'+1:0' is not a point"},
			{"0:0,128,255:255", "'128' is not a point"},
			{"0:0;255:255", "'0:0;255:255' is not a point"},
			{"a:b,255:255", "'a:b' is not a point"},
			{"0:0,,255:255", "'' is not a point"},
			{"0:0,255:255,", "'' is not a point"},
			{"1.5:2,255:255", "'1.5:2' is not a point"},
			{"0:0:0,255:255", "'0:0:0' is not a point"},
			{"0:0,0255:255", "'0255:255' is not a point"},
		};
		const ToneCurve before = Curve("0:10,255:245");
		for (const Bad& b : bad)
		{
			ToneCurve parsed = before;
			error.clear();
			const bool ok = ToneCurve::Parse(b.text, parsed, error);
			std::printf("%-26s -> %s\n", (std::string("\"") + b.text + "\"").c_str(), ok ? "accepted" : error.c_str());
			CHECK(!ok);
			CHECK(error.find(b.message) != std::string::npos);
			CHECK(parsed.ToString() == before.ToString());
		}

		// The constructor checks the same conditions
		const std::vector<std::vector<cv::Point>> invalid = {
			{},
			{{0, 0}},
			{{0, 0}, {0, 5}},
			{{0, 0}, {300, 0}},
			{{0, 0}, {255, -1}},
		};
		for (const std::vector<cv::Point>& points : invalid)
		{
			bool thrown = false;
			try
			{
				ToneCurve curve_from_points(points);
			}
			catch (const std::invalid_argument&)
			{
				thrown = true;
			}
			CHECK(thrown);
		}
	}

	void TestEditing()
	{
		ToneCurve curve;

		// A new point goes in order of x
		CHECK(curve.Insert({128, 191}) == 1);
		CHECK(curve.ToString() == "0:0,128:191,255:255");
		CHECK(LutValues(curve)[128] == 191);

		// A point at the x of another moves that point
		CHECK(curve.Insert({128, 100}) == 1);
		CHECK(curve.ToString() == "0:0,128:100,255:255");

		// Positions outside the window are clamped to levels 0 to 255
		CHECK(curve.Insert({-5, 300}) == 0);
		CHECK(curve.ToString() == "0:255,128:100,255:255");
		curve = ToneCurve();
		curve.Insert({128, 191});

		// Points cannot pass their neighbours or leave the range
		CHECK(curve.Move(1, {300, 300}) == cv::Point(254, 255));
		CHECK(curve.Move(1, {-10, -10}) == cv::Point(1, 0));
		CHECK(curve.Move(0, {200, 10}) == cv::Point(0, 10));
		CHECK(curve.Move(2, {1, 100}) == cv::Point(2, 100));
		CHECK(curve.ToString() == "0:10,1:0,2:100");
		CHECK(curve.Move(2, {255, 255}) == cv::Point(255, 255));

		// Random drags keep the points in order with distinct x
		cv::RNG rng(7);
		for (int i = 0; i < 6; ++i)
			curve.Insert({rng.uniform(0, 256), rng.uniform(0, 256)});
		bool ordered = true;
		for (int drag = 0; drag < 2000; ++drag)
		{
			const int index = rng.uniform(0, static_cast<int>(curve.Points().size()));
			curve.Move(index, {rng.uniform(-50, 306), rng.uniform(-50, 306)});
			const std::vector<cv::Point>& points = curve.Points();
			for (std::size_t k = 1; k < points.size(); ++k)
				ordered = ordered && points[k - 1].x < points[k].x;
			ordered = ordered && MissedPoints(curve) == 0;
		}
		CHECK(ordered);

		// The nearest point within the distance, if any
		curve = Curve("0:0,128:191,255:255");
		CHECK(curve.Nearest({130, 190}, 5.0) == 1);
		CHECK(curve.Nearest({140, 190}, 5.0) == -1);
		CHECK(curve.Nearest({3, 4}, 5.0) == 0);

		// At least two points are kept
		CHECK(curve.Remove(1));
		CHECK(curve.ToString() == "0:0,255:255");
		CHECK(!curve.Remove(0));
		CHECK(curve.ToString() == "0:0,255:255");

		bool thrown = false;
		try
		{
			curve.Move(2, {0, 0});
		}
		catch (const std::out_of_range&)
		{
			thrown = true;
		}
		CHECK(thrown);
	}

	void TestApply()
	{
		// One table for all channels, as cv::LUT applies a one-channel table
		const cv::Mat lut = Curve("0:255,255:0").Lut();
		cv::Mat image(4, 5, CV_8UC3, cv::Scalar(10, 100, 200));
		cv::Mat result;
		cv::LUT(image, lut, result);
		CHECK(result.size() == image.size() && result.type() == CV_8UC3);
		CHECK(result.at<cv::Vec3b>(2, 3) == cv::Vec3b(245, 155, 55));

		const cv::Mat brighter = Curve("0:0,128:191,255:255").Lut();
		cv::Mat gray(3, 3, CV_8UC1, cv::Scalar(128));
		cv::LUT(gray, brighter, result);
		CHECK(result.type() == CV_8UC1 && result.at<uchar>(1, 1) == 191);
	}

	int Brightness(const cv::Mat& image, cv::Point2d p)
	{
		const cv::Vec3b& pixel = image.at<cv::Vec3b>(cvRound(p.y), cvRound(p.x));
		return pixel[0] + pixel[1] + pixel[2];
	}

	void TestDrawing()
	{
		// Pixels and levels convert both ways; positions outside are clamped
		bool round_trip = true;
		for (int x = 0; x < 256; ++x)
		{
			for (int y = 0; y < 256; y += 5)
			{
				const cv::Point2d pixel = CurvePixel(x, y);
				round_trip = round_trip && CurveLevels(cvRound(pixel.x), cvRound(pixel.y)) == cv::Point(x, y);
			}
		}
		CHECK(round_trip);
		CHECK(CurveLevels(-100, -100) == cv::Point(0, 255));
		CHECK(CurveLevels(10000, 10000) == cv::Point(255, 0));

		const ToneCurve brighter = Curve("0:0,128:191,255:255");
		const cv::Mat image = DrawCurve(brighter, cv::Mat());
		CHECK(image.rows == kCurveImageSize && image.cols == kCurveImageSize && image.type() == CV_8UC3);

		// The curve goes through its points and not along the diagonal
		CHECK(Brightness(image, CurvePixel(64, brighter.Evaluate(64))) < 200);
		CHECK(Brightness(image, CurvePixel(200, brighter.Evaluate(200))) < 200);
		CHECK(Brightness(image, CurvePixel(100, 140)) == 3 * 255);

		// The histogram is drawn behind the curve
		cv::Mat histogram = cv::Mat::zeros(256, 1, CV_32F);
		histogram.at<float>(50) = 1000.0f;
		const cv::Mat with_histogram = DrawCurve(brighter, histogram);
		CHECK(Brightness(image, CurvePixel(50, 20)) == 3 * 255);
		CHECK(Brightness(with_histogram, CurvePixel(50, 20)) < 3 * 255);
		CHECK(Brightness(with_histogram, CurvePixel(60, 20)) == 3 * 255);

		// The selected point is filled, the others are hollow
		const cv::Mat selected = DrawCurve(brighter, cv::Mat(), 1);
		CHECK(Brightness(image, CurvePixel(128, 191)) > 600);
		CHECK(Brightness(selected, CurvePixel(128, 191)) < 200);

		// Lines through the mouse position, in red
		const cv::Mat with_cursor = DrawCurve(brighter, cv::Mat(), -1, cv::Point(300, 400));
		const cv::Vec3b& line = with_cursor.at<cv::Vec3b>(400, 2);
		CHECK(line[2] > 150 && line[0] < 120 && line[1] < 120);
		CHECK(with_cursor.at<cv::Vec3b>(2, 300) == line);
	}

	struct Test
	{
		const char* name;
		void (*run)();
	};

	const Test kTests[] = {
		{"identity", TestIdentity},
		{"control_points", TestControlPoints},
		{"uneven_spacing", TestUnevenSpacing},
		{"monotone", TestMonotone},
		{"flat_ends", TestFlatEnds},
		{"smooth", TestSmooth},
		{"parse", TestParse},
		{"editing", TestEditing},
		{"apply", TestApply},
		{"drawing", TestDrawing},
	};
}

int main(int argc, char** argv)
{
	const char* only = argc > 1 ? argv[1] : nullptr;
	bool found = false;
	for (const Test& test : kTests)
	{
		if (only && std::strcmp(only, test.name) != 0)
			continue;
		found = true;
		const int before = g_failures;
		std::printf("[ RUN  ] %s\n", test.name);
		test.run();
		std::printf("[ %s ] %s\n", g_failures == before ? " OK " : "FAIL", test.name);
	}
	if (!found)
	{
		std::fprintf(stderr, "Unknown test: %s\n", only);
		return 1;
	}
	return g_failures == 0 ? 0 : 1;
}
