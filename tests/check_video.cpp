// Checks a video written by curvetool against its input. Both videos are
// decoded to the end and their frames counted, and every output frame is
// compared with the input frame after the curve. The output is compressed, so
// that comparison allows a small error. With --images it also checks the PNG
// images written with --images, which are lossless: frame_before.png must be
// the first frame of the input and frame_after.png exactly that frame after
// the curve.
//
// Usage: check_video [--frames N] [--images DIR] input output curve
//   --frames N    the input and the output must have exactly N frames
// The exit status is 0 when every check passes.

#include "Curve.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace
{
	// Largest mean absolute difference, in levels, allowed between an output
	// frame and the input frame after the curve, and between their mean
	// levels. With the curve 0:0,128:191,255:255, the Motion JPEG output of
	// barriers.avi is within 1.3 and 0.1 levels, while a frame without the
	// curve is off by more than 35 levels and the next or previous frame by
	// more than 10.
	constexpr double kMaxFrameError = 4.0;
	constexpr double kMaxMeanError = 1.0;

	int g_failures = 0;

	void Fail(const std::string& message)
	{
		std::fprintf(stderr, "check failed: %s\n", message.c_str());
		++g_failures;
	}

	int Usage()
	{
		std::fprintf(stderr, "Usage: check_video [--frames N] [--images DIR] input output curve\n");
		return 2;
	}

	// The image `name` in `dir`, which must be `expected` exactly
	void CheckImage(const std::string& dir, const char* name, const cv::Mat& expected)
	{
		const std::string path = dir + "/" + name;
		const cv::Mat image = cv::imread(path, expected.channels() == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
		if (image.empty())
			Fail("could not read " + path);
		else if (image.size() != expected.size() || image.type() != expected.type() ||
		         cv::norm(image, expected, cv::NORM_INF) != 0.0)
			Fail(path + " is not the expected image");
		else
			std::printf("%s is the expected image\n", path.c_str());
	}

	// The image `name` in `dir`, which must have the given size
	void CheckImageSize(const std::string& dir, const char* name, cv::Size size)
	{
		const std::string path = dir + "/" + name;
		const cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
		if (image.empty())
			Fail("could not read " + path);
		else if (image.size() != size)
			Fail(path + " has the wrong size");
	}
}

int main(int argc, char** argv)
{
	long expected_frames = -1;
	std::string images_dir;
	int arg = 1;
	for (; arg + 1 < argc && argv[arg][0] == '-'; arg += 2)
	{
		const std::string option = argv[arg];
		if (option == "--frames")
			expected_frames = std::atol(argv[arg + 1]);
		else if (option == "--images")
			images_dir = argv[arg + 1];
		else
			return Usage();
	}
	if (argc - arg != 3)
		return Usage();
	const std::string input_path = argv[arg];
	const std::string output_path = argv[arg + 1];

	ToneCurve curve;
	std::string error;
	if (!ToneCurve::Parse(argv[arg + 2], curve, error))
	{
		std::fprintf(stderr, "Bad curve: %s\n", error.c_str());
		return 2;
	}
	const cv::Mat lut = curve.Lut();

	cv::VideoCapture input(input_path);
	cv::VideoCapture output(output_path);
	if (!input.isOpened() || !output.isOpened())
	{
		std::fprintf(stderr, "Could not open %s\n", (input.isOpened() ? output_path : input_path).c_str());
		return 1;
	}

	long input_frames = 0;
	long output_frames = 0;
	long mismatched = 0;
	double worst_error = 0.0;
	double worst_mean_error = 0.0;
	long worst_frame = 0;
	cv::Mat first_frame;
	cv::Size size;
	while (true)
	{
		cv::Mat in, out;
		const bool has_in = input.read(in) && !in.empty();
		const bool has_out = output.read(out) && !out.empty();
		if (has_in)
			++input_frames;
		if (has_out)
			++output_frames;
		if (!has_in && !has_out)
			break;
		if (!has_in || !has_out)
			continue;  // count the rest of the longer video

		if (input_frames == 1)
		{
			first_frame = in.clone();
			size = in.size();
		}
		if (in.size() != out.size() || in.type() != out.type())
		{
			++mismatched;
			continue;
		}

		// Mean absolute difference from the expected frame, and difference of the means
		cv::Mat expected;
		cv::LUT(in, lut, expected);
		const double error_levels = cv::norm(expected, out, cv::NORM_L1) / (static_cast<double>(out.total()) * out.channels());
		const double mean_error = std::abs(cv::mean(expected.reshape(1))[0] - cv::mean(out.reshape(1))[0]);
		if (error_levels > worst_error)
		{
			worst_error = error_levels;
			worst_frame = input_frames;
		}
		worst_mean_error = std::max(worst_mean_error, mean_error);
	}

	std::printf("%s: %ld frames\n%s: %ld frames\n", input_path.c_str(), input_frames, output_path.c_str(),
	            output_frames);
	std::printf("Largest mean absolute difference from the input frame after the curve: %.2f levels (frame %ld)\n",
	            worst_error, worst_frame);
	std::printf("Largest difference between the mean levels of the two: %.2f levels\n", worst_mean_error);

	if (input_frames == 0)
		Fail("no frames could be read from " + input_path);
	if (output_frames != input_frames)
		Fail("the output has " + std::to_string(output_frames) + " frames and the input " +
		     std::to_string(input_frames));
	if (expected_frames >= 0 && input_frames != expected_frames)
		Fail("expected " + std::to_string(expected_frames) + " frames");
	if (mismatched > 0)
		Fail(std::to_string(mismatched) + " output frames differ from the input in size or type");
	if (worst_error > kMaxFrameError)
		Fail("an output frame differs too much from the input frame after the curve");
	if (worst_mean_error > kMaxMeanError)
		Fail("the mean level of an output frame differs too much from the input frame after the curve");

	if (!images_dir.empty() && !first_frame.empty())
	{
		cv::Mat first_after;
		cv::LUT(first_frame, lut, first_after);
		CheckImage(images_dir, "frame_before.png", first_frame);
		CheckImage(images_dir, "frame_after.png", first_after);
		CheckImageSize(images_dir, "curve.png", cv::Size(kCurveImageSize, kCurveImageSize));
		CheckImageSize(images_dir, "histogram_before.png", cv::Size(256, 256));
		CheckImageSize(images_dir, "histogram_after.png", cv::Size(256, 256));
	}

	if (g_failures == 0)
		std::printf("All checks passed (%ld of %ld frames written, %dx%d)\n", output_frames, input_frames,
		            size.width, size.height);
	return g_failures == 0 ? 0 : 1;
}
