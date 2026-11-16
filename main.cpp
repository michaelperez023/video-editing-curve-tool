// main.cpp
//
// Video editing curve tool: a Curves tool, like the one in Adobe Photoshop,
// for video. The mouse adds, moves and removes the control points of a tone
// curve in the curve window, and a frame of the video is shown with the curve
// applied, next to the original, with the histograms of both. The curve can
// then be applied to every frame and written to a new video file. With
// --no-gui the curve is given on the command line and no windows are opened.
//
// Michael Perez, CAP 4410 Computer Vision, Assignment 1, Florida Polytechnic
// University (2019). In 2026 saving was fixed, the curve code and the point
// editing were rewritten, and command-line options and a headless mode were
// added.

#include "Curve.h"
#include "Histogram1D.h"
#include "VideoProcessor.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace
{
	namespace fs = std::filesystem;

	// The windows of the 2019 version
	const char* const kCurveWindow = "Video Editing Curve Tool";
	const char* const kVideoWindow = "Video";
	const char* const kVideoHistogramWindow = "Video Histogram";
	const char* const kPreviewWindow = "Preview Interface";
	const char* const kPreviewHistogramWindow = "Preview Histogram";

	// A click within this distance (in levels) of a control point picks it
	constexpr double kPickDistance = 5.0;

	// Larger videos are shown scaled down to fit in this size
	const cv::Size kMaxDisplaySize(640, 480);

	struct Options
	{
		std::string input = "barriers.avi";
		std::string output;      // the video to write; empty means <input>_curves.avi
		std::string images_dir;  // where to write the PNG images; empty means none
		ToneCurve curve;         // the identity unless --curve is given
		bool gui = true;
	};

	void PrintUsage(std::ostream& out, const char* program)
	{
		out << "Usage: " << program << " [options] [video]\n"
		    << "\n"
		    << "Applies a tone curve to every frame of a video (default: barriers.avi in\n"
		    << "the current directory) and writes the result to a new video file.\n"
		    << "\n"
		    << "Options:\n"
		    << "  --curve POINTS  control points x:y,x:y,... that map input level x to\n"
		    << "                  output level y, from 0 to 255 (default 0:0,255:255,\n"
		    << "                  which changes nothing); e.g. 0:0,128:191,255:255\n"
		    << "  --out FILE      the video to write (default: <video>_curves.avi next to\n"
		    << "                  the input); it may not be the input video\n"
		    << "  --images DIR    also write the curve, the first frame (in the windows:\n"
		    << "                  the frame shown) before and after, and their\n"
		    << "                  histograms, as PNG files to DIR\n"
		    << "  --no-gui        do not open windows: apply the curve, write the video\n"
		    << "                  and exit\n"
		    << "  -h, --help      show this message\n"
		    << "\n"
		    << "In the curve window, click to add a point, drag a point to move it and\n"
		    << "right-click a point to remove it. Keys: Space plays and pauses the video,\n"
		    << "p applies the curve to the whole video and writes it, r resets the curve,\n"
		    << "q or Esc quits.\n";
	}

	enum class ParseResult { Run, Help, Error };

	ParseResult ParseArguments(int argc, char** argv, Options& options)
	{
		bool input_given = false;
		for (int i = 1; i < argc; ++i)
		{
			const std::string arg = argv[i];
			if (arg == "-h" || arg == "--help")
				return ParseResult::Help;

			if (arg == "--no-gui")
			{
				options.gui = false;
			}
			else if (arg == "--curve")
			{
				std::string error;
				if (i + 1 >= argc)
				{
					std::cerr << "--curve needs control points, for example 0:0,128:191,255:255\n";
					return ParseResult::Error;
				}
				if (!ToneCurve::Parse(argv[++i], options.curve, error))
				{
					std::cerr << "--curve: " << error << " (example: 0:0,128:191,255:255)\n";
					return ParseResult::Error;
				}
			}
			else if (arg == "--out" || arg == "--images")
			{
				const bool out = arg == "--out";
				if (i + 1 >= argc || argv[i + 1][0] == '\0')
				{
					std::cerr << arg << (out ? " needs a file name\n" : " needs a directory\n");
					return ParseResult::Error;
				}
				(out ? options.output : options.images_dir) = argv[++i];
			}
			else if (arg.size() > 1 && arg[0] == '-')
			{
				std::cerr << "Unknown option: " << arg << "\n";
				return ParseResult::Error;
			}
			else if (input_given)
			{
				std::cerr << "Only one video can be given\n";
				return ParseResult::Error;
			}
			else
			{
				options.input = arg;
				input_given = true;
			}
		}
		return ParseResult::Run;
	}

	// <input>_curves.avi in the directory of the input
	std::string DefaultOutput(const std::string& input)
	{
		const fs::path path(input);
		return (path.parent_path() / (path.stem().string() + "_curves.avi")).string();
	}

	// Whether `a` and `b` are the same existing file, even when they are
	// spelled differently or one is a link to the other
	bool SameFile(const std::string& a, const std::string& b)
	{
		std::error_code error;
		return fs::exists(a, error) && fs::exists(b, error) && fs::equivalent(a, b, error);
	}

	bool MakeDirectory(const fs::path& dir)
	{
		std::error_code error;
		if (dir.empty() || fs::is_directory(dir, error))
			return true;
		fs::create_directories(dir, error);
		if (error)
		{
			std::cerr << "Could not create " << dir.string() << ": " << error.message() << "\n";
			return false;
		}
		return true;
	}

	std::string Fixed(double value, int precision)
	{
		std::ostringstream text;
		text << std::fixed << std::setprecision(precision) << value;
		return text.str();
	}

	// Frame rate as written by people: 50, 29.97
	std::string Rate(double fps)
	{
		std::ostringstream text;
		text << std::setprecision(4) << fps;
		return text.str();
	}

	std::string FourccText(int fourcc)
	{
		std::string text;
		for (int i = 0; i < 4; ++i)
		{
			const char c = static_cast<char>((fourcc >> (8 * i)) & 0xFF);
			text += (c >= 32 && c < 127) ? c : '?';
		}
		return text;
	}

	// The frame as one gray channel, for the histograms
	cv::Mat Gray(const cv::Mat& frame)
	{
		if (frame.channels() == 1)
			return frame;
		cv::Mat gray;
		cv::cvtColor(frame, gray, frame.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
		return gray;
	}

	// A colour frame whose three channels are equal (a gray video stored in
	// colour, like barriers.avi) as one channel, which makes smaller PNG files
	cv::Mat Compact(const cv::Mat& frame)
	{
		if (frame.channels() != 3)
			return frame;
		std::vector<cv::Mat> channels;
		cv::split(frame, channels);
		if (cv::countNonZero(channels[0] != channels[1]) == 0 && cv::countNonZero(channels[1] != channels[2]) == 0)
			return channels[0];
		return frame;
	}

	// Mean and standard deviation of all the levels in a series of frames
	class LevelStats
	{
	public:
		void Add(const cv::Mat& frame)
		{
			cv::Scalar mean, deviation;
			cv::meanStdDev(frame.reshape(1), mean, deviation);
			const double n = static_cast<double>(frame.total()) * frame.channels();
			count_ += n;
			sum_ += mean[0] * n;
			sum_squares_ += (deviation[0] * deviation[0] + mean[0] * mean[0]) * n;
		}

		double Mean() const { return count_ > 0.0 ? sum_ / count_ : 0.0; }

		double StandardDeviation() const
		{
			if (count_ <= 0.0)
				return 0.0;
			const double mean = Mean();
			return std::sqrt(std::max(0.0, sum_squares_ / count_ - mean * mean));
		}

	private:
		double count_ = 0.0;
		double sum_ = 0.0;
		double sum_squares_ = 0.0;
	};

	// The frame processor that VideoProcessor calls for every frame: applies
	// the lookup table and keeps statistics of the levels before and after
	class CurveFrameProcessor final : public FrameProcessor
	{
	public:
		explicit CurveFrameProcessor(cv::Mat lut) : lut_(std::move(lut)) {}

		void process(cv::Mat& input, cv::Mat& output) override
		{
			cv::LUT(input, lut_, output);
			before_.Add(input);
			after_.Add(output);
		}

		const LevelStats& Before() const { return before_; }
		const LevelStats& After() const { return after_; }

	private:
		cv::Mat lut_;
		LevelStats before_;
		LevelStats after_;
	};

	// Applies the lookup table to every frame of `input` and writes the
	// result to `output`. If `window` is not empty the frames are shown in it
	// while they are written, and a key press stops. Returns true if every
	// frame was read, processed and written.
	//
	// The video is written as Motion JPEG, which OpenCV can always write to an
	// AVI file and which keeps the levels that the curve produced. Re-encoding
	// barriers.avi with its own MPEG-4 codec through OpenCV's FFmpeg backend
	// made the frames about 3 levels darker on average. The codec of the input
	// is used only if the file cannot be opened for Motion JPEG.
	bool SaveVideo(const std::string& input, const std::string& output, const cv::Mat& lut, bool color,
	               const std::string& window)
	{
		VideoProcessor processor;
		if (!processor.setInput(input))
		{
			std::cerr << "Could not open the video '" << input << "'.\n";
			return false;
		}

		double fps = processor.getFrameRate();
		if (!std::isfinite(fps) || fps <= 0.0)
			fps = 25.0;  // not every container gives a frame rate

		int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
		bool opened = processor.setOutput(output, codec, fps, color);
		char name[4];
		const int input_codec = processor.getCodec(name);
		if (!opened && input_codec != 0 && input_codec != -1 && input_codec != codec)
		{
			codec = input_codec;
			opened = processor.setOutput(output, codec, fps, color);
		}
		if (!opened)
		{
			std::cerr << "Could not open '" << output << "' to write a video.\n";
			return false;
		}

		CurveFrameProcessor curve(lut);
		processor.setFrameProcessor(&curve);
		if (!window.empty())
		{
			processor.displayOutput(window);
			processor.setDelay(1);  // keeps the windows responsive; a key press stops
		}

		const auto start = std::chrono::steady_clock::now();
		processor.run();
		processor.closeOutput();
		const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

		const long read = processor.getNumberOfProcessedFrames();
		const long written = processor.getNumberOfWrittenFrames();
		const cv::Size size = processor.getFrameSize();
		std::cout << "Wrote " << written << " of " << read << " frames to " << output << " ("
		          << FourccText(codec) << ", " << size.width << "x" << size.height << ", " << Rate(fps)
		          << " fps) in " << Fixed(elapsed.count(), 1) << " s\n";
		if (read > 0)
		{
			std::cout << "Levels over all frames: mean " << Fixed(curve.Before().Mean(), 1)
			          << ", standard deviation " << Fixed(curve.Before().StandardDeviation(), 1)
			          << " before; mean " << Fixed(curve.After().Mean(), 1) << ", standard deviation "
			          << Fixed(curve.After().StandardDeviation(), 1) << " after\n";
		}

		if (processor.writeFailed())
		{
			std::cerr << "Could not write frame " << written + 1 << " to " << output
			          << ": it does not have the size and number of channels of the video.\n";
			return false;
		}
		if (processor.isStopped())
		{
			std::cout << "Stopped by a key press; " << output << " has the first " << written << " frames.\n";
			return false;
		}
		if (read == 0)
		{
			std::cerr << "No frames could be read from '" << input << "'.\n";
			return false;
		}
		return written == read;
	}

	// Writes the curve window, the frame before and after the curve and the
	// histograms of both to `dir` as PNG files
	bool WriteImages(const std::string& dir, const ToneCurve& curve, const cv::Mat& frame, const cv::Mat& lut)
	{
		cv::Mat after;
		cv::LUT(frame, lut, after);
		const cv::Mat gray_before = Gray(frame);
		const cv::Mat gray_after = Gray(after);
		Histogram1D histogram;

		const std::pair<const char*, cv::Mat> files[] = {
			{"curve.png", DrawCurve(curve, histogram.getHistogram(gray_before))},
			{"frame_before.png", Compact(frame)},
			{"frame_after.png", Compact(after)},
			{"histogram_before.png", histogram.getHistogramImage(gray_before)},
			{"histogram_after.png", histogram.getHistogramImage(gray_after)},
		};
		bool ok = true;
		for (const auto& [name, image] : files)
		{
			const std::string path = (fs::path(dir) / name).string();
			if (!cv::imwrite(path, image))
			{
				std::cerr << "Could not write " << path << "\n";
				ok = false;
			}
			else
			{
				std::cout << "Wrote " << path << "\n";
			}
		}
		return ok;
	}

	// Whether the windows are OpenCV's own macOS (Cocoa) windows. OpenCV 4.10
	// and later can name the backend; with older versions the windows on macOS
	// are taken to be Cocoa windows.
	bool CocoaWindows()
	{
#if !defined(__APPLE__)
		return false;
#elif CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 10)
		return cv::currentUIFramework() == "COCOA";
#else
		return true;
#endif
	}

	// The windows of the interactive tool and what they show
	class CurveTool
	{
	public:
		CurveTool(const Options& options, cv::VideoCapture& capture, const cv::Mat& first_frame)
			: options_(options), capture_(capture), frame_(first_frame), curve_(options.curve), lut_(curve_.Lut())
		{
		}

		int Run()
		{
			CreateWindows();
			cv::setMouseCallback(kCurveWindow, OnMouse, this);
			ShowFrame();
			cv::waitKey(1);

			// Closing the curve window quits where the backend can tell: Win32 and
			// Qt report a closed window as not visible, and GTK does not report
			// visibility. OpenCV's macOS (Cocoa) windows cannot be closed, but from
			// OpenCV 4.7 on they report a minimised window as not visible, so with
			// them only q and Esc quit.
			const bool can_detect_close =
				!CocoaWindows() && cv::getWindowProperty(kCurveWindow, cv::WND_PROP_VISIBLE) >= 1;

			std::cout << "In the curve window, click to add a point, drag a point to move it and\n"
			          << "right-click a point to remove it. Keys: Space play/pause, p apply the curve\n"
			          << "to the video and write " << options_.output << ", r reset the curve, q or Esc quit."
			          << std::endl;

			using Clock = std::chrono::steady_clock;
			double fps = capture_.get(cv::CAP_PROP_FPS);
			if (!std::isfinite(fps) || fps <= 0.0)
				fps = 25.0;
			const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / fps));
			auto next_frame = Clock::now();

			while (true)
			{
				int wait = 20;  // milliseconds
				if (playing_)
				{
					const auto remaining =
						std::chrono::duration_cast<std::chrono::milliseconds>(next_frame - Clock::now()).count();
					wait = static_cast<int>(std::clamp<long long>(remaining, 1, 20));
				}
				const int key = cv::waitKey(wait);
				if (key == 'q' || key == 'Q' || key == 27)
					break;
				if (can_detect_close && cv::getWindowProperty(kCurveWindow, cv::WND_PROP_VISIBLE) < 1)
					break;

				if (key == ' ')
				{
					playing_ = !playing_;
					next_frame = Clock::now();
				}
				else if (key == 'p' || key == 'P')
				{
					Save();
				}
				else if (key == 'r' || key == 'R')
				{
					curve_ = ToneCurve();
					dragged_ = -1;
					CurveChanged();
				}

				if (playing_ && Clock::now() >= next_frame)
				{
					if (!NextFrame())
						playing_ = false;
					ShowFrame();
					next_frame += period;
					if (next_frame < Clock::now())
						next_frame = Clock::now();  // after a delay, do not try to catch up
				}
			}
			cv::destroyAllWindows();
			return 0;
		}

	private:
		// Creates the five windows, with the curve on the left, the preview
		// and its histogram on the top row and the video and its histogram on
		// the row below
		void CreateWindows()
		{
			cv::Size display = frame_.size();
			if (display.width > kMaxDisplaySize.width || display.height > kMaxDisplaySize.height)
			{
				const double scale = std::min(static_cast<double>(kMaxDisplaySize.width) / display.width,
				                              static_cast<double>(kMaxDisplaySize.height) / display.height);
				display = cv::Size(cvRound(display.width * scale), cvRound(display.height * scale));
			}
			const int video_flags = display == frame_.size() ? cv::WINDOW_AUTOSIZE : cv::WINDOW_NORMAL;

			cv::namedWindow(kCurveWindow, cv::WINDOW_AUTOSIZE);
			cv::namedWindow(kPreviewWindow, video_flags);
			cv::namedWindow(kPreviewHistogramWindow, cv::WINDOW_AUTOSIZE);
			cv::namedWindow(kVideoWindow, video_flags);
			cv::namedWindow(kVideoHistogramWindow, cv::WINDOW_AUTOSIZE);
			if (video_flags == cv::WINDOW_NORMAL)
			{
				cv::resizeWindow(kPreviewWindow, display.width, display.height);
				cv::resizeWindow(kVideoWindow, display.width, display.height);
			}

			const int gap = 10;
			const int title = 40;  // room for the title bar
			const int middle = kCurveImageSize + gap;
			const int right = middle + display.width + gap;
			const int second_row = display.height + title;
			cv::moveWindow(kCurveWindow, 0, 0);
			cv::moveWindow(kPreviewWindow, middle, 0);
			cv::moveWindow(kPreviewHistogramWindow, right, 0);
			cv::moveWindow(kVideoWindow, middle, second_row);
			cv::moveWindow(kVideoHistogramWindow, right, second_row);
		}

		static void OnMouse(int event, int x, int y, int /*flags*/, void* tool)
		{
			static_cast<CurveTool*>(tool)->Mouse(event, x, y);
		}

		void Mouse(int event, int x, int y)
		{
			if (saving_)
				return;
			cursor_ = cv::Point(x, y);
			const cv::Point level = CurveLevels(x, y);
			switch (event)
			{
			case cv::EVENT_LBUTTONDOWN:
				// Pick the point under the mouse, or add one there; either can
				// be dragged at once
				dragged_ = curve_.Nearest(level, kPickDistance);
				if (dragged_ < 0)
				{
					dragged_ = curve_.Insert(level);
					CurveChanged();
					return;
				}
				break;
			case cv::EVENT_MOUSEMOVE:
				if (dragged_ >= 0)
				{
					curve_.Move(dragged_, level);
					CurveChanged();
					return;
				}
				break;
			case cv::EVENT_LBUTTONUP:
				dragged_ = -1;
				break;
			case cv::EVENT_RBUTTONDOWN:
			{
				const int nearest = curve_.Nearest(level, kPickDistance);
				if (nearest >= 0 && curve_.Remove(nearest))
				{
					dragged_ = -1;
					CurveChanged();
					return;
				}
				break;
			}
			default:
				break;
			}
			ShowCurve();
		}

		// Reads the next frame. At the end of the video it goes back to the
		// first frame and returns false.
		bool NextFrame()
		{
			cv::Mat next;
			if (capture_.read(next) && !next.empty())
			{
				frame_ = next;
				return true;
			}
			if (capture_.open(options_.input) && capture_.read(next) && !next.empty())
				frame_ = next;
			return false;
		}

		void CurveChanged()
		{
			lut_ = curve_.Lut();
			ShowPreview();
			ShowCurve();
		}

		void ShowCurve()
		{
			cv::imshow(kCurveWindow, DrawCurve(curve_, frame_histogram_, dragged_, cursor_));
		}

		void ShowPreview()
		{
			cv::Mat preview;
			cv::LUT(frame_, lut_, preview);
			cv::imshow(kPreviewWindow, preview);
			cv::imshow(kPreviewHistogramWindow, histogram_.getHistogramImage(Gray(preview)));
		}

		// Shows the current frame, and everything computed from it
		void ShowFrame()
		{
			frame_histogram_ = histogram_.getHistogram(Gray(frame_));
			cv::imshow(kVideoWindow, frame_);
			cv::imshow(kVideoHistogramWindow, Histogram1D::getImageOfHistogram(frame_histogram_, 1));
			ShowPreview();
			ShowCurve();
		}

		void Save()
		{
			playing_ = false;
			saving_ = true;
			dragged_ = -1;
			std::cout << "Applying the curve " << curve_.ToString() << " to " << options_.input
			          << " (press any key to stop)" << std::endl;
			SaveVideo(options_.input, options_.output, lut_, frame_.channels() == 3, kPreviewWindow);
			if (!options_.images_dir.empty() && MakeDirectory(options_.images_dir))
				WriteImages(options_.images_dir, curve_, frame_, lut_);
			saving_ = false;
			ShowFrame();
		}

		const Options& options_;
		cv::VideoCapture& capture_;
		cv::Mat frame_;            // the current frame of the input video
		ToneCurve curve_;
		cv::Mat lut_;              // lookup table of curve_
		Histogram1D histogram_;
		cv::Mat frame_histogram_;  // histogram of frame_ (gray levels)
		int dragged_ = -1;         // control point being moved, or -1
		cv::Point cursor_{-1, -1}; // mouse position in the curve window
		bool playing_ = false;
		bool saving_ = false;
	};
}

int main(int argc, char** argv)
{
	Options options;
	switch (ParseArguments(argc, argv, options))
	{
	case ParseResult::Help:
		PrintUsage(std::cout, argv[0]);
		return 0;
	case ParseResult::Error:
		PrintUsage(std::cerr, argv[0]);
		return 2;
	case ParseResult::Run:
		break;
	}

	// Never write over the input video
	if (options.output.empty())
		options.output = DefaultOutput(options.input);
	if (SameFile(options.input, options.output))
	{
		std::cerr << "The output '" << options.output << "' is the input video; choose another file with --out.\n";
		return 1;
	}

	try
	{
		cv::VideoCapture capture(options.input);
		cv::Mat first_frame;
		if (!capture.isOpened() || !capture.read(first_frame) || first_frame.empty())
		{
			std::cerr << "Could not open the video '" << options.input << "'.\n";
			return 1;
		}
		const double fps = capture.get(cv::CAP_PROP_FPS);
		const double frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
		std::cout << "Video: " << options.input << " (" << first_frame.cols << "x" << first_frame.rows;
		if (std::isfinite(fps) && fps > 0.0)
			std::cout << ", " << Rate(fps) << " fps";
		if (std::isfinite(frames) && frames > 0.0)
			std::cout << ", " << static_cast<long>(frames) << " frames";
		std::cout << ")\n";

		// Create the directories first, so that a bad path is reported before
		// the video is processed
		if (!MakeDirectory(fs::path(options.output).parent_path()))
			return 1;
		if (!options.images_dir.empty() && !MakeDirectory(options.images_dir))
			return 1;

		if (options.gui)
		{
			CurveTool tool(options, capture, first_frame);
			return tool.Run();
		}

		const cv::Mat lut = options.curve.Lut();
		std::cout << "Curve: " << options.curve.ToString() << " (lookup table";
		for (const int level : {0, 64, 128, 192, 255})
			std::cout << " " << level << "->" << static_cast<int>(lut.at<uchar>(0, level));
		std::cout << ")\n";

		bool ok = SaveVideo(options.input, options.output, lut, first_frame.channels() == 3, "");
		if (!options.images_dir.empty())
			ok = WriteImages(options.images_dir, options.curve, first_frame, lut) && ok;
		return ok ? 0 : 1;
	}
	catch (const cv::Exception& e)
	{
		std::cerr << "OpenCV error: " << e.what() << "\n";
		if (options.gui)
			std::cerr << "To run without windows, use --no-gui.\n";
		return 1;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}
}
