/**
 * @file main.cpp
 * Computer Vision Assignment 1
 * Dr. Abid
 * @author Michael Perez
 */

#include "opencv2/imgproc.hpp"
#include "../Histogram1D.h"
#include "../VideoProcessor.h"
#include "../overhauser.hpp"
#include "../overhauser.cpp"

using namespace cv;
using namespace std;

// Declaring a function that is used before it is defined
void applyCurve(Mat& img, Mat& out);

// Global variables
const string video_name = "barriers.avi";
Mat frame; // current video frame
Mat preview; // preview of one edited frame
Mat curvesImg;
Mat histImage;
vector<Point2f> pts;
int selectedPt = -1;
CRSpline* spline = 0;
unsigned char LUT_GRAY[256];
bool playVideo = false;
char key;
Histogram1D h;

// Compare two points by x value
bool mycomp(Point2f p1, Point2f p2)
{
	return p1.x < p2.x;
}

// Calculate euclidean distance
float dist(Point2f p1, Point2f p2)
{
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

// Returns the distance to the nearest point
int findNearestPt(Point2f pt, float maxDist)
{
	float ptx = pt.x;
	float minDist = FLT_MAX;
	int ind = -1;
	for (int i = 0; i < pts.size(); ++i)
	{
		float d = dist(pt, pts[i]);
		if (minDist > d)
		{
			ind = i;
			minDist = d;
		}
	}
	if (minDist > maxDist)
	{
		ind = -1;
	}

	return ind;
}

// 
float F(float t, float x, CRSpline* spline)
{
	vec3 rv = spline->GetInterpolatedSplinePoint(t);
	return x - rv.x;
}

// Solves for 
float solveForX(float x, CRSpline* slpine)
{
	float a = -1.0f, b = 1.0, c, e = 1e-2;
	c = (a + b) / 2;
	while ((fabs(b - a) > e) && (F(c, x, slpine) != 0))
	{
		if (F(a, x, slpine) * F(c, x, slpine) < 0)
		{
			b = c;
		}
		else
		{
			a = c;
		}
		c = (a + b) / 2;
	}
	return c;
}

int ind = -1;

void mouseHandler(int event, int x, int y, int flags, void* param)
{
	Point2f m;
	m.x = x;
	m.y = y;
	curvesImg = Scalar(0, 0, 0);

	// Events are mouse button presses, mouse button releases, mouse movements, etc.
	// 
	switch (event)
	{
	// Remove point
	case EVENT_RBUTTONDOWN:
		ind = findNearestPt(m, 5);
		if (ind == -1)
		{ }
		else
		{
			pts.erase(pts.begin() + ind);
			ind = -1;
		}
		break;
	// Add new point
	case EVENT_LBUTTONDOWN:
		ind = findNearestPt(m, 5);
		if (ind == -1)
		{
			pts.push_back(m);
			selectedPt = pts.size() - 1;
		}
		else
			selectedPt = ind;
		break;

	// Apply the Curve Tool effect to the preview window continuously as the mouse moves
	case EVENT_MOUSEMOVE:
		if (ind != -1)
		{
			// Apply curve to the frame
			applyCurve(frame, preview);

			// Display preview only if there are enough points to form a curve
			if (pts.size() > 2)
			{
				imshow("Preview Interface", preview);
				histImage = h.getHistogramImage(preview);
				imshow("Preview Histogram", histImage);
			}
			else
			{
				imshow("Preview Interface", frame);
				histImage = h.getHistogramImage(frame);
				imshow("Preview Histogram", histImage);
			}
			// Have the point follow the mouse
			pts[selectedPt].x = m.x;
			pts[selectedPt].y = m.y;
		}
		break;
	case EVENT_LBUTTONUP:
		// Apply curve to the frame
		applyCurve(frame, preview);

		// Display preview only if there are enough points to form a curve
		if (pts.size() > 2)
		{
			imshow("Preview Interface", preview);
			histImage = h.getHistogramImage(preview);
			imshow("Preview Histogram", histImage);
		}
		else
		{
			imshow("Preview Interface", frame);
			histImage = h.getHistogramImage(frame);
			imshow("Preview Histogram", histImage);
		}
		ind = -1;
		break;
	case EVENT_RBUTTONUP:
		// Apply curve to the frame
		applyCurve(frame, preview);

		// Display preview only if there are enough points to form a curve
		if (pts.size() > 2)
		{
			imshow("Preview Interface", preview);
			histImage = h.getHistogramImage(preview);
			imshow("Preview Histogram", histImage);
		}
		else
		{
			imshow("Preview Interface", frame);
			histImage = h.getHistogramImage(frame);
			imshow("Preview Histogram", histImage);
		}
		ind = -1;
		break;
	}

	// Make sure at least two points are at the right and left end of the windows
	// so that the curve is horizontal
	sort(pts.begin(), pts.end(), mycomp);
	if (pts.size() > 0)
	{
		pts[pts.size() - 1].x = 255;
		pts[0].x = 0;
	}

	// Draw points
	for (int i = 0; i < pts.size(); ++i)
		circle(curvesImg, pts[i], 5, Scalar(0, 0, 255), 1, 8, 0);

	// Draw/Replace the curve
	if (spline) { delete spline; }
	spline = new CRSpline();

	// Add spline points to the spline
	for (int i = 0; i < pts.size(); ++i)
	{
		vec3 v(pts[i].x, pts[i].y, 0);
		spline->AddSplinePoint(v);
	}


	vec3 rv_last(0, 0, 0);
	if (pts.size() > 2) // if there are enough points to form a line
	{
		for (int i = 0; i < 256; ++i)
		{
			// Calculates the y value of each point on the curve
			float t = solveForX(i, spline);

			// Creates a vector
			vec3 rv = spline->GetInterpolatedSplinePoint(t);

			// Make sure in 0-255 range
			if (rv.y > 255) { rv.y = 255; }
			if (rv.y < 0) { rv.y = 0; }
			unsigned char I = (unsigned char)(rv.y);

			// Populate look up table
			LUT_GRAY[i] = 255 - I;
			if (i > 0)
			{
				// Display line
				line(curvesImg, Point(rv.x, rv.y), Point(rv_last.x, rv_last.y), Scalar(0, 0, 255), 1);
			}
			rv_last = rv;
		}
	}

	// Draw the lines that follow the mouse
	Scalar col = Scalar(0, 0, 255);
	line(curvesImg, Point(0, m.y), Point(curvesImg.cols, m.y), col, 1);
	line(curvesImg, Point(m.x, 0), Point(m.x, curvesImg.rows), col, 1);

	// Display the curve tool image in the window
	imshow("Video Editing Curve Tool", curvesImg);
}

// processing function
void applyCurve(Mat& img, Mat& out)
{
	// Canny edges transformation for testing purposes:
	//Canny(img, out, 100, 200);
	//threshold(out, out, 128, 255, THRESH_BINARY_INV);

	if (pts.size() > 2)
		LUT(img, Mat(256, 1, CV_8UC1, LUT_GRAY), out);
	else
		out = img;
}

/**
 * @function main
 */
int main(int argc, char** argv)
{
	// Display windows
	namedWindow("Video");
	namedWindow("Video Editing Curve Tool");
	namedWindow("Preview Interface");
	curvesImg = Mat::zeros(256, 256, CV_8UC3);
	setMouseCallback("Video Editing Curve Tool", mouseHandler, NULL);

	// Open video file
	VideoCapture capture(video_name);

	// Check if video was opened
	if (!capture.isOpened())
	{
		cout << "Error opening video file" << endl;
		cin.get(); // Wait for key press
		return -1;
	}

	// Display first frame of video in two windows
	capture >> frame;
	namedWindow("Video", WINDOW_NORMAL);
	imshow("Video", frame);
	imshow("Preview Interface", frame);

	// Display Histogram of first frame in both histogram windows
	Histogram1D h;
	histImage = h.getHistogramImage(frame);
	imshow("Video Histogram", histImage);
	imshow("Preview Histogram", histImage);

	// Initialize Look up table
	for (int i = 0; i < 256; ++i)
		LUT_GRAY[i] = i;

	// Set up Curve Tool window
	curvesImg = Mat::zeros(256, 256, CV_8UC3);
	setMouseCallback("Video Editing Curve Tool", mouseHandler, NULL);

	// Main loop
	while(1)
	{
		key = waitKey(10); // Scan for key press every 10 ms

		capture.set(CAP_PROP_POS_FRAMES, 0);
		capture >> frame;

		if (key == ' ') // Play video
			playVideo = !playVideo;

		if (key == 'p') // Apply curve tool to video
		{
			//process on entire video
			// Create instance
			VideoProcessor processor;

			// Open video file
			processor.setInput(video_name);

			// Play the video at the original frame rate
			processor.setDelay(1000. / processor.getFrameRate());

			// Set the frame processor callback function
			processor.setFrameProcessor(applyCurve);

			// output a video
			processor.setOutput(video_name, 0, processor.getFrameRate(), false);

			// Start the process
			processor.run();

			key = waitKey(10000);

			//reopen video file
			VideoCapture capture(video_name);

			// display first frame of video in two windows
			capture >> frame;
			namedWindow("Video", WINDOW_NORMAL);
			imshow("Video", frame);
			imshow("Preview Interface", frame);
		}

		if (playVideo)
		{
			//reopen video file
			VideoCapture capture(video_name);
			
			if (!capture.isOpened())
			{
				cout << "Error opening video file" << endl;
				cin.get(); //wait for key press
				return -1;
			}

			// Calculate the frame rate and delay
			double rate = capture.get(CAP_PROP_FPS);
			int delay = 1000 / rate;

			while (1)
			{
				if(!capture.read(frame)) // Read frames and check if last frame
				{
					playVideo = false;

					capture.set(CAP_PROP_POS_FRAMES, 0);
					capture >> frame;
					imshow("Video", frame);
					break;
				}

				// Display frame
				imshow("Video", frame);

				// Display histogram
				Mat histImage = h.getHistogramImage(frame);
				imshow("Video Histogram", histImage);

				// Wait the required time in between frames
				waitKey(delay);
			}
		}
	}
	// Close the video file.
	capture.release();

	return 0;
}
