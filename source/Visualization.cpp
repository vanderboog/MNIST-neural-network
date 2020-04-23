#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void displayImage(const cv::Mat &im, const std::string strDigit)
{
	////////////////////////////////////////////////////////////////////////////////////////////
	/// Scales the image into readable size and prints the network result onto image
	////////////////////////////////////////////////////////////////////////////////////////////

	cv::Mat imScaled;
	cv::resize(im, imScaled, cv::Size(280, 280));

	// Write digit label on image
	cv::putText(imScaled,
				strDigit,
				cv::Point(5, 20),				// Coordinates
				cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
				1.0,							// Scale. 2.0 = 2x bigger
				cv::Scalar(255, 0, 0),			// BGR Color
				1);								// Line Thickness (Optional)

	/// Write required action to close the program
	cv::putText(imScaled,
				"Press <q> to close",
				cv::Point(5, 275),				// Coordinates
				cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
				0.5,							// Scale. 2.0 = 2x bigger
				cv::Scalar(255, 0, 0),			// BGR Color
				1);								// Line Thickness (Optional)

	cv::imshow("Test image", imScaled);
}