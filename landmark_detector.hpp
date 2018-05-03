#ifndef LANDMARK_DETECTOR_HPP_
#define LANDMARK_DETECTOR_HPP_
#include "caffe/caffe.hpp"
#include "opencv2/core/core.hpp"
#include <string>

using namespace caffe;
using std::string;

/**
* @brief A Detector implemented by Caffe. Given an image
*        will calculate key points of human face.
*/
template<typename T>
class LandmarkDetector {
public:
	/**
	* @brief Given model_file(*.proto) as caffe net of trainer and
	*        trained_file(*.caffemodel).
	*/
	LandmarkDetector(const string& model_file,
					 const string& trained_file);

	/**
	* @brief Return landmark in std::vector.
	*/
	std::vector<T> getLandmark(const cv::Mat& img);
private:
	shared_ptr<Net<T> > net_;
	int num_channels_;
	cv::Size input_geometry_;
	std::vector<cv::Mat>* input_channels_;

	/**
	 * @brief Detect landmark.
	 * Use img as input, do forward.
	 */
	std::vector<T> Detect(const cv::Mat& img);

	/**
	 * @brief Detect landmark.
	 * Use void* as input, do forward.
	 */
	std::vector<T> Detect(const void *data);

	/**
	 * Wrap the input layer of the network in separate cv::Mat objects
	 * (one per channel).
	 */
	void WrapInputLayer();

	/**
	* @brief Pre-process image, make it to input_channels_.
	* Subtract mean, divided by std, for each channels separately.
	* Write the separate channels directly to the input layer.
	*/
	void Preprocess(const cv::Mat& img);
};
#endif