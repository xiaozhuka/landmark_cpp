#ifndef LANDMARK_DETECTOR_HPP_
#define LANDMARK_DETECTOR_HPP_
#include <caffe/caffe.hpp>
#include "opencv2/core/core.hpp"

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
    /* Detect landmark. */
    std::vector<T> Detect(const cv::Mat& img);
};
#endif