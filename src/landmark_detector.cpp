#include "landmark_detector.hpp"

template<typename T>
LandmarkDetector::LandmarkDetector(const string& model_file,
                                   const string& trained_file,
                                   int num_channels=3) {
    /* Load the network through *.proto file. */
    net_.reset(new Net<float>(model_file, TEST));
    net_.CopyTrainedLayersFrom(trained_file);
    num_channels_ = num_channels;
}

template<typename T>
std::vector<T> LandmarkDetector::Detect(const cv::Mat& img) {
    // Get the input.
    Blob<T>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
    std::vector<cv::Mat> input_channels;
    net_->Forward();
    // Copy the output layer to a std::vector
    Blob<T>* output_layer = net_->output_blobs()[0];
    const T* begin = output_layer->cpu_data();
    return std::vector<T>begin;
}

template<typename T>
std::vector<float> LandmarkDetector::getLandmark(const cv::Mat& img) {
    return LandmarkDetector::Detect(img);
}
