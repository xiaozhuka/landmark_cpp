#include "landmark_detector.hpp"

template<typename T>
LandmarkDetector<T>::LandmarkDetector(const string& model_file,
	const string& trained_file) {
	/* Load the network through *.proto file. */
	net_.reset(new Net<float>(model_file, TEST));
	net_.CopyTrainedLayersFrom(trained_file);
	Blob<T>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/**
	 * What does Reshape do??
	 * See https://stackoverflow.com/questions/35100458/caffe-layersetup-and-reshape
	 * *LayerSetUp*
	    (a) Verify the layer has exactly the right number of input/output blobs
		(b) Read the parameters of the layer from the prototxt
		(c) Initialize internal parameters
	 * *Reshape*
		used to allocate memory for parameters and output blobs 
		and can be called even after the net was setup.
		For instance, it is common for detection networks to change the input shape, 
		thus Reshapeing all consequent blobs
	 */
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	net_->Reshape();
}

template<typename T>
void LandmarkDetector<T>::WrapInputLayer() {
	Blob<float>* input_layer = net_->input_blobs()[0];
	int width = input_layer->width();
	int height = input_layer->height();
	// data ptr
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels_->push_back(channel);
		input_data += width * height;
	}
}

template<typename T>
void LandmarkDetector<T>::Preprocess(const cv::Mat& img) {
	cv::Mat subtracted;
	cv::Mat tmp_mean;
	cv::Mat tmp_std;
	// calculate mean, std for every channels.
	cv::meanStdDev(img, tmp_mean, tmp_std);
	// subtract
	cv::subtract(img, tmp_mean, subtracted);
	cv::Mat normalized;
	sv::divide(subtracted, tmp_std, normalized);
	cv::split(normalized, *input_channels_);
}

template<typename T>
std::vector<T> LandmarkDetector<T>::Detect(const cv::Mat& img) {
	Blob<T>* input_layer = net_->input_blobs[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	net_.Reshape();
	// link img to input, input_channels_
	WrapInputLayer();
	Preprocess(img);
	net_.Forward();

	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	return std::vector<T>begin;
}

template<typename T>
std::vector<T> LandmarkDetector<T>::getLandmark(const cv::Mat& img) {
	return Detect(img);
}
