#include "landmark_detector.hpp"
#include <string>
#include <fstream>
#include <opencv2\opencv.hpp>

using std::string;

int main() {
	const string model_file = "./finetune/train.prototxt";
	const string trained_file = "./finetune/model/_iter_10000.caffemodel";
	LandmarkDetector<float> landmarkDetector = LandmarkDetector<float>(model_file, trained_file);

	const string validation_parent_path = "./validation_result/";

	const string val_img_list_file_txt_path = "./dataset/img_list_val.txt";
	fstream val_img_list_fin(val_img_list_file_txt_path, std::ios::in);
	string single_img_path;
	cv::Mat img;
	std::vector<float> single_landmark;
	
	while (std::getline(val_img_list_fin, single_img_path)) {
		img = cv::imread(single_img_path);
		if (img.data) {
			single_landmark = landmarkDetector.getLandmark(img);
		}
		else{
			printf("Wrong in read %s", single_img_path.c_str());
			continue;
		}
		//for (auto iter = single_landmark.begin(); iter != single_landmark.end(); ++iter) {
		//	float x = *iter;
		//	++iter;
		//	float y = *iter;
		//}
		for (int i = 0; i < single_landmark.size(); i = i + 2) {
			float x = single_landmark[i];
			float y = single_landmark[i + 1];
			
			cv::Point center = cv::Point((int)(x * img.cols), (int)(y * img.rows));
			cv::circle(img, center, 1, cv::Scalar(0, 0, 255));
		}

		int basename_pos = single_img_path.find_last_of('/\\');
		string basename(single_img_path.substr(basename_pos + 1));
		cv::imwrite(validation_parent_path + basename, img);
	}
}