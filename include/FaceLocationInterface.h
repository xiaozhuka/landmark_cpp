#ifndef _EVGPU_FACELOCATION_H_
#define _EVGPU_FACELOCATION_H_

#if defined(_WIN32) || defined(_WIN64)
#ifdef   EVGPUFACELOCATION_EXPORTS
#define   EVGPUFACELOCATION_DLL_API   __declspec(dllexport)  
#else
#define   EVGPUFACELOCATION_DLL_API   __declspec(dllimport)  
#endif 
#else
#define   EVGPUFACELOCATION_DLL_API
#endif

#include <vector>
using std::vector;

//[20171219]基於416新策略(GPU端做五官定位及特徵提取)集成到480版本
//#define REPLACE_BY_NEW_STRATEGY
//#define EV_FACE_PHONE_DETECT // 20180119是否進行活體檢查
#ifdef REPLACE_BY_NEW_STRATEGY
#define MAX_LANDMARK_NUM 25
#define MAX_FEATURE_DIM_NUM 128
#endif

typedef enum emFacePose{
	FACE_POS_NONE = 0,	//未分类 或 非人臉
	FACE_POS_FRONT,	//正脸
	FACE_POS_YAW,	//侧脸
	FACE_POS_PITCH,	//低头
	FACE_POS_OTHER	//其他
}FACE_POSE;

struct SFaceRectInfo 
{
	SFaceRectInfo()
	{
		nX = -1;
		nY = -1;
		nWidth = -1;
		nHeight = -1;
		nPoseType = -1;
		nPhoneDetect = -1;
	}
	int  nX ;
	int  nY ;
	int  nWidth ;
	int nHeight ;
	int nPoseType;// [20170714] Face Pose Classify
	int nPhoneDetect;//20180117活體檢測是否檢測到手機
};
typedef vector<SFaceRectInfo>  VEC_FACERECTINFO ;

enum  EImgType
{
	EImgType_Rgb24		= 0 ,//彩图
	EImgType_Gray		= 1 ,//灰度图、Y变量
};

enum ERRCODE
{
	ERR_TYPE_SUCC		= 0 ,//成功
	ERR_TYPE_LOADFILTER = 1 ,//加载分类器失败
	ERR_TYPE_BUFFEMPTY  = 2 ,//buffer为空
	ERR_TYPE_NOINIT		= 3	,//未初始化
	ERR_TYPE_NODEV		= 4 ,//没有对应index的GPU设备
	ERR_TYPE_DISABLE	= 5 ,//GPU设备已经用光，能力不足
};
//初始化针对同一个进程只有第一次生效，剩下都是计数模式增长使用实例个数(保证GPU和CPU不混合使用)
//初始化 nDevType:0: GPU 1:CPU  dConfidence:置信度 [20170518] nProcessIdx:GPU Server的實例序號(默認為0),内部根據進程分配GPU設備
int	EVGPUFACELOCATION_DLL_API InitializeFaceLib(double dConfidence, int nDevType = 0, int nProcessIdx = 0);

//执行定位 Image buffer(如果有ROI需要给ROI的外接矩形的buffer), 宽 ,高 ， 图片类型，缩小倍数因子(必须大于等于1，等于1不缩)，反馈Rect数组
int EVGPUFACELOCATION_DLL_API LocationFace(unsigned char* szBuf,int nWidth,int nHeight,EImgType eImgType,int nShrink,VEC_FACERECTINFO& vecRect) ;
//释放 这个是计数模式的，只有计数变成0 GPU设备才会被停止
int EVGPUFACELOCATION_DLL_API ReleaseFaceLib();


#endif