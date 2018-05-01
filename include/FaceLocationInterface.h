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

//[20171219]���416�²���(GPU������ٶ�λ��������ȡ)���ɵ�480�汾
//#define REPLACE_BY_NEW_STRATEGY
//#define EV_FACE_PHONE_DETECT // 20180119�Ƿ��M�л��w�z��
#ifdef REPLACE_BY_NEW_STRATEGY
#define MAX_LANDMARK_NUM 25
#define MAX_FEATURE_DIM_NUM 128
#endif

typedef enum emFacePose{
	FACE_POS_NONE = 0,	//δ���� �� ����Ę
	FACE_POS_FRONT,	//����
	FACE_POS_YAW,	//����
	FACE_POS_PITCH,	//��ͷ
	FACE_POS_OTHER	//����
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
	int nPhoneDetect;//20180117���w�z�y�Ƿ�z�y���֙C
};
typedef vector<SFaceRectInfo>  VEC_FACERECTINFO ;

enum  EImgType
{
	EImgType_Rgb24		= 0 ,//��ͼ
	EImgType_Gray		= 1 ,//�Ҷ�ͼ��Y����
};

enum ERRCODE
{
	ERR_TYPE_SUCC		= 0 ,//�ɹ�
	ERR_TYPE_LOADFILTER = 1 ,//���ط�����ʧ��
	ERR_TYPE_BUFFEMPTY  = 2 ,//bufferΪ��
	ERR_TYPE_NOINIT		= 3	,//δ��ʼ��
	ERR_TYPE_NODEV		= 4 ,//û�ж�Ӧindex��GPU�豸
	ERR_TYPE_DISABLE	= 5 ,//GPU�豸�Ѿ��ù⣬��������
};
//��ʼ�����ͬһ������ֻ�е�һ����Ч��ʣ�¶��Ǽ���ģʽ����ʹ��ʵ������(��֤GPU��CPU�����ʹ��)
//��ʼ�� nDevType:0: GPU 1:CPU  dConfidence:���Ŷ� [20170518] nProcessIdx:GPU Server�Č�����̖(Ĭ�J��0),�ڲ������M�̷���GPU�O��
int	EVGPUFACELOCATION_DLL_API InitializeFaceLib(double dConfidence, int nDevType = 0, int nProcessIdx = 0);

//ִ�ж�λ Image buffer(�����ROI��Ҫ��ROI����Ӿ��ε�buffer), �� ,�� �� ͼƬ���ͣ���С��������(������ڵ���1������1����)������Rect����
int EVGPUFACELOCATION_DLL_API LocationFace(unsigned char* szBuf,int nWidth,int nHeight,EImgType eImgType,int nShrink,VEC_FACERECTINFO& vecRect) ;
//�ͷ� ����Ǽ���ģʽ�ģ�ֻ�м������0 GPU�豸�Żᱻֹͣ
int EVGPUFACELOCATION_DLL_API ReleaseFaceLib();


#endif