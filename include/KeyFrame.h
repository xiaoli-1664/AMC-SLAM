/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"
#include "ImuTypes.h"

#include "GeometricCamera.h"
#include "SerializationUtils.h"

#include <mutex>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>


namespace ORB_SLAM3
{

class Map;
class MapPoint;
class MultiFrame;
class KeyFrameDatabase;
class GaussianProcess;

class GeometricCamera;

class MultiKeyFrame
{
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar & mnId;
        ar & const_cast<long unsigned int&>(mnFrameId);
        ar & const_cast<double&>(mTimeStamp);
        // Grid
        ar & const_cast<int&>(mnGridCols);
        ar & const_cast<int&>(mnGridRows);
        // ar & const_cast<float&>(mvfGridElementWidthInv);
        // ar & const_cast<float&>(mvfGridElementHeightInv);

        // Variables of tracking
        //ar & mnTrackReferenceForFrame;
        //ar & mnFuseTargetForKF;
        // Variables of local mapping
        //ar & mnBALocalForKF;
        //ar & mnBAFixedForKF;
        //ar & mnNumberOfOpt;
        // Variables used by KeyFrameDatabase
        //ar & mnLoopQuery;
        //ar & mnLoopWords;
        //ar & mLoopScore;
        //ar & mnRelocQuery;
        //ar & mnRelocWords;
        //ar & mRelocScore;
        //ar & mnMergeQuery;
        //ar & mnMergeWords;
        //ar & mMergeScore;
        //ar & mnPlaceRecognitionQuery;
        //ar & mnPlaceRecognitionWords;
        //ar & mPlaceRecognitionScore;
        //ar & mbCurrentPlaceRecognition;
        // Variables of loop closing
        //serializeMatrix(ar,mTcwGBA,version);
        //serializeMatrix(ar,mTcwBefGBA,version);
        //serializeMatrix(ar,mVwbGBA,version);
        //serializeMatrix(ar,mVwbBefGBA,version);
        //ar & mBiasGBA;
        //ar & mnBAGlobalForKF;
        // Variables of Merging
        //serializeMatrix(ar,mTcwMerge,version);
        //serializeMatrix(ar,mTcwBefMerge,version);
        //serializeMatrix(ar,mTwcBefMerge,version);
        //serializeMatrix(ar,mVwbMerge,version);
        //serializeMatrix(ar,mVwbBefMerge,version);
        //ar & mBiasMerge;
        //ar & mnMergeCorrectedForKF;
        //ar & mnMergeForKF;
        //ar & mfScaleMerge;
        //ar & mnBALocalForMerge;

        // Scale
        ar & mfScale;
        // Calibration parameters
        // ar & const_cast<float&>(mvfx);
        // ar & const_cast<float&>(mvfy);
        // ar & const_cast<float&>(minvfx);
        // ar & const_cast<float&>(minvfy);
        // ar & const_cast<float&>(mvcx);
        // ar & const_cast<float&>(mvcy);
        ar & const_cast<float&>(mbf);
        ar & const_cast<float&>(mb);
        ar & const_cast<float&>(mThDepth);
        // serializeMatrix(ar, mvDistCoef, version);
        // Number of Keypoints
        ar & const_cast<int&>(N);
        // KeyPoints
        serializeVectorKeyPoints<Archive>(ar, mvKeys, version);
        serializeVectorKeyPoints<Archive>(ar, mvKeysUn, version);
        ar & const_cast<vector<float>& >(mvuRight);
        ar & const_cast<vector<float>& >(mvDepth);
        // serializeMatrix<Archive>(ar,mvDescriptors,version);
        // BOW
        ar & mBowVec;
        ar & mFeatVec;
        // Pose relative to parent
        serializeSophusSE3<Archive>(ar, mTbp, version);
        // Scale
        ar & const_cast<int&>(mnScaleLevels);
        ar & const_cast<float&>(mfScaleFactor);
        ar & const_cast<float&>(mfLogScaleFactor);
        ar & const_cast<vector<float>& >(mvScaleFactors);
        ar & const_cast<vector<float>& >(mvLevelSigma2);
        ar & const_cast<vector<float>& >(mvInvLevelSigma2);
        // Image bounds and calibration
        // ar & const_cast<int&>(mvMinX);
        // ar & const_cast<int&>(mvMinY);
        // ar & const_cast<int&>(mvMaxX);
        // ar & const_cast<int&>(mvMaxY);
        // ar & boost::serialization::make_array(mvK_.data(), mvK_.size());
        // Pose
        serializeSophusSE3<Archive>(ar, mTbw, version);
        // MapPointsId associated to keypoints
        ar & mvBackupMapPointsId;
        // Grid
        ar & mvGrid;
        // Connected KeyFrameWeight
        ar & mBackupConnectedKeyFrameIdWeights;
        // Spanning Tree and Loop Edges
        ar & mbFirstConnection;
        ar & mBackupParentId;
        ar & mvBackupChildrensId;
        ar & mvBackupLoopEdgesId;
        ar & mvBackupMergeEdgesId;
        // Bad flags
        ar & mbNotErase;
        ar & mbToBeErased;
        ar & mbBad;

        ar & mHalfBaseline;

        ar & mnOriginMapId;

        // Camera variables
        ar & mnBackupIdCamera;
        ar & mnBackupIdCamera2;

        // Fisheye variables
        ar & mvLeftToRightMatch;
        ar & mvRightToLeftMatch;
        ar & const_cast<int&>(NLeft);
        ar & const_cast<int&>(NRight);
        serializeSophusSE3<Archive>(ar, mTlr, version);
        // serializeVectorKeyPoints<Archive>(ar, mvKeysRight, version);
        // ar & mGridRight;

        // Inertial variables
        ar & mImuBias;
        ar & mBackupImuPreintegrated;
        ar & mImuCalib;
        ar & mBackupPrevKFId;
        ar & mBackupNextKFId;
        ar & bImu;
        ar & boost::serialization::make_array(mVw.data(), mVw.size());
        ar & boost::serialization::make_array(mOwb.data(), mOwb.size());
        ar & mbHasVelocity;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MultiKeyFrame();
    MultiKeyFrame(MultiFrame &F, Map* pMap, KeyFrameDatabase* pKFDB);

    // Pose functions
    void SetPose(const Sophus::SE3f &Tbw);
    void SetPose(const Sophus::SE3f &Tbw, const std::vector<Sophus::SE3f> &Twc);

    void SetVelocity(const Eigen::VectorXf &Vw_);

    Sophus::SE3f GetPose();
    // TODO: 完成定义, 插值计算Tcw
    Sophus::SE3f GetCameraPose(int cam);
    Sophus::SE3f GetCameraPoseInverse(int cam);

    Sophus::SE3f GetPoseInverse();
    Eigen::Vector3f GetFrameCenter();
    std::vector<Sophus::SE3f> GetCameraPoses();
    Eigen::Vector3f GetCameraCenter(int cam);

    Eigen::Vector3f GetImuPosition();
    Eigen::Matrix3f GetImuRotation();
    Sophus::SE3f GetImuPose();
    Eigen::Matrix3f GetRotation();
    Eigen::Vector3f GetTranslation();
    Eigen::VectorXf GetVelocity();
    bool isVelocitySet();

    // Bag of Words Representation
    void ComputeBoW();

    // Covisibility graph functions
    void AddConnection(MultiKeyFrame* pKF, const int &weight);
    void EraseConnection(MultiKeyFrame* pKF);

    void UpdateConnections(bool upParent=true);
    void UpdateBestCovisibles();
    std::set<MultiKeyFrame *> GetConnectedKeyFrames();
    std::vector<MultiKeyFrame* > GetVectorCovisibleKeyFrames();
    std::vector<MultiKeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<MultiKeyFrame*> GetCovisiblesByWeight(const int &w);
    int GetWeight(MultiKeyFrame* pKF);

    // Spanning tree functions
    void AddChild(MultiKeyFrame* pKF);
    void EraseChild(MultiKeyFrame* pKF);
    void ChangeParent(MultiKeyFrame* pKF);
    std::set<MultiKeyFrame*> GetChilds();
    MultiKeyFrame* GetParent();
    bool hasChild(MultiKeyFrame* pKF);
    void SetFirstConnection(bool bFirst);

    // Loop Edges
    void AddLoopEdge(MultiKeyFrame* pKF);
    std::set<MultiKeyFrame*> GetLoopEdges();;

    // Merge Edges
    void AddMergeEdge(MultiKeyFrame* pKF);
    set<MultiKeyFrame*> GetMergeEdges();

    // MapPoint observation functions
    int GetNumberMPs();
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    void EraseMapPointMatch(const int &idx);
    void EraseMapPointMatch(MapPoint* pMP);
    void EraseMapPointMatch(MapPoint* pMp, int cam);
    void ReplaceMapPointMatch(const int &idx, MapPoint* pMP);
    std::set<MapPoint*> GetMapPoints();
    std::vector<MapPoint*> GetMapPointMatches();
    int TrackedMapPoints(const int &minObs);
    MapPoint* GetMapPoint(const size_t &idx);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(int cam, const float &x, const float  &y, const float  &r, const bool bRight = false) const;
    bool UnprojectStereo(int i, Eigen::Vector3f &x3D);

    // Image
    bool IsInImage(int cam, const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag
    void SetBadFlag();
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(const int q);

    static bool weightComp( int a, int b){
        return a>b;
    };

    static bool lId(MultiKeyFrame* pKF1, MultiKeyFrame* pKF2){
        return pKF1->mnId<pKF2->mnId;
    }

    Map* GetMap();
    void UpdateMap(Map* pMap);

    void SetNewBias(const IMU::Bias &b);
    Eigen::Vector3f GetGyroBias();

    Eigen::Vector3f GetAccBias();

    IMU::Bias GetImuBias();

    bool ProjectPointDistort(int cam, MapPoint* pMP, cv::Point2f &kp, float &u, float &v);
    bool ProjectPointUnDistort(int cam, MapPoint* pMP, cv::Point2f &kp, float &u, float &v);

    void PreSave(set<MultiKeyFrame*>& spKF,set<MapPoint*>& spMP, set<GeometricCamera*>& spCam);
    void PostLoad(map<long unsigned int, MultiKeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid, map<unsigned int, GeometricCamera*>& mpCamId);


    void SetORBVocabulary(ORBVocabulary* pORBVoc);
    void SetKeyFrameDatabase(KeyFrameDatabase* pKFDB);

    bool bImu;

    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:
    int nCamera;

    static long unsigned int nNextId;
    long unsigned int mnId;
    const long unsigned int mnFrameId;

    const double mTimeStamp;
    // 关键帧中各相机对应的时间戳
    std::vector<double> mvTimeStamps;

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;

    std::vector<float> mvfGridElementWidthInv;
    std::vector<float> mvfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    //Number of optimizations by BA(amount of iterations in BA)
    long unsigned int mnNumberOfOpt;

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;
    long unsigned int mnMergeQuery;
    int mnMergeWords;
    float mMergeScore;
    long unsigned int mnPlaceRecognitionQuery;
    int mnPlaceRecognitionWords;
    float mPlaceRecognitionScore;

    bool mbCurrentPlaceRecognition;

    
    // Variables used by loop closing
    Sophus::SE3f mTbwGBA;
    Sophus::SE3f mTbwBefGBA;
    Eigen::VectorXf mVwbGBA = Eigen::VectorXf::Zero(6);
    Eigen::VectorXf mVwbBefGBA = Eigen::VectorXf::Zero(6);
    IMU::Bias mBiasGBA;
    long unsigned int mnBAGlobalForKF;

    // Variables used by merging
    Sophus::SE3f mTBwMerge;
    Sophus::SE3f mTBwBefMerge;
    Sophus::SE3f mTwBBefMerge;
    Eigen::VectorXf mVwbMerge = Eigen::VectorXf::Zero(6);
    Eigen::VectorXf mVwbBefMerge = Eigen::VectorXf::Zero(6);
    IMU::Bias mBiasMerge;
    long unsigned int mnMergeCorrectedForKF;
    long unsigned int mnMergeForKF;
    float mfScaleMerge;
    long unsigned int mnBALocalForMerge;

    float mfScale;

    // Calibration parameters
    std::vector<float> mvfx;
    std::vector<float> mvfy;
    std::vector<float> mvcx;
    std::vector<float> mvcy;
    std::vector<float> minvfx;
    std::vector<float> minvfy;

    const float mbf, mb, mThDepth;
    std::vector<cv::Mat> mvDistCoef;

    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;

    // 特征点对应的相机id,特征点全局id对应的局部id
    std::unordered_map<size_t, int> mmpKeyToCam;
    std::unordered_map<size_t, int> mmpGlobalToLocalID;

    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth; // negative value for monocular points
    std::vector<cv::Mat> mvDescriptors;

    //BoW
    DBoW2::BowVector mBowVec;
    std::vector<DBoW2::BowVector> mBowVecs;
    DBoW2::FeatureVector mFeatVec;
    std::vector<DBoW2::FeatureVector> mFeatVecs;

    // Pose relative to parent (this is computed when bad flag is activated)
    Sophus::SE3f mTbp;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2;

    // Image bounds and calibration
    std::vector<float> mvMinY;
    std::vector<float> mvMinX;
    std::vector<float> mvMaxX;
    std::vector<float> mvMaxY;

    // Preintegrated IMU measurements from previous keyframe
    MultiKeyFrame* mPrevKF;
    MultiKeyFrame* mNextKF;

    IMU::Preintegrated* mpImuPreintegrated;
    IMU::Calib mImuCalib;

    unsigned int mnOriginMapId;

    string mNameFile;

    int mnDataset;

    std::vector <MultiKeyFrame*> mvpLoopCandKFs;
    std::vector <MultiKeyFrame*> mvpMergeCandKFs;

    GaussianProcess* mpGP;
    //bool mbHasHessian;
    //cv::Mat mHessianPose;

    static std::vector<Sophus::SE3f> mTbc;
    // The following variables need to be accessed trough a mutex to be thread safe.
protected:
    // 每个相机到机器人base的转换

    // sophus poses
    Sophus::SE3<float> mTbw;
    Eigen::Matrix3f mRbw;
    Sophus::SE3<float> mTwb;
    Eigen::Matrix3f mRwb;
    std::vector<Sophus::SE3f> mTwc;
    std::vector<Sophus::SE3f> mTcw;

    // IMU position
    Eigen::Vector3f mOwb;
    // 机器人的速度旋量，在世界坐标系下表示，连续时间slam中可估算
    Eigen::VectorXf mVw = Eigen::VectorXf::Zero(6);
    bool mbHasVelocity;

    //Transformation matrix between cameras in stereo fisheye
    Sophus::SE3<float> mTlr;
    Sophus::SE3<float> mTrl;

    // Imu bias
    IMU::Bias mImuBias;

    // MapPoints associated to keypoints
    std::vector<MapPoint*> mvpMapPoints;
    // For save relation without pointer, this is necessary for save/load function
    std::vector<long long int> mvBackupMapPointsId;

    // BoW
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBvocabulary;

    // Grid over the image to speed up feature matching for each camera
    std::vector<std::vector< std::vector <std::vector<size_t> > > > mvGrid;

    std::map<MultiKeyFrame*,int> mConnectedKeyFrameWeights;
    std::vector<MultiKeyFrame*> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;
    // For save relation without pointer, this is necessary for save/load function
    std::map<long unsigned int, int> mBackupConnectedKeyFrameIdWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    MultiKeyFrame* mpParent;
    std::set<MultiKeyFrame*> mspChildrens;
    std::set<MultiKeyFrame*> mspLoopEdges;
    std::set<MultiKeyFrame*> mspMergeEdges;
    // For save relation without pointer, this is necessary for save/load function
    long long int mBackupParentId;
    std::vector<long unsigned int> mvBackupChildrensId;
    std::vector<long unsigned int> mvBackupLoopEdgesId;
    std::vector<long unsigned int> mvBackupMergeEdgesId;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;    

    float mHalfBaseline; // Only for visualization

    Map* mpMap;

    // Backup variables for inertial
    long long int mBackupPrevKFId;
    long long int mBackupNextKFId;
    IMU::Preintegrated mBackupImuPreintegrated;

    // Backup for Cameras
    unsigned int mnBackupIdCamera, mnBackupIdCamera2;

    // Calibration
    std::vector<Eigen::Matrix3f> mvK_;

    // Mutex
    std::mutex mMutexPose; // for pose, velocity and biases
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
    std::mutex mMutexMap;

public:
    std::vector<GeometricCamera*> mvpCamera;

    //Indexes of stereo observations correspondences
    std::vector<int> mvLeftToRightMatch, mvRightToLeftMatch;

    Sophus::SE3f GetRelativePoseTrl();
    Sophus::SE3f GetRelativePoseTlr();

    //KeyPoints in the right image (for stereo fisheye, coordinates are needed)
    // const std::vector<cv::KeyPoint> mvKeysRight;

    const int NLeft, NRight;

    // std::vector< std::vector <std::vector<size_t> > > mGridRight;

    Sophus::SE3<float> GetRightPose();
    Sophus::SE3<float> GetRightPoseInverse();

    Eigen::Vector3f GetRightCameraCenter();
    Eigen::Matrix<float,3,3> GetRightRotation();
    Eigen::Vector3f GetRightTranslation();

    void PrintPointDistribution(){
        int left = 0, right = 0;
        int Nlim = (NLeft != -1) ? NLeft : N;
        for(int i = 0; i < N; i++){
            if(mvpMapPoints[i]){
                if(i < Nlim) left++;
                else right++;
            }
        }
        cout << "Point distribution in KeyFrame: left-> " << left << " --- right-> " << right << endl;
    }


};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
