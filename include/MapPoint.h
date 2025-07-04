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


#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include "Converter.h"

#include "SerializationUtils.h"

#include <opencv2/core/core.hpp>
#include <mutex>
#include <unordered_set>
#include <unordered_map>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>

namespace ORB_SLAM3
{

class MultiKeyFrame;
class Map;
class MultiFrame;

class GPObs
{
public:
    GPObs(double time_, int cam_, cv::KeyPoint obs_, float ur_): time(time_), cam(cam_), obs(obs_), ur(ur_) {}
    GPObs(const GPObs& other): time(other.time), cam(other.cam), obs(other.obs), ur(other.ur) {}
    GPObs(GPObs&& other): time(other.time), cam(other.cam), obs(other.obs), ur(other.ur) {}

    double time;
    int cam;
    cv::KeyPoint obs;
    float ur;

    bool operator==(const GPObs& other) const
    {
        return time == other.time && cam == other.cam && obs.pt == other.obs.pt && ur == other.ur;
    }
};

class MapPoint
{

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & mnId;
        ar & mnFirstKFid;
        ar & mnFirstFrame;
        ar & nObs;
        // Variables used by the tracking
        //ar & mTrackProjX;
        //ar & mTrackProjY;
        //ar & mTrackDepth;
        //ar & mTrackDepthR;
        //ar & mTrackProjXR;
        //ar & mTrackProjYR;
        //ar & mbTrackInView;
        //ar & mbTrackInViewR;
        //ar & mnTrackScaleLevel;
        //ar & mnTrackScaleLevelR;
        //ar & mTrackViewCos;
        //ar & mTrackViewCosR;
        //ar & mnTrackReferenceForFrame;
        //ar & mnLastFrameSeen;

        // Variables used by local mapping
        //ar & mnBALocalForKF;
        //ar & mnFuseCandidateForKF;

        // Variables used by loop closing and merging
        //ar & mnLoopPointForKF;
        //ar & mnCorrectedByKF;
        //ar & mnCorrectedReference;
        //serializeMatrix(ar,mPosGBA,version);
        //ar & mnBAGlobalForKF;
        //ar & mnBALocalForMerge;
        //serializeMatrix(ar,mPosMerge,version);
        //serializeMatrix(ar,mNormalVectorMerge,version);

        // Protected variables
        ar & boost::serialization::make_array(mWorldPos.data(), mWorldPos.size());
        ar & boost::serialization::make_array(mNormalVector.data(), mNormalVector.size());
        //ar & BOOST_SERIALIZATION_NVP(mBackupObservationsId);
        //ar & mObservations;
        ar & mBackupObservationsId;
        // ar & mBackupObservationsId2;
        serializeMatrix(ar,mDescriptor,version);
        ar & mBackupRefKFId;
        //ar & mnVisible;
        //ar & mnFound;

        ar & mbBad;
        ar & mBackupReplacedId;

        ar & mfMinDistance;
        ar & mfMaxDistance;

    }


public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MapPoint();

    MapPoint(const Eigen::Vector3f &Pos, MultiKeyFrame* pRefKF, Map* pMap);
    MapPoint(const double invDepth, cv::Point2f uv_init, MultiKeyFrame* pRefKF, MultiKeyFrame* pHostKF, Map* pMap);
    MapPoint(const Eigen::Vector3f &Pos,  Map* pMap, MultiFrame* pFrame, const int &idxF);

    void SetWorldPos(const Eigen::Vector3f &Pos);
    Eigen::Vector3f GetWorldPos();

    Eigen::Vector3f GetNormal();
    void SetNormalVector(const Eigen::Vector3f& normal);

    MultiKeyFrame* GetReferenceKeyFrame();

    std::map<MultiKeyFrame*,std::vector<int>> GetObservations();
    int Observations();

    void AddObservation(MultiKeyFrame* pKF,int idx);
    void EraseObservation(MultiKeyFrame* pKF);
    void EraseObservation(MultiKeyFrame* pKF, int c);

    void AddGPObservation(MultiKeyFrame* pKF, double time, int cam, cv::KeyPoint obs, float ur);
    void EraseGPObservation(MultiKeyFrame* pKF, const GPObs& obs);
    std::unordered_multimap<MultiKeyFrame*, GPObs> GetGPObservations();

    std::vector<int> GetIndexInKeyFrame(MultiKeyFrame* pKF);
    bool IsInKeyFrame(MultiKeyFrame* pKF);
    bool IsInKFCamera(MultiKeyFrame* pKF, int c);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint* pMP);    
    MapPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, MultiKeyFrame*pKF);
    int PredictScale(const float &currentDist, MultiFrame* pF);

    Map* GetMap();
    void UpdateMap(Map* pMap);

    void PrintObservations();

    void PreSave(set<MultiKeyFrame*>& spKF,set<MapPoint*>& spMP);
    void PostLoad(map<long unsigned int, MultiKeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid);

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;

    // Variables used by the tracking
    std::vector<float> mvTrackProjX;
    std::vector<float> mvTrackProjY;
    std::vector<float> mvTrackDepth;
    std::vector<float> mvTrackDepthR;
    std::vector<float> mvTrackProjXR;
    std::vector<float> mvTrackProjYR;
    // 判断该地图点是否在当前视野中，且需要匹配
    std::vector<bool> mvbTrackInView;
    std::vector<bool> mvbTrackInViewR;
    // 当前帧的尺度金字塔层数
    std::vector<int> mvnTrackScaleLevel;
    std::vector<int> mvnTrackScaleLevelR;
    // 当前帧的视角余弦值
    std::vector<float> mvTrackViewCos;
    std::vector<float> mvTrackViewCosR;

    long unsigned int mnUpdateLocalKF = -1;
    long unsigned int mnTrackReferenceForFrame;
    std::vector<long unsigned int> mvnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // 保存两关键帧之间该点在非关键帧的观测，跟踪线程中添加观测，用于局部建图线程中的localBA并删除外点观测，所以要做线程保护
    std::unordered_multimap<MultiKeyFrame*, GPObs> mObservationsForGPBA;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;    
    Eigen::Vector3f mPosGBA;
    long unsigned int mnBAGlobalForKF;
    long unsigned int mnBALocalForMerge;

    // Variable used by merging
    Eigen::Vector3f mPosMerge;
    Eigen::Vector3f mNormalVectorMerge;


    // Fopr inverse depth optimization
    double mInvDepth;
    double mInitU;
    double mInitV;
    MultiKeyFrame* mpHostKF;

    static std::mutex mGlobalMutex;

    unsigned int mnOriginMapId;

protected:    

     // Position in absolute coordinates
     Eigen::Vector3f mWorldPos;

     // Keyframes observing the point and associated index in keyframe
     // 每个KF对应的观测相机的全局特征点序号
     std::map<MultiKeyFrame*, std::vector<int>> mObservations;
     int nCamera;
     // For save relation without pointer, this is necessary for save/load function
    //  std::map<long unsigned int, int> mBackupObservationsId1;
    //  std::map<long unsigned int, int> mBackupObservationsId2;
    std::map<long unsigned int, std::vector<int>> mBackupObservationsId;

     // Mean viewing direction
     Eigen::Vector3f mNormalVector;

     // Best descriptor to fast matching
     cv::Mat mDescriptor;

     // Reference KeyFrame
     MultiKeyFrame* mpRefKF;
     long unsigned int mBackupRefKFId;

     // Tracking counters
     int mnVisible;
     int mnFound;

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;
     MapPoint* mpReplaced;
     // For save relation without pointer, this is necessary for save/load function
     long long int mBackupReplacedId;

     // Scale invariance distances
     float mfMinDistance;
     float mfMaxDistance;

     Map* mpMap;

     // Mutex
     std::mutex mMutexPos;
     std::mutex mMutexFeatures;
     std::mutex mMutexMap;
     std::mutex mMutexGPFeatures;

};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
