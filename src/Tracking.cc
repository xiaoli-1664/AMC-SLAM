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


#include "Tracking.h"

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "G2oTypes.h"
#include "Optimizer.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"
#include "MLPnPsolver.h"
#include "GeometricTools.h"

#include <iostream>

#include <mutex>
#include <chrono>
#include <opencv2/core/persistence.hpp>


using namespace std;

namespace ORB_SLAM3
{


Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq):
    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
    mbReadyToInitializate(false), mpSystem(pSys), mpViewer(NULL), bStepByStep(false),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpLastKeyFrame(static_cast<MultiKeyFrame*>(NULL))
{
    std::cout << "Tracking constructor" << std::endl;
    // Load camera parameters from settings file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    cv::FileNode node = fSettings["Ransac.threshold"];
    if (node.empty())
    {
        mRansacThreshold = 2.0;
    }
    else
    {
        mRansacThreshold = node.real();
    }
    std::cout << "Ransac threshold: " << mRansacThreshold << std::endl;
    if(settings){
        ParseGaussianParamFile(fSettings);
        newParameterLoader(settings);
    }
    else{

        std::cout << "No settings file provided, using default values" << std::endl;
        ParseGaussianParamFile(fSettings);

        bool b_parse_cam = ParseCamParamFile(fSettings);
        if(!b_parse_cam)
        {
            std::cout << "*Error with the camera parameters in the config file*" << std::endl;
        }

        // Load ORB parameters
        bool b_parse_orb = ParseORBParamFile(fSettings);
        if(!b_parse_orb)
        {
            std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
        }

        bool b_parse_imu = true;

        if(!b_parse_cam || !b_parse_orb || !b_parse_imu)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

    initID = 0; lastID = 0;
    mbInitWith3KFs = false;
    mnNumDataset = 0;

    vector<GeometricCamera*> vpCams = mpAtlas->GetAllCameras();
    std::cout << "There are " << vpCams.size() << " cameras in the atlas" << std::endl;
    for(GeometricCamera* pCam : vpCams)
    {
        std::cout << "Camera " << pCam->GetId();
        std::cout << "fx: " << pCam->getParameter(0) << " fy: " << pCam->getParameter(1) << " cx: " << pCam->getParameter(2) << " cy: " << pCam->getParameter(3) << std::endl;
        if(pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            std::cout << " is pinhole" << std::endl;
        }
        else if(pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            std::cout << " is fisheye" << std::endl;
        }
        else
        {
            std::cout << " is unknown" << std::endl;
        }
    }

#ifdef REGISTER_TIMES
    vdRectStereo_ms.clear();
    vdResizeImage_ms.clear();
    vdORBExtract_ms.clear();
    vdStereoMatch_ms.clear();
    vdIMUInteg_ms.clear();
    vdPosePred_ms.clear();
    vdLMTrack_ms.clear();
    vdNewKF_ms.clear();
    vdTrackTotal_ms.clear();
#endif
}

#ifdef REGISTER_TIMES
double calcAverage(vector<double> v_times)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += value;
    }

    return accum / v_times.size();
}

double calcDeviation(vector<double> v_times, double average)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

double calcAverage(vector<int> v_values)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += value;
        total++;
    }

    return accum / total;
}

double calcDeviation(vector<int> v_values, double average)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += pow(value - average, 2);
        total++;
    }
    return sqrt(accum / total);
}

void Tracking::LocalMapStats2File()
{
    ofstream f;
    f.open("LocalMapTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
    for(int i=0; i<mpLocalMapper->vdLMTotal_ms.size(); ++i)
    {
        f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << ","
          << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBASync_ms[i] << ","
          << mpLocalMapper->vdKFCullingSync_ms[i] <<  "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
    }

    f.close();

    f.open("LBA_Stats.txt");
    f << fixed << setprecision(6);
    f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
    for(int i=0; i<mpLocalMapper->vdLBASync_ms.size(); ++i)
    {
        f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << ","
          << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
          << mpLocalMapper->vnLBA_edges[i] << endl;
    }


    f.close();
}

void Tracking::TrackStats2File()
{
    ofstream f;
    f.open("SessionInfo.txt");
    f << fixed;
    f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
    f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

    f << "OpenCV version: " << CV_VERSION << endl;

    f.close();

    f.open("TrackingTimeStats.txt");
    f << fixed << setprecision(6);

    f << "#Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]" << endl;

    for(int i=0; i<vdTrackTotal_ms.size(); ++i)
    {
        double stereo_rect = 0.0;
        if(!vdRectStereo_ms.empty())
        {
            stereo_rect = vdRectStereo_ms[i];
        }

        double resize_image = 0.0;
        if(!vdResizeImage_ms.empty())
        {
            resize_image = vdResizeImage_ms[i];
        }

        double stereo_match = 0.0;
        if(!vdStereoMatch_ms.empty())
        {
            stereo_match = vdStereoMatch_ms[i];
        }

        double imu_preint = 0.0;
        if(!vdIMUInteg_ms.empty())
        {
            imu_preint = vdIMUInteg_ms[i];
        }

        f << stereo_rect << "," << resize_image << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << ","
          << vdPosePred_ms[i] <<  "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i] << "," << vdTrackTotal_ms[i] << endl;
    }

    f.close();
}

void Tracking::PrintTimeStats()
{
    // Save data in files
    TrackStats2File();
    LocalMapStats2File();


    ofstream f;
    f.open("ExecMean.txt");
    f << fixed;
    //Report the mean and std of each one
    std::cout << std::endl << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    cout << "OpenCV version: " << CV_VERSION << endl;
    f << "OpenCV version: " << CV_VERSION << endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    f << "---------------------------" << std::endl;
    f << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    double average, deviation;
    if(!vdRectStereo_ms.empty())
    {
        average = calcAverage(vdRectStereo_ms);
        deviation = calcDeviation(vdRectStereo_ms, average);
        std::cout << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdResizeImage_ms.empty())
    {
        average = calcAverage(vdResizeImage_ms);
        deviation = calcDeviation(vdResizeImage_ms, average);
        std::cout << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
        f << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdORBExtract_ms);
    deviation = calcDeviation(vdORBExtract_ms, average);
    std::cout << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;
    f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

    if(!vdStereoMatch_ms.empty())
    {
        average = calcAverage(vdStereoMatch_ms);
        deviation = calcDeviation(vdStereoMatch_ms, average);
        std::cout << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdIMUInteg_ms.empty())
    {
        average = calcAverage(vdIMUInteg_ms);
        deviation = calcDeviation(vdIMUInteg_ms, average);
        std::cout << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
        f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdPosePred_ms);
    deviation = calcDeviation(vdPosePred_ms, average);
    std::cout << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;
    f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdLMTrack_ms);
    deviation = calcDeviation(vdLMTrack_ms, average);
    std::cout << "LM Track: " << average << "$\\pm$" << deviation << std::endl;
    f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdNewKF_ms);
    deviation = calcDeviation(vdNewKF_ms, average);
    std::cout << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;
    f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdTrackTotal_ms);
    deviation = calcDeviation(vdTrackTotal_ms, average);
    std::cout << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping time stats
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Local Mapping" << std::endl << std::endl;
    f << std::endl << "Local Mapping" << std::endl << std::endl;

    average = calcAverage(mpLocalMapper->vdKFInsert_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
    std::cout << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
    std::cout << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCreation_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
    std::cout << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLBA_ms);
    deviation = calcDeviation(mpLocalMapper->vdLBA_ms, average);
    std::cout << "LBA: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdKFCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFCulling_ms, average);
    std::cout << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLMTotal_ms);
    deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
    std::cout << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping LBA complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_edges);
    deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
    std::cout << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFopt);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
    std::cout << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
    std::cout << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_MPs);
    deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
    std::cout << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    std::cout << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    std::cout << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;
    f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

    // Map complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Map complexity" << std::endl;
    std::cout << "KFs in map: " << mpAtlas->GetAllKeyFrames().size() << std::endl;
    std::cout << "MPs in map: " << mpAtlas->GetAllMapPoints().size() << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "Map complexity" << std::endl;
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBestMap = vpMaps[0];
    for(int i=1; i<vpMaps.size(); ++i)
    {
        if(pBestMap->GetAllKeyFrames().size() < vpMaps[i]->GetAllKeyFrames().size())
        {
            pBestMap = vpMaps[i];
        }
    }

    f << "KFs in map: " << pBestMap->GetAllKeyFrames().size() << std::endl;
    f << "MPs in map: " << pBestMap->GetAllMapPoints().size() << std::endl;

    f << "---------------------------" << std::endl;
    f << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdDataQuery_ms);
    deviation = calcDeviation(mpLoopClosing->vdDataQuery_ms, average);
    f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdEstSim3_ms);
    deviation = calcDeviation(mpLoopClosing->vdEstSim3_ms, average);
    f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdPRTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdPRTotal_ms, average);
    f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopFusion_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopFusion_ms, average);
    f << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopOptEss_ms, average);
    f << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopTotal_ms, average);
    f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nLoop << std::endl;
    average = calcAverage(mpLoopClosing->vnLoopKFs);
    deviation = calcDeviation(mpLoopClosing->vnLoopKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeMaps_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeMaps_ms, average);
    f << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdWeldingBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdWeldingBA_ms, average);
    f << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeOptEss_ms, average);
    f << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeTotal_ms, average);
    f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nMerges << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nMerges << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeKFs);
    deviation = calcDeviation(mpLoopClosing->vnMergeKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeMPs);
    deviation = calcDeviation(mpLoopClosing->vnMergeMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdGBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdGBA_ms, average);
    f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdUpdateMap_ms);
    deviation = calcDeviation(mpLoopClosing->vdUpdateMap_ms, average);
    f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdFGBATotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdFGBATotal_ms, average);
    f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    f << "Numb abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    std::cout << "Num abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAKFs);
    deviation = calcDeviation(mpLoopClosing->vnGBAKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAMPs);
    deviation = calcDeviation(mpLoopClosing->vnGBAMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f.close();

}

#endif

Tracking::~Tracking()
{
    //f_track_stats.close();

}

void Tracking::newParameterLoader(Settings* settings)
{
    mvpCameras = settings->camera();
    mvpCameras.pop_back();


    mnCamera = mvpCameras.size();

    mvDistCoef = std::vector<cv::Mat>(mvpCameras.size(), cv::Mat::zeros(4,1,CV_32F));

    mImageScale = 1.0f;

    mvK.resize(mvpCameras.size());
    mvK_.resize(mvpCameras.size());
    mTbc.resize(mvpCameras.size());
    for (int i = 0; i < mvpCameras.size(); ++i)
    {
        cv::Mat K = cv::Mat::eye(3,3,CV_32F);
        Eigen::Matrix3f K_ = Eigen::Matrix3f::Identity();
        K.at<float>(0,0) = mvpCameras[i]->getParameter(0);
        K.at<float>(1,1) = mvpCameras[i]->getParameter(1);
        K.at<float>(0,2) = mvpCameras[i]->getParameter(2);
        K.at<float>(1,2) = mvpCameras[i]->getParameter(3);
        K_(0,0) = mvpCameras[i]->getParameter(0);
        K_(1,1) = mvpCameras[i]->getParameter(1);
        K_(0,2) = mvpCameras[i]->getParameter(2);
        K_(1,2) = mvpCameras[i]->getParameter(3);

        mvK[i] = K;
        mvK_[i] = K_;

        mTbc[i] = settings->Tbc()[i];

        mpAtlas->AddCamera(mvpCameras[i]);
    }

    mbf = settings->bf();
    mThDepth = settings->b() * settings->thDepth();

    mMinFrames = 0;
    mMaxFrames = settings->fps();
    mbRGB = settings->rgb();

    // ORB parameters
    int nFeatures = settings->nFeatures();
    int nLevels = settings->nLevels();
    int fIniThFAST = settings->initThFAST();
    int fMinThFAST = settings->minThFAST();
    float scaleFactor = settings->scaleFactor();

    mnFeatures = nFeatures;
    mvnFeaturesCams.resize(mnCamera+1, 1/(mnCamera+1));
    for (int i = 0; i <= mnCamera; ++i)
    {
        mvnFeaturesCams[i] = mnFeatures / (mnCamera + 1);
        std::cout << "Camera " << i << " has " << mvnFeaturesCams[i] << " features" << std::endl;
        mvpORBextractors.push_back(new ORBextractor(mvnFeaturesCams[i], scaleFactor, nLevels, fIniThFAST, fMinThFAST));
    }

}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
    mnCamera = fSettings["Camera.number"];
    mvDistCoef.resize(mnCamera, cv::Mat::zeros(4,1,CV_32F));
    std::cout << endl << "Camera Parameters: " << endl;
    bool b_miss_params = false;

    string dataset = fSettings["dataset"];

    cv::FileNode node = fSettings["Camera.calibfile"];
    if (node.empty()) return false;
    vector<string> vCamCalibFile;
    node >> vCamCalibFile;
    for (int i = 0; i < mnCamera; ++i)
    {
        vCamCalibFile[i] = dataset + vCamCalibFile[i];
        std::cout << "- Camera " << i << " calibration file: " << vCamCalibFile[i] << endl;
    }
    for (int i = 0; i < mnCamera; ++i)
        if (!ParseEachCamParamFile(vCamCalibFile[i]))
        {
            std::cout << "*Error with the camera calib parameters in the config file*" << std::endl;
            return false;
        }

    node = fSettings["Camera.bf"];
    if(!node.empty() && node.isReal())
    {
        mbf = node.real();
    }
    else
    {
        std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    std::cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        std::cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        std::cout << "- color order: BGR (ignored if grayscale)" << endl;

    float fx = mvpCameras.back()->getParameter(0);
    node = fSettings["ThDepth"];
    if(!node.empty()  && node.isReal())
    {
        mThDepth = node.real();
        mThDepth = mbf*mThDepth/fx;
        std::cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }
    else
    {
        std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    return true;
}

bool Tracking::ParseEachCamParamFile(string strSettingPath)
{
    float fx, fy, cx, cy;
    std::ifstream f(strSettingPath);
    if (!f.is_open()) 
    {
        std::cerr << "Cannot open the camera parameter file" << std::endl;
        return false;
    }

    nlohmann::json data = nlohmann::json::parse(f);

    // Camera extrinsic parameters
    std::vector<float> vectorTbc = data["sensor_to_vehicle"];

    Eigen::Matrix4f mat;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            mat(i,j) = vectorTbc[i*4+j];
        }
    }

    Sophus::SE3f Tbc(mat);
    mTbc.push_back(Tbc);

    // Camera intrinsic parameters
    std::vector<float> vectorK = data["intrinsics"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    Eigen::Matrix3f K_ = Eigen::Matrix3f::Identity();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            K.at<float>(i,j) = vectorK[i*3+j];
            K_(i,j) = vectorK[i*3+j];
        }
    }
    mvK.push_back(K);
    mvK_.push_back(K_);

    fx = K.at<float>(0,0);
    fy = K.at<float>(1,1);
    cx = K.at<float>(0,2);
    cy = K.at<float>(1,2);
    vector<float> vCamCalib{fx,fy,cx,cy};
    GeometricCamera* mpCamera = new Pinhole(vCamCalib);
    mpCamera = mpAtlas->AddCamera(mpCamera);
    mvpCameras.push_back(mpCamera);   
    return true;
}

bool Tracking::ParseGaussianParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::FileNode node = fSettings["Gaussian.Qc"];
    vector<double> vQc;
    if(!node.empty() && node.isSeq())
    {
        for(cv::FileNodeIterator it = node.begin(); it != node.end(); ++it)
        {
            vQc.push_back((*it).real());
        }
    }
    else
    {
        std::cerr << "*Gaussian.Qc parameter doesn't exist or is not a sequence*" << std::endl;
        b_miss_params = true;
    }
    Eigen::Matrix<double,6,6> Qc = Eigen::Matrix<double,6,6>::Zero();
    for(int i=0; i<6; ++i)
    {
        Qc(i,i) = vQc[i];
    }
    mpgp = new GaussianProcess(Qc);
    std::cout << "Gaussian Process Parameters: Qc: " << endl << Qc << endl;
    
    node = fSettings["Velocity"];
    int i = 0;
    if (!node.empty() && node.isSeq())
    {
        for (cv::FileNodeIterator it = node.begin(); it != node.end(); ++it)
        {
            MultiFrame::iniVel(i++) = (*it).real();
        }
    }
    else
    {
        std::cerr << "*Velocity parameter doesn't exist or is not a sequence*" << std::endl;
        b_miss_params = true;
    }

    if (b_miss_params)
    {
        return false;
    }
    return true;
}

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    int nLevels, fIniThFAST, fMinThFAST;
    float fScaleFactor;

    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if(!node.empty() && node.isInt())
    {
        mnFeatures = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.scaleFactor"];
    if(!node.empty() && node.isReal())
    {
        fScaleFactor = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.nLevels"];
    if(!node.empty() && node.isInt())
    {
        nLevels = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.iniThFAST"];
    if(!node.empty() && node.isInt())
    {
        fIniThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.minThFAST"];
    if(!node.empty() && node.isInt())
    {
        fMinThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    mvnFeaturesCams.resize(mnCamera+1, 1/(mnCamera+1));
    for (int i = 0; i <= mnCamera; ++i)
    {
        mvnFeaturesCams[i] = mnFeatures / (mnCamera+1);
        // if (i >= mnCamera-1) num_features = nFeatures / 2;
        mvpORBextractors.push_back(new ORBextractor(mvnFeaturesCams[i],fScaleFactor,nLevels,fIniThFAST,fMinThFAST));
    }

    std::cout << endl << "ORB Extractor Parameters: " << endl;
    std::cout << "- Number of Features: " << mnFeatures << endl;
    std::cout << "- Scale Levels: " << nLevels << endl;
    std::cout << "- Scale Factor: " << fScaleFactor << endl;
    std::cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    std::cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::Mat cvTbc;
    cv::FileNode node = fSettings["Tbc"];
    if(!node.empty())
    {
        cvTbc = node.mat();
        if(cvTbc.rows != 4 || cvTbc.cols != 4)
        {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    }
    else
    {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }
    std::cout << endl;
    std::cout << "Left camera to Imu Transform (Tbc): " << endl << cvTbc << endl;
    Eigen::Matrix<float,4,4,Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
    Sophus::SE3f Tbc(eigTbc);

    node = fSettings["InsertKFsWhenLost"];
    mInsertKFsLost = true;
    if(!node.empty() && node.isInt())
    {
        mInsertKFsLost = (bool) node.operator int();
    }

    if(!mInsertKFsLost)
        std::cout << "Do not insert keyframes when lost visual tracking " << endl;



    float Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if(!node.empty() && node.isInt())
    {
        mImuFreq = node.operator int();
        mImuPer = 0.001; //1.0 / (double) mImuFreq;
    }
    else
    {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if(!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if(!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if(!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if(!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.fastInit"];
    mFastInit = false;
    if(!node.empty())
    {
        mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
    }

    if(mFastInit)
        std::cout << "Fast IMU initialization. Acceleration is not checked \n";

    if(b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(mImuFreq);
    std::cout << endl;
    std::cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
    std::cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    std::cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    std::cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    std::cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);


    return true;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

bool Tracking::GetStepByStep()
{
    return bStepByStep;
}

Sophus::SE3f Tracking::GrabImageMultiCam(const std::vector<cv::Mat> &vImg, const std::vector<double> &vTimeStamps, string filename) 
{
    mvImGray = vImg;

    for (int c = 0; c < mvImGray.size(); ++c) {
        if (mvImGray[c].channels() == 3) {
            if (mbRGB)
                cv::cvtColor(mvImGray[c], mvImGray[c], cv::COLOR_RGB2GRAY); 
            else
                cv::cvtColor(mvImGray[c], mvImGray[c], cv::COLOR_BGR2GRAY);
        }
        else if (mvImGray[c].channels() == 4) {
            if (mbRGB)
                cv::cvtColor(mvImGray[c], mvImGray[c], cv::COLOR_RGBA2GRAY);
            else
                cv::cvtColor(mvImGray[c], mvImGray[c], cv::COLOR_BGRA2GRAY);
        }
    }

    // int num_feats = 0;
    // for (int i = 0; i <= mnCamera; ++i)
    // {
    //     mvpORBextractors[i]->SetNum(mvnFeaturesCams[i]);
    //     num_feats += mvnFeaturesCams[i];
    //     std::cout << "Number of features in camera " << i << ": " << mvnFeaturesCams[i] << std::endl;
    // }

    // std::cout << "num of tracks:" << mnTrackFeatures << std::endl;

    mCurrentFrame =  MultiFrame(mvImGray, vTimeStamps, mTbc, mvpORBextractors, mpgp, mpORBVocabulary, mvpCameras, mvK, mvDistCoef, mbf, mThDepth, &mLastFrame);
    // std::cout << "before track currentframe's velocity: " << mCurrentFrame.GetVelocity().transpose() << std::endl;

    // std::cout << "Frame: " << mCurrentFrame.mnId << " - TimeStamp: " << mCurrentFrame.mTimeStamp << std::endl;
    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch);
#endif

    Track();

    // std::cout << "after track currentframe's velocity: " << mCurrentFrame.GetVelocity().transpose() << std::endl;

    return mCurrentFrame.GetPose();
}

void Tracking::Track() {
    if (bStepByStep) {
        std::cout << "Tracking: Waitting to the next step" << std::endl;
        while (!mbStep && bStepByStep) 
            usleep(500);
        mbStep = false;
    }

    Map* pCurrentMap = mpAtlas->GetCurrentMap();
    if (!pCurrentMap){
        std::cout << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
        return;
    }

    if (mState != NO_IMAGES_YET) {
        if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
        {
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
    }
    else 
        mState = NOT_INITIALIZED;

    mLastProcessedState = mState;
    mbCreatedMap = false;

    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    mbMapUpdated = false;

    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();
    if (nCurMapChangeIndex > nMapChangeIndex) {
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true;
    }

    if (mState == NOT_INITIALIZED) {
        StereoInitialization();

        if (mState != OK) {
            mLastFrame = MultiFrame(mCurrentFrame);
            return;
        }

        if (mpAtlas->GetAllMaps().size() == 1) 
            mnFirstFrameId = mCurrentFrame.mnId;
    }
    else {
        mState = OK;
        bool bOK;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
#endif

        if (!mbOnlyTracking) {

            if (mState == OK) {
                CheckReplacedInLastFrame();

                 /*std::cout << "Tracking with motion model" << std::endl;*/
                bOK = TrackWithMotionModel();
                if (!bOK)
                    bOK = TrackReferenceKeyFrame();
                // std::cout << "Tracking with motion model done" << std::endl;

                if (!bOK) {
                    std::cout << "Tracking motion model fail" << std::endl;
                    if (pCurrentMap->KeyFramesInMap()>10)
                    {
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                    }
                    else
                        {
                            // mState = LOST;
                        }
                }
            }

            else
            {
                //TODO: 处理异常情况
                std::cout << "ERROR: Tracking state not ok" << std::endl;
            }
        }
        else
        {
            // TODO:纯定位模式
        }

        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

        double timePosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPosePred - time_StartPosePred).count();
        vdPosePred_ms.push_back(timePosePred);
#endif

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
#endif

        if (!mbOnlyTracking)
        {
            // std::cout << "TrackLocalMap frame vel: " << mCurrentFrame.GetVelocity().transpose() << endl;
            if (bOK)
            {
                // std::cout << "TrackLocalMap" << std::endl;
                bOK = TrackLocalMap();
                // std::cout << "TrackLocalMap done" << std::endl;
            }
                // std::cout << "frame vel: " << mCurrentFrame.GetVelocity().transpose() << endl;

            if (!bOK)
            {
                // std::cout << "ERROR: TrackLocalMap failed" << endl;
                std::cout << "frame id: " << mCurrentFrame.mnId << endl;
                std::cout << "frame vel: " << mCurrentFrame.GetVelocity().transpose() << endl;
                // terminate();
                // mCurrentFrame.SetPose(Sophus::SE3f::exp(-(mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp) * mLastFrame.GetVelocity()) * mLastFrame.GetPose());
                // mCurrentFrame.SetVelocity(mLastFrame.GetVelocity());
            }
        }
        else
        {
            // TODO:纯定位模式
        }

        if (bOK)
            mState = OK;
        else if (mState == OK)
        {
            // TODO: 处理异常情况
        }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

        double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLMTrack - time_StartLMTrack).count();
        vdLMTrack_ms.push_back(timeLMTrack);
#endif

        // std::cout << "drawer update" << std::endl;
        mpFrameDrawer->Update(this);
        if (mCurrentFrame.isSet())
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        if (bOK || mState == RECENTLY_LOST)
        {
            if (mLastFrame.isSet() && mCurrentFrame.isSet())
            {
                mVelocity = (mLastFrame.GetPose() * mCurrentFrame.GetPoseW()).log();
                mbVelocity = true;
            }
            else
            {
                mbVelocity = false;
            }
            // std::cout << "delete outliers" << std::endl;
            // std::vector<int> vnTrackFeaturesCam(mCurrentFrame.nCamera, 0);
            // std::vector<float> vfFeaturesResponse(mCurrentFrame.nCamera, 0);
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
                // int cam = mCurrentFrame.mmpKeyToCam[i];
                // pMP = mCurrentFrame.mvpMapPoints[i];
                // if (pMP)
                    // vnTrackFeaturesCam[cam]++;
                // vfFeaturesResponse[cam] += mCurrentFrame.mvKeys[i].response;
            }

            // float total_response = 0;
            // for (int i = 0; i < mCurrentFrame.nCamera; ++i)
            // {
            //     if (i < mCurrentFrame.nCamera-1)
            //         total_response += vfFeaturesResponse[i];
            //     else
            //         total_response += 2 * vfFeaturesResponse[i];
            // }

            // for (int i = 0; i < mCurrentFrame.nCamera; ++i)
            // {
            //     if (i < mCurrentFrame.nCamera-1)
            //         mvnFeaturesCams[i] = (int) ((float)mnFeatures * vfFeaturesResponse[i] / total_response);
            //     else
            //     {
            //         mvnFeaturesCams[i] = (int) ((float)mnFeatures * vfFeaturesResponse[i] / total_response);
            //         mvnFeaturesCams[i+1] = mvnFeaturesCams[i];
            //     }
            // }

            // if (mpAtlas->KeyFramesInMap() > 10)
            // {
            //     vector<float> ratio_cam(mCurrentFrame.nCamera, 0);
                
            //     for (int i = 0; i < mCurrentFrame.nCamera; ++i)
            //     {
            //         std::cout << "cam: " << i << "response: " << vfFeaturesResponse[i] << std::endl;
            //         ratio_cam[i] = (float) vnTrackFeaturesCam[i] / mvnFeaturesCams[i];
            //         if (ratio_cam[i] < 0.05)
            //         {
            //             if (mvnFeaturesCams[i] < 40) continue;
            //             if (i < mCurrentFrame.nCamera-1)
            //             {
            //                 mnTrackFeatures += mvnFeaturesCams[i] * 0.3;
            //                 mvnFeaturesCams[i] *= 0.7;
            //             }
            //             else
            //             {
            //                 mnTrackFeatures += mvnFeaturesCams[i] * 0.6;
            //                 mvnFeaturesCams[i] *= 0.7;
            //                 mvnFeaturesCams[i+1] = mvnFeaturesCams[i];
            //             }
            //         }
            //     }

            //     float total_ratio = 0;
            //     int nAddTrackFeats = 0;
            //     if (mnTrackFeatures > 0)
            //     {
            //         for (int i = 0; i < mCurrentFrame.nCamera; ++i)
            //         {
            //             if (ratio_cam[i] >= 0.05)
            //             {
            //                 if (i < mCurrentFrame.nCamera-1)
            //                 {
            //                     total_ratio += ratio_cam[i];
            //                 }
            //                 else
            //                     total_ratio += 2 * ratio_cam[i];
            //             }
            //         }

            //         for (int i = 0; i < mCurrentFrame.nCamera; ++i)
            //         {
            //             if (ratio_cam[i] >= 0.05)
            //             {
            //                 if (i < mCurrentFrame.nCamera-1)
            //                 {
            //                     int nAddTrackFeat = (int) ((float)mnTrackFeatures * ratio_cam[i] / total_ratio);
            //                     mvnFeaturesCams[i] += nAddTrackFeat;
            //                     nAddTrackFeats += nAddTrackFeat;
            //                 }
            //                 else
            //                 {
            //                     int nAddTrackFeat = (int) ((float)mnTrackFeatures * ratio_cam[i] / total_ratio);
            //                     mvnFeaturesCams[i] += nAddTrackFeat;
            //                     mvnFeaturesCams[i+1] = mvnFeaturesCams[i];
            //                     nAddTrackFeats += 2 * nAddTrackFeat;
            //                 }
            //             }
            //         }

            //     }

            //     mnTrackFeatures -= nAddTrackFeats;
            //     if (mnTrackFeatures < 0) mnTrackFeatures = 0;
            // }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
#endif
            bool bNeedKF = NeedNewKeyFrame();

            if (bNeedKF && bOK)
            {
                // std::cout << "CreateNewKeyFrame" << std::endl;
                CreateNewKeyFrame();
            }

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

            double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndNewKF - time_StartNewKF).count();
            vdNewKF_ms.push_back(timeNewKF);
#endif

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            // std::cout << "add gp observation" << std::endl;
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                {
                    // 添加地图点的非关键帧观测，使高斯过程有充足的约束，外点在后端优化中剔除
                    if (!bNeedKF && bOK)
                    {
                        int cam = mCurrentFrame.mmpKeyToCam[i];
                        int localID= mCurrentFrame.mmpGlobalToLocalID[i];
                        float ur = -1;
                        if (cam == mCurrentFrame.nCamera-1) ur = mCurrentFrame.mvuRight[localID];
                        // pMP->AddGPObservation(mpLastKeyFrame, mCurrentFrame.mvTimeStamps[cam], 
                        //                 cam, mCurrentFrame.mvKeysUn[i], ur);
                    }   
                    if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    {
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
                }
            }

        }

        if (mState == LOST)
        {
            // TODO: 跟踪丢失处理
        }

        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastPrevFrame = MultiFrame(mLastFrame);
        mLastFrame = MultiFrame(mCurrentFrame);
        mLastFrame.mpPrevFrame = &mLastPrevFrame;

    }

    if (mState == OK || mState == RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if(mCurrentFrame.isSet())
        {
            Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr_);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState==LOST);
        }
        else
        {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState==LOST);
        }
    }

}

bool Tracking::Relocalization()
{
    std::cout << "Start relocalization" << std::endl;

    mCurrentFrame.ComputeBoW();

    vector<MultiKeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

    if (vpCandidateKFs.empty())
    {
        std::cout << "No relocalization candidates" << std::endl;
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    ORBmatcher matcher(0.75, true);

    return true;
}

void Tracking::StereoInitialization()
{
        mCurrentFrame.SetPose(Sophus::SE3f());

        MultiKeyFrame* pKFini = new MultiKeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

        mpAtlas->AddKeyFrame(pKFini);

        for (int i = 0; i < mCurrentFrame.N; ++i) {
            int cameraID = mCurrentFrame.nCamera-1;
            if (mCurrentFrame.mmpKeyToCam[i] != cameraID)
                continue;
            int localId = mCurrentFrame.mmpGlobalToLocalID[i];
            float z = mCurrentFrame.mvDepth[localId];
            if (z > 0) {
                Eigen::Vector3f x3D;
                mCurrentFrame.UnprojectStereo(i, x3D);
                MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                pNewMP->AddObservation(pKFini, i);
                pKFini->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpAtlas->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i] = pNewMP;
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        mpLocalMapper->InsertKeyFrame(pKFini);

       

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = pKFini->GetMapPointMatches();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mLastFrame = MultiFrame(mCurrentFrame);
        mLastFrame.mpPrevFrame = nullptr;
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;       

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(pKFini->GetPose());

        mState = OK;
}


// void Tracking::MonocularInitialization()
// {

//     if(!mbReadyToInitializate)
//     {
//         // Set Reference Frame
//         if(mCurrentFrame.mvKeys.size()>100)
//         {

//             mInitialFrame = MultiFrame(mCurrentFrame);
//             mLastFrame = MultiFrame(mCurrentFrame);
//             mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
//             for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
//                 mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

//             fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

//             mbReadyToInitializate = true;

//             return;
//         }
//     }
//     else
//     {
//         if (((int)mCurrentFrame.mvKeys.size()<=100)||((mSensor == System::IMU_MONOCULAR)&&(mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0)))
//         {
//             mbReadyToInitializate = false;

//             return;
//         }

//         // Find correspondences
//         ORBmatcher matcher(0.9,true);
//         int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

//         // Check if there are enough correspondences
//         if(nmatches<100)
//         {
//             mbReadyToInitializate = false;
//             return;
//         }

//         Sophus::SE3f Tcw;
//         vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

//         if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,mCurrentFrame.mvKeysUn,mvIniMatches,Tcw,mvIniP3D,vbTriangulated))
//         {
//             for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
//             {
//                 if(mvIniMatches[i]>=0 && !vbTriangulated[i])
//                 {
//                     mvIniMatches[i]=-1;
//                     nmatches--;
//                 }
//             }

//             // Set Frame Poses
//             mInitialFrame.SetPose(Sophus::SE3f());
//             mCurrentFrame.SetPose(Tcw);

//             CreateInitialMapMonocular();
//         }
//     }
// }

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    MultiKeyFrame* pKFini = new MultiKeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    MultiKeyFrame* pKFcur = new MultiKeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap());

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpAtlas->AddMapPoint(pMP);
    }


    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);

    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth;

    invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
        mpSystem->ResetActiveMap();
        return;
    }

    // Scale initial baseline
    Sophus::SE3f Tc2w = pKFcur->GetPose();
    Tc2w.translation() *= invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp;

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;
    //mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    // Compute here initial velocity
    vector<MultiKeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();

    Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
    mbVelocity = false;
    Eigen::Vector3f phi = deltaT.so3().log();

    double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp);
    phi *= aux;

    mLastFrame = MultiFrame(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;

    initID = pKFcur->mnId;
}


void Tracking::CreateMapInAtlas()
{
    mnLastInitFrameId = mCurrentFrame.mnId;
    mpAtlas->CreateNewMap();
    // if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
    //     mpAtlas->SetInertialSensor();
    mbSetInit=false;

    mnInitialFrameId = mCurrentFrame.mnId+1;
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mbVelocity = false;
    //mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    // if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    // {
    //     mbReadyToInitializate = false;
    // }

    // if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
    // {
    //     delete mpImuPreintegratedFromLastKF;
    //     mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    // }

    if(mpLastKeyFrame)
        mpLastKeyFrame = static_cast<MultiKeyFrame*>(NULL);

    if(mpReferenceKF)
        mpReferenceKF = static_cast<MultiKeyFrame*>(NULL);

    mLastFrame = MultiFrame();
    mCurrentFrame = MultiFrame();
    mvIniMatches.clear();

    mbCreatedMap = true;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    MultiKeyFrame* pRef = mLastFrame.mpReferenceKF;
    Sophus::SE3f Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    const int Nfeat = mLastFrame.Nleft == -1? mLastFrame.N : mLastFrame.Nleft;
    vDepthIdx.reserve(Nfeat);
    for(int i=0; i<Nfeat;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    std::sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
            bCreateNew = true;

        if(bCreateNew)
        {
            Eigen::Vector3f x3D;

            if(mLastFrame.Nleft == -1){
                mLastFrame.UnprojectStereo(i, x3D);
            }
            else{
                x3D = mLastFrame.UnprojectStereoFishEye(i);
            }

            MapPoint* pNewMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mLastFrame,i);
            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;

    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    // std::cout << "SetPose" << std::endl;
    // mCurrentFrame.SetPose((Sophus::SE3f::exp((mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp) * mLastFrame.GetVelocity()) * mLastFrame.GetPose().inverse()).inverse());
    // mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
    double dt = mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp;

    mCurrentFrame.SetVelocity(mVelocity / dt);
    // mCurrentFrame.SetPose(Sophus::SE3f::exp(-(mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp) * mLastFrame.GetVelocity()) * mLastFrame.GetPose());
    mCurrentFrame.SetPose(Sophus::SE3f::exp(-mVelocity) * mLastFrame.GetPose());

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    int th = 7;

    // std::cout << "SearchByProjection" << std::endl;
    // mCurrentFrame.mMatchedFeatures.reserve(1000);
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame, th, false);

    // int inliers = MCRansac(&mCurrentFrame, &mLastFrame, 10, 30);

    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame ,2*th, false);
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);
    }

    if (nmatches < 20) {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    Optimizer::PoseGPOptimizationFromeLastFrame(&mCurrentFrame, false);
    // std::cout << "Pose optimization done" << std::endl;

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; ++i)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                int cam_idx = mCurrentFrame.mmpKeyToCam[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mvbTrackInView[cam_idx] = false;
                pMP->mvnLastFrameSeen[cam_idx] = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    // std::cout << "Matches after optimization: " << nmatchesMap << std::endl;

    return nmatchesMap >= 10;
}

bool Tracking::TrackReferenceKeyFrame()
{
    mCurrentFrame.ComputeBoW();

    ORBmatcher matcher(0.7, true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
    {
        cout << "Not enough matches with reference keyframe" << endl;
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetVelocity(mLastFrame.GetVelocity());
    mCurrentFrame.SetPose(mLastFrame.GetPose());

    Optimizer::PoseGPOptimizationFromeLastFrame(&mCurrentFrame, false);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; ++i)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                int cam_idx = mCurrentFrame.mmpKeyToCam[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mvbTrackInView[cam_idx] = false;
                pMP->mvnLastFrameSeen[cam_idx] = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                nmatchesMap++;
        }
    }

    return nmatchesMap >= 10;
}

int Tracking::MCRansac(MultiFrame* CurrentFrame, MultiFrame* LastFrame, int maxIt, int minMatch)
{
    int it = 0, bestInliers = 0;

    Eigen::Matrix<double, 6, 1> bestVel;

    std::unordered_set<int> bestSamples;

    const int min_set = 3;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int> vMatchedFeatures(CurrentFrame->mMatchedFeatures.begin(), CurrentFrame->mMatchedFeatures.end());

    std::uniform_int_distribution<int> dis(0, vMatchedFeatures.size() - 1);

    std::vector<bool> vbBestInliers(vMatchedFeatures.size(), false);

    while (it < maxIt)
    {
        ++it;
        // std::shuffle(vMatchedFeatures.begin(), vMatchedFeatures.end(), gen);
        std::unordered_set<int> samples;
        samples.reserve(min_set);

        while (samples.size() < min_set)
        {
            int randi = dis(gen);
            int index = vMatchedFeatures[randi];

            if (samples.find(index) == samples.end())
                samples.insert(index);
        }

        Eigen::Matrix<double, 6, 1> vel;
        std::vector<bool> vbInliers(vMatchedFeatures.size(), false);
        int inliers = Optimizer::OptimizeVel(CurrentFrame, LastFrame, vMatchedFeatures, samples, vel, vbInliers);

        if (inliers > bestInliers)
        {
            bestInliers = inliers;
            bestVel = vel;
            bestSamples = samples;
            vbBestInliers = vbInliers;
        }
    }

    if (bestInliers < minMatch)
        return 0;

    int outliers = 0;
    for (int i = 0; i < vMatchedFeatures.size(); ++i)
    {
        if (!vbBestInliers[i])
        {
            CurrentFrame->mvbOutlier[vMatchedFeatures[i]] = true;
            ++outliers;
        }
    }
    // std::cout << "inliers: " << bestInliers << " outliers: " << outliers << std::endl;
    // std::cout << "size: " << vMatchedFeatures.size() << std::endl;
    return bestInliers;
}

bool Tracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    mTrackedFr++;

    // Update Local KeyFrames
    UpdateLocalMap();
    SearchLocalPoints();

    // mCurrentFrame.mMatchedFeatures.reserve(1000);
    // TOO check outliers before PO
    int aux1 = 0, aux2=0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
            else
                mCurrentFrame.mMatchedFeatures.insert(i);
        }

    std::chrono::steady_clock::time_point time_StartPO = std::chrono::steady_clock::now();
    int ransac_inliers = MCRansac(&mCurrentFrame, &mLastFrame, 23, 30);
    std::chrono::steady_clock::time_point time_EndPO = std::chrono::steady_clock::now();
    float timePO = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPO - time_StartPO).count();
    vrtime.push_back(timePO);

    int inliers;

    inliers = Optimizer::PoseGPOptimizationFromeLastFrame(&mCurrentFrame, false);

    aux1 = 0, aux2 = 0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    mnMatchesInliers = 0;

    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else 
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }

    mpLocalMapper->mnMatchesInliers = mnMatchesInliers;

    // std::cout << "inliers: " << mnMatchesInliers << std::endl;

    if (mCurrentFrame.mnId < mnLastRelocFrameId+mMaxFrames && mnMatchesInliers < 50)
        return false;
    
    if ((mnMatchesInliers > 10) && (mState == RECENTLY_LOST))
        return true;

    if (mnMatchesInliers < 30)
        return false;
    else
        return true;
}

bool Tracking::NeedNewKeyFrame()
{
    
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        /*if(mSensor == System::MONOCULAR)
        {
            std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
        }*/
        return false;
    }

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;

    int N_prev = 0;
    for (int c = 0; c < mCurrentFrame.nCamera-1; ++c) N_prev += mCurrentFrame.mvpKeysCams[c].size();
    for (int i = 0; i < mCurrentFrame.mvDepth.size(); ++i)
    {
        int n = i + N_prev;
        if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
        {
            if (mCurrentFrame.mvpMapPoints[n] && !mCurrentFrame.mvbOutlier[n])
                nTrackedClose++;
            else
                nNonTrackedClose++;
        } 
    }

    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    thRefRatio = 0.75f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle); 
    //Condition 1c: tracking is weak
    const bool c1c = (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    // Temporal condition for continus motion model
    bool c3 = false;
    if (mpLastKeyFrame)
    {
        // if (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp >= 1.0)
            // c3 = true;

        Sophus::SE3f deltaT = mpLastKeyFrame->GetPose() * mCurrentFrame.GetPoseW();
        // 0.08
        if (deltaT.translation().norm() > 2 || deltaT.so3().log().norm() > 0.08)
            c3 = true;
    }

    bool c4 = false;
    if (((mnMatchesInliers < 75) && (mnMatchesInliers > 15)) || mState == RECENTLY_LOST)
        c4 = true;
    else
        c4 = false;

    bool c5  = false;
    float v = mCurrentFrame.GetVelocity().block<3,1>(0,0).norm();
    float w = mCurrentFrame.GetVelocity().block<3,1>(3,0).norm();
    if (v < 0.3 && w < 0.1)
        c5 = true;
    else
        c5 = false;

    if (((c1a || c1b || c1c) && c2) || c3 || c4)
    {
        if (!c3 && c5) return false;
        if (bLocalMappingIdle || mpLocalMapper->IsInitializing())
            return true;
        else
        {
            mpLocalMapper->InterruptBA();
            if (mpLocalMapper->KeyframesInQueue() < 3)
                return true;
            else
                return false;
        }
    }
    else 
        return false;

}

void Tracking::CreateNewKeyFrame()
{

    if (mpLocalMapper->IsInitializing())
        return;

    if (!mpLocalMapper->SetNotStop(true))
        return;

    MultiKeyFrame* pKF = new MultiKeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

    mpReferenceKF = pKF;

    mCurrentFrame.mpReferenceKF = pKF;

    if (mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
        Verbose::PrintMess("No last keyframe", Verbose::VERBOSITY_NORMAL);

    mCurrentFrame.UpdatePoseMatrices();
    // cout << "create new MPs" << endl;
    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    int maxPoint = 100;
    vector<pair<float,int>> vDepthIdx;
    int N_stereo = mCurrentFrame.mvpKeysCams[mCurrentFrame.nCamera-1].size();
    int N_prev = 0;
    for (int c = 0; c < mCurrentFrame.nCamera-1; ++c) N_prev += mCurrentFrame.mvpKeysCams[c].size();
    vDepthIdx.reserve(N_stereo);

    for (int i = 0; i < N_stereo; ++i)
    {
        float z = mCurrentFrame.mvDepth[i];
        if (z > 0)
            vDepthIdx.push_back(make_pair(z, i));
    }

    if (!vDepthIdx.empty())
    {
        std::sort(vDepthIdx.begin(), vDepthIdx.end());

        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); ++j)
        {
            int i = vDepthIdx[j].second + N_prev;
            bool bCreateNew = false;

            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1)
            {
                bCreateNew = true;
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
            }

            if (bCreateNew)
            {
                Eigen::Vector3f x3D;

                mCurrentFrame.UnprojectStereo(i, x3D);

                MapPoint* pNewMP = new MapPoint(x3D, pKF, mpAtlas->GetCurrentMap());
                pNewMP->AddObservation(pKF, i);

                pKF->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpAtlas->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i] = pNewMP;
                nPoints++;
            }
            else
                nPoints++;
            
            if (vDepthIdx[j].first > mThDepth && nPoints > maxPoint)
                break;
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);
    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;

} 

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for (int i = 0, iend = mCurrentFrame.N; i < iend; i++)
    {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP)
        {
            if (pMP->isBad())
            {
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
            }
            else
            {
                int cam = mCurrentFrame.mmpKeyToCam[i];
                pMP->mvnLastFrameSeen[cam] = mCurrentFrame.mnId;
                pMP->mvbTrackInView[cam] = false;
                pMP->mvbTrackInViewR[cam] = false;
            }
            
        }
    }

    int nToMatch = 0;
    // Project points in frame and check its visibility
    for (vector<MapPoint*>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; ++vit)
    {
        MapPoint* pMP = *vit;

        if (pMP->isBad())
            continue;

        for (int c = 0; c < mCurrentFrame.nCamera; ++c)
        {
            if (pMP->mvnLastFrameSeen[c] == mCurrentFrame.mnId)
                continue;

            // Project (this fills MapPoint variables for matching)
            if (mCurrentFrame.isInFrustum(c, pMP, 0.5))
            {
                pMP->IncreaseVisible();
                nToMatch++;
            }

            if (pMP->mvbTrackInView[c])
            {
                mCurrentFrame.mvmProjectPoints[c][pMP->mnId] = cv::Point2f(pMP->mvTrackProjX[c], pMP->mvTrackProjY[c]);
            }
        }
    }

    if (nToMatch > 0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;

        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
    }

}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    int count_pts = 0;

    for(vector<MultiKeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {
        MultiKeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {

            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<MultiKeyFrame*,int> keyframeCounter;
    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2))
    {
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if (pMP->mnUpdateLocalKF == mCurrentFrame.mnId)
                    continue;
                if(!pMP->isBad())
                {
                    const map<MultiKeyFrame*, vector<int>> observations = pMP->GetObservations();
                    for(map<MultiKeyFrame*, vector<int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                    pMP->mnUpdateLocalKF = mCurrentFrame.mnId;
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    else
    {
        // error
        std::cout << "error!" << endl;
    }
    // else
    // {
    //     for(int i=0; i<mLastFrame.N; i++)
    //     {
    //         // Using lastframe since current frame has not matches yet
    //         if(mLastFrame.mvpMapPoints[i])
    //         {
    //             MapPoint* pMP = mLastFrame.mvpMapPoints[i];
    //             if(!pMP)
    //                 continue;
    //             if(!pMP->isBad())
    //             {
    //                 const map<MultiKeyFrame*, vector<int>> observations = pMP->GetObservations();
    //                 for(map<MultiKeyFrame*, vector<int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
    //                     keyframeCounter[it->first]++;
    //             }
    //             else
    //             {
    //                 // MODIFICATION
    //                 mLastFrame.mvpMapPoints[i]=NULL;
    //             }
    //         }
    //     }
    // }


    int max=0;
    MultiKeyFrame* pKFmax= static_cast<MultiKeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<MultiKeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        MultiKeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<MultiKeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80) // 80
            break;

        MultiKeyFrame* pKF = *itKF;

        const vector<MultiKeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);


        for(vector<MultiKeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            MultiKeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<MultiKeyFrame*> spChilds = pKF->GetChilds();
        for(set<MultiKeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            MultiKeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        MultiKeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if(mSensor == System::MULTICAMERA && mvpLocalKeyFrames.size()<80)
    {
        MultiKeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for(int i=0; i<Nd; i++){
            if (!tempKeyFrame)
                break;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                tempKeyFrame=tempKeyFrame->mPrevKF;
            }
        }
    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

// TODO: 重定位实现
// bool Tracking::Relocalization()
// {
//     Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
//     // Compute Bag of Words Vector
//     mCurrentFrame.ComputeBoW();

//     // Relocalization is performed when tracking is lost
//     // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
//     vector<MultiKeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

//     if(vpCandidateKFs.empty()) {
//         Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
//         return false;
//     }

//     const int nKFs = vpCandidateKFs.size();

//     // We perform first an ORB matching with each candidate
//     // If enough matches are found we setup a PnP solver
//     ORBmatcher matcher(0.75,true);

//     vector<MLPnPsolver*> vpMLPnPsolvers;
//     vpMLPnPsolvers.resize(nKFs);

//     vector<vector<MapPoint*> > vvpMapPointMatches;
//     vvpMapPointMatches.resize(nKFs);

//     vector<bool> vbDiscarded;
//     vbDiscarded.resize(nKFs);

//     int nCandidates=0;

//     for(int i=0; i<nKFs; i++)
//     {
//         MultiKeyFrame* pKF = vpCandidateKFs[i];
//         if(pKF->isBad())
//             vbDiscarded[i] = true;
//         else
//         {
//             int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
//             if(nmatches<15)
//             {
//                 vbDiscarded[i] = true;
//                 continue;
//             }
//             else
//             {
//                 MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
//                 pSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
//                 vpMLPnPsolvers[i] = pSolver;
//                 nCandidates++;
//             }
//         }
//     }

//     // Alternatively perform some iterations of P4P RANSAC
//     // Until we found a camera pose supported by enough inliers
//     bool bMatch = false;
//     ORBmatcher matcher2(0.9,true);

//     while(nCandidates>0 && !bMatch)
//     {
//         for(int i=0; i<nKFs; i++)
//         {
//             if(vbDiscarded[i])
//                 continue;

//             // Perform 5 Ransac Iterations
//             vector<bool> vbInliers;
//             int nInliers;
//             bool bNoMore;

//             MLPnPsolver* pSolver = vpMLPnPsolvers[i];
//             Eigen::Matrix4f eigTcw;
//             bool bTcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers, eigTcw);

//             // If Ransac reachs max. iterations discard keyframe
//             if(bNoMore)
//             {
//                 vbDiscarded[i]=true;
//                 nCandidates--;
//             }

//             // If a Camera Pose is computed, optimize
//             if(bTcw)
//             {
//                 Sophus::SE3f Tcw(eigTcw);
//                 mCurrentFrame.SetPose(Tcw);
//                 // Tcw.copyTo(mCurrentFrame.mTbw);

//                 set<MapPoint*> sFound;

//                 const int np = vbInliers.size();

//                 for(int j=0; j<np; j++)
//                 {
//                     if(vbInliers[j])
//                     {
//                         mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
//                         sFound.insert(vvpMapPointMatches[i][j]);
//                     }
//                     else
//                         mCurrentFrame.mvpMapPoints[j]=NULL;
//                 }

//                 int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

//                 if(nGood<10)
//                     continue;

//                 for(int io =0; io<mCurrentFrame.N; io++)
//                     if(mCurrentFrame.mvbOutlier[io])
//                         mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

//                 // If few inliers, search by projection in a coarse window and optimize again
//                 if(nGood<50)
//                 {
//                     int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

//                     if(nadditional+nGood>=50)
//                     {
//                         nGood = Optimizer::PoseOptimization(&mCurrentFrame);

//                         // If many inliers but still not enough, search by projection again in a narrower window
//                         // the camera has been already optimized with many points
//                         if(nGood>30 && nGood<50)
//                         {
//                             sFound.clear();
//                             for(int ip =0; ip<mCurrentFrame.N; ip++)
//                                 if(mCurrentFrame.mvpMapPoints[ip])
//                                     sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
//                             nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

//                             // Final optimization
//                             if(nGood+nadditional>=50)
//                             {
//                                 nGood = Optimizer::PoseOptimization(&mCurrentFrame);

//                                 for(int io =0; io<mCurrentFrame.N; io++)
//                                     if(mCurrentFrame.mvbOutlier[io])
//                                         mCurrentFrame.mvpMapPoints[io]=NULL;
//                             }
//                         }
//                     }
//                 }


//                 // If the pose is supported by enough inliers stop ransacs and continue
//                 if(nGood>=50)
//                 {
//                     bMatch = true;
//                     break;
//                 }
//             }
//         }
//     }

//     if(!bMatch)
//     {
//         return false;
//     }
//     else
//     {
//         mnLastRelocFrameId = mCurrentFrame.mnId;
//         cout << "Relocalized!!" << endl;
//         return true;
//     }

// }

// void Tracking::Reset(bool bLocMap)
// {
//     Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

//     if(mpViewer)
//     {
//         mpViewer->RequestStop();
//         while(!mpViewer->isStopped())
//             usleep(3000);
//     }

//     // Reset Local Mapping
//     if (!bLocMap)
//     {
//         Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
//         mpLocalMapper->RequestReset();
//         Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
//     }


//     // Reset Loop Closing
//     Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
//     mpLoopClosing->RequestReset();
//     Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

//     // Clear BoW Database
//     Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
//     mpKeyFrameDB->clear();
//     Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

//     // Clear Map (this erase MapPoints and KeyFrames)
//     mpAtlas->clearAtlas();
//     mpAtlas->CreateNewMap();
//     if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
//         mpAtlas->SetInertialSensor();
//     mnInitialFrameId = 0;

//     MultiKeyFrame::nNextId = 0;
//     MultiFrame::nNextId = 0;
//     mState = NO_IMAGES_YET;

//     mbReadyToInitializate = false;
//     mbSetInit=false;

//     mlRelativeFramePoses.clear();
//     mlpReferences.clear();
//     mlFrameTimes.clear();
//     mlbLost.clear();
//     mCurrentFrame = MultiFrame();
//     mnLastRelocFrameId = 0;
//     mLastFrame = MultiFrame();
//     mpReferenceKF = static_cast<MultiKeyFrame*>(NULL);
//     mpLastKeyFrame = static_cast<MultiKeyFrame*>(NULL);
//     mvIniMatches.clear();

//     if(mpViewer)
//         mpViewer->Release();

//     Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
// }

void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();


    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = MultiFrame::nNextId;
    //mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    mbReadyToInitializate = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    std::cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    std::cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    std::cout << num_lost << " Frames set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = MultiFrame();
    mLastFrame = MultiFrame();
    mpReferenceKF = static_cast<MultiKeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<MultiKeyFrame*>(NULL);
    mvIniMatches.clear();

    mbVelocity = false;

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

// void Tracking::ChangeCalibration(const string &strSettingPath)
// {
//     cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
//     float fx = fSettings["Camera.fx"];
//     float fy = fSettings["Camera.fy"];
//     float cx = fSettings["Camera.cx"];
//     float cy = fSettings["Camera.cy"];

//     mK_.setIdentity();
//     mK_(0,0) = fx;
//     mK_(1,1) = fy;
//     mK_(0,2) = cx;
//     mK_(1,2) = cy;

//     cv::Mat K = cv::Mat::eye(3,3,CV_32F);
//     K.at<float>(0,0) = fx;
//     K.at<float>(1,1) = fy;
//     K.at<float>(0,2) = cx;
//     K.at<float>(1,2) = cy;
//     K.copyTo(mK);

//     cv::Mat DistCoef(4,1,CV_32F);
//     DistCoef.at<float>(0) = fSettings["Camera.k1"];
//     DistCoef.at<float>(1) = fSettings["Camera.k2"];
//     DistCoef.at<float>(2) = fSettings["Camera.p1"];
//     DistCoef.at<float>(3) = fSettings["Camera.p2"];
//     const float k3 = fSettings["Camera.k3"];
//     if(k3!=0)
//     {
//         DistCoef.resize(5);
//         DistCoef.at<float>(4) = k3;
//     }
//     DistCoef.copyTo(mDistCoef);

//     mbf = fSettings["Camera.bf"];

//     MultiFrame::mbInitialComputations = true;
// }

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, MultiKeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId;
    list<ORB_SLAM3::MultiKeyFrame*>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for(auto lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
    {
        if(*lbL)
            continue;

        MultiKeyFrame* pKF = *lRit;

        while(pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if(pKF->GetMap() == pMap)
        {
            (*lit).translation() *= s;
        }
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    while(!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }


    if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    //mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap)
{
    mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
    if(!strNameFile_kf.empty())
        mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
}

float Tracking::GetImageScale()
{
    return mImageScale;
}

#ifdef REGISTER_LOOP
void Tracking::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Tracking::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Tracking STOP" << endl;
        return true;
    }

    return false;
}

bool Tracking::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool Tracking::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void Tracking::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
}
#endif

} //namespace ORB_SLAM
