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

#include "Frame.h"

#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"

#include <thread>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>

namespace ORB_SLAM3
{

long unsigned int MultiFrame::nNextId=0;
int MultiFrame::nCamera=0;
bool MultiFrame::mbInitialComputations=true;
std::vector<float> MultiFrame::mvcx, MultiFrame::mvcy, MultiFrame::mvfx, MultiFrame::mvfy, MultiFrame::minvfx, MultiFrame::minvfy;
std::vector<float> MultiFrame::mvMaxX, MultiFrame::mvMinX, MultiFrame::mvMaxY, MultiFrame::mvMinY, MultiFrame::mvGridElementWidthInv, MultiFrame::mvGridElementHeightInv;
std::vector<Sophus::SE3f> MultiFrame::mTbc;
std::vector<Sophus::SO3f> MultiFrame::mRbc_ini;
std::vector<Eigen::Matrix3d> MultiFrame::mRbc_ini_cov;

Eigen::Matrix<double, 6, 1> MultiFrame::iniVel = Eigen::Matrix<double, 6, 1>::Zero();

//For stereo fisheye matching
cv::BFMatcher MultiFrame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

MultiFrame::MultiFrame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<MultiKeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(false)
{
#ifdef REGISTER_TIMES
    mTimeStereoMatch = 0;
    mTimeORB_Ext = 0;
#endif
}


//Copy Constructor
MultiFrame::MultiFrame(const MultiFrame &frame)
    :mpcpi(frame.mpcpi), mpGP(frame.mpGP), mpORBvocabulary(frame.mpORBvocabulary), mvpORBextractor(frame.mvpORBextractor),
    mTimeStamp(frame.mTimeStamp), mvTimeStamps(frame.mvTimeStamps), 
    mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
    mvKeysUn(frame.mvKeysUn), mmpKeyToCam(frame.mmpKeyToCam), mmpGlobalToLocalID(frame.mmpGlobalToLocalID), 
    mvuRight(frame.mvuRight), mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mvBowVecs(frame.mvBowVecs), mvFeatVecs(frame.mvFeatVecs), mFeatVec(frame.mFeatVec),
    mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
    mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
    mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
    mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
    mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
    mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame),
    mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
    mvpCamera(frame.mvpCamera), Nleft(frame.Nleft), Nright(frame.Nright),
    vmono(frame.vmono), mvLeftToRightMatch(frame.mvLeftToRightMatch),
    mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
    mTlr(frame.mTlr), mRlr(frame.mRlr), mtlr(frame.mtlr), mTrl(frame.mTrl),
    mTbw(frame.mTbw), mTcw(frame.mTcw), mTwc(frame.mTwc), mbHasPose(false), mbHasVelocity(false)
{
    mvK.reserve(frame.mvK.size());
    mvK_.reserve(frame.mvK_.size());
    mvDistCoef.reserve(frame.mvDistCoef.size());
    mvDescriptors.reserve(frame.mvDescriptors.size());

    for (auto &i : frame.mvK) {
        mvK.push_back(i.clone());
        mvK_.push_back(Converter::toMatrix3f(i));
    }

    for (auto &i : frame.mvDistCoef) {
        mvDistCoef.push_back(i.clone());
    }

    for (auto &i : frame.mvDescriptors) {
        mvDescriptors.push_back(i.clone());
    }

    mvGrid.resize(frame.mvGrid.size());
    // TODO:是否要赋值mGridRight？
    for (int c = 0; c < frame.mvGrid.size(); ++c) {
        for (int i = 0; i < FRAME_GRID_COLS; ++i) {
            for (int j = 0; j < FRAME_GRID_ROWS; ++j) {
                mvGrid[c][i][j] = frame.mvGrid[c][i][j];
            }
        }
    }

    // for(int i=0;i<FRAME_GRID_COLS;i++)
    //     for(int j=0; j<FRAME_GRID_ROWS; j++){
    //         mGrid[i][j]=frame.mGrid[i][j];
    //         if(frame.Nleft > 0){
    //             mGridRight[i][j] = frame.mGridRight[i][j];
    //         }
    //     }
    if(frame.HasVelocity())
    {
        SetVelocity(frame.GetVelocity());
    }

    if(frame.mbHasPose)
        SetPose(frame.GetPose(), frame.mTwc);

    mvmProjectPoints = frame.mvmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;

#ifdef REGISTER_TIMES
    mTimeStereoMatch = frame.mTimeStereoMatch;
    mTimeORB_Ext = frame.mTimeORB_Ext;
#endif
}

MultiFrame::MultiFrame(const std::vector<cv::Mat> &vimgs, const std::vector<double> &vtimesStamps, std::vector<Sophus::SE3f> &Tbc, std::vector<ORBextractor*> &vextractors, GaussianProcess* pGP, ORBVocabulary* voc, std::vector<GeometricCamera*> &vpCameras, std::vector<cv::Mat> &vK, std::vector<cv::Mat> &vdistCoef, const float &bf, const float &thDepth, MultiFrame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpGP(pGP), mpORBvocabulary(voc), mvpORBextractor(vextractors), mTimeStamp(vtimesStamps.back()), mvTimeStamps(vtimesStamps),
     mbf(bf), mb(thDepth), mThDepth(thDepth), mImuCalib(ImuCalib), mpPrevFrame(pPrevF), mpImuPreintegrated(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<MultiKeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(true), mvpCamera(vpCameras)
{
    mvK.reserve(vK.size());
    mvK_.reserve(vK.size());
    mvDistCoef.reserve(vdistCoef.size());
    for (auto &i : vK) {
        mvK.push_back(i.clone());
        mvK_.push_back(Converter::toMatrix3f(i));
    }

    for (auto &i : vdistCoef) {
        mvDistCoef.push_back(i.clone());
    }


    mnId = nNextId++;

    if (mnId == 0)
        mpPrevFrame = nullptr;

    mnScaleLevels = mvpORBextractor[0]->GetLevels();
    mfScaleFactor = mvpORBextractor[0]->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mvpORBextractor[0]->GetScaleFactors();
    mvInvScaleFactors = mvpORBextractor[0]->GetInverseScaleFactors();
    mvLevelSigma2 = mvpORBextractor[0]->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mvpORBextractor[0]->GetInverseScaleSigmaSquares();

    if (mbInitialComputations) {
        mTbc = Tbc;
        mRbc_ini.resize(mTbc.size());
        mRbc_ini_cov.resize(mTbc.size());
        nCamera = mvpCamera.size();
        mvfx.resize(nCamera);
        mvfy.resize(nCamera);
        mvcx.resize(nCamera);
        mvcy.resize(nCamera);
        minvfx.resize(nCamera);
        minvfy.resize(nCamera);
        mvMaxX.resize(nCamera);
        mvMinX.resize(nCamera);
        mvMaxY.resize(nCamera);
        mvMinY.resize(nCamera);
        mvGridElementWidthInv.resize(nCamera);
        mvGridElementHeightInv.resize(nCamera);
        mTwist << iniVel.cast<float>();
#pragma omp parallel for num_threads(nCamera)
        for (int c = 0; c < nCamera; ++c) {
            mRbc_ini[c] = mTbc[c].so3();
            mRbc_ini_cov[c] = Eigen::Matrix3d::Identity() * 0.2;
            mvfx[c] = mvK[c].at<float>(0, 0);
            mvfy[c] = mvK[c].at<float>(1, 1);
            mvcx[c] = mvK[c].at<float>(0, 2);
            mvcy[c] = mvK[c].at<float>(1, 2);
            minvfx[c] = 1.0f / mvfx[c];
            minvfy[c] = 1.0f / mvfy[c];

            ComputeImageBounds(vimgs.back(), c);

            mvGridElementWidthInv[c] = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mvMaxX[c] - mvMinX[c]);
            mvGridElementHeightInv[c] = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mvMaxY[c] - mvMinY[c]); 
        }
        mbInitialComputations = false;
    }

    mb = mbf / mvfx[nCamera-1];

    // 最后一个是双目相机的右图像，不需要去畸变
    mTcw.resize(nCamera, Sophus::SE3f());
    mTwc = mTcw;
    mvDescriptors.resize(nCamera+1);
    mvpKeysCams.resize(nCamera+1);
    mvpKeysUnCams.resize(nCamera);
    vmono.resize(nCamera+1);
    mvGrid.resize(nCamera);

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif

#pragma omp parallel for num_threads(nCamera+1)
    for (int c = 0; c <= nCamera; ++c) {
        if (mnId == 0 && c < nCamera-1) continue;
        ExtractORB(c, vimgs[c], 0, 0);
        UndistortKeyPoints(c);
        // 提前准备栅格的容量
        if (c == nCamera) continue;
        const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;
        int nReserve = 0.5f*mvpKeysCams[c].size()/(nCells);
        for (unsigned int i = 0; i < FRAME_GRID_COLS; i++) {
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++) {
                mvGrid[c][i][j].reserve(nReserve);
            }
        }
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    // 将关键点的全局序号添加到对应栅格中
    int currPtIdx = 0;

    N = 0;
    for (int c = 0; c < nCamera; ++c) N += mvpKeysCams[c].size();

    mvKeys.reserve(N);
    mvKeysUn.reserve(N);

    for (int c = 0; c < nCamera; ++c) {
        for (int i = 0; i < mvpKeysUnCams[c].size(); ++i) {
            mvKeys.push_back(mvpKeysCams[c][i]);
            mvKeysUn.push_back(mvpKeysUnCams[c][i]);
            mmpKeyToCam[currPtIdx] = c;
            mmpGlobalToLocalID[currPtIdx] = i;
            cv::KeyPoint kp = mvpKeysCams[c][i];

            int nGridPosX, nGridPosY;
            if (PosInGrid(c, kp, nGridPosX, nGridPosY)) {
            mvGrid[c][nGridPosX][nGridPosY].push_back(currPtIdx);
            }
            ++currPtIdx;
    }
    }

    mvpMapPoints.resize(N, static_cast<MapPoint*>(NULL));
    mvbOutlier.resize(N, false);
    mvmProjectPoints.resize(nCamera);
    mmMatchedInImage.clear();

    if (mpPrevFrame) {
        if (mpPrevFrame->HasVelocity())
            SetVelocity(mpPrevFrame->GetVelocity());
    }
    else {
        mTwist << iniVel.cast<float>();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    mvDescriptors.pop_back();
}

// void MultiFrame::AssignFeaturesToGrid(int cam)
// {
//     // Fill matrix with points
//     const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

//     int nReserve = 0.5f*N/(nCells);

//     for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
//         for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
//             mGrid[i][j].reserve(nReserve);
//             if(Nleft != -1){
//                 mGridRight[i][j].reserve(nReserve);
//             }
//         }



//     for(int i=0;i<N;i++)
//     {
//         const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
//                                                  : (i < Nleft) ? mvKeys[i]
//                                                                  : mvKeysRight[i - Nleft];

//         int nGridPosX, nGridPosY;
//         if(PosInGrid(kp,nGridPosX,nGridPosY)){
//             if(Nleft == -1 || i < Nleft)
//                 mGrid[nGridPosX][nGridPosY].push_back(i);
//             else
//                 mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
//         }
//     }
// }

void MultiFrame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0,x1};
    vmono[flag] = (*(mvpORBextractor[flag]))(im, cv::Mat(), mvpKeysCams[flag], mvDescriptors[flag], vLapping);
    // if(flag==0)
    //     monoLeft = (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors,vLapping);
    // else
    //     monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
}

bool MultiFrame::isSet() const {
    return mbIsSet;
}

void MultiFrame::SetPose(const Sophus::SE3<float> &Tbw) {
    mTbw = Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void MultiFrame::SetPose(const Sophus::SE3f& Tbw, const std::vector<Sophus::SE3f>& Twc)
{
    mTbw = Tbw;
    
    UpdatePoseMatrices(Twc);
    mbIsSet = true;
    mbHasPose = true;
}

void MultiFrame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

void MultiFrame::SetVelocity(Eigen::VectorXf Twist)
{
    mTwist = Twist;
    mbHasVelocity = true;
}

Eigen::VectorXf MultiFrame::GetVelocity() const
{
    return mTwist;
}

void MultiFrame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::VectorXf &Vwb)
{
    mTwist = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTbw = mImuCalib.mTcb * Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void MultiFrame::UpdatePoseMatrices()
{
    Sophus::SE3<float> Twb = mTbw.inverse();
    mRwb = Twb.rotationMatrix();
    mOw = Twb.translation();
    mRbw = mTbw.rotationMatrix();
    mtbw = mTbw.translation();

    mTcw[nCamera-1] = mTbc[nCamera-1].inverse() * mTbw;
    mTwc[nCamera-1] = mTcw[nCamera-1].inverse();

    if (mnId == 0) return;
    // gp插值得到frame里每个相机时刻的base位姿
    if (!mpPrevFrame) {
        std::cout << "No previous frame" << std::endl;
        return;
    }
    Sophus::SE3f prevTwb = mpPrevFrame->GetPose().inverse();
    Sophus::SE3f Twb_c;
    double t1 = mpPrevFrame->mTimeStamp, t2 = mTimeStamp, t = 0;
    for (int c = 0; c < nCamera - 1; ++c) {
        t = mvTimeStamps[c];
        Twb_c = mpGP->QueryPose(prevTwb.cast<double>(), Twb.cast<double>(), mpPrevFrame->mTwist.cast<double>(), mTwist.cast<double>(), t1, t2, t).cast<float>();
        mTwc[c] = Twb_c * mTbc[c];
        mTcw[c] = mTwc[c].inverse();
    }
}

void MultiFrame::UpdatePoseMatrices(const std::vector<Sophus::SE3f> &Twc)
{
    mTwc = Twc;
    for (int c = 0; c < nCamera; ++c) {
        mTcw[c] = mTwc[c].inverse();
    }
    Sophus::SE3f Twb = mTbw.inverse();
    mRwb = Twb.rotationMatrix();
    mOw = Twb.translation();
    mRbw = mTbw.rotationMatrix();
    mtbw = mTbw.translation();
}

Eigen::Matrix<float,3,1> MultiFrame::GetImuPosition() const {
    return mRwb * mImuCalib.mTcb.translation() + mOw;
}

Eigen::Matrix<float,3,3> MultiFrame::GetImuRotation() {
    return mRwb * mImuCalib.mTcb.rotationMatrix();
}

Sophus::SE3<float> MultiFrame::GetImuPose() {
    return mTbw.inverse() * mImuCalib.mTcb;
}

Sophus::SE3f MultiFrame::GetRelativePoseTrl()
{
    return mTrl;
}

Sophus::SE3f MultiFrame::GetRelativePoseTlr()
{
    return mTlr;
}

Eigen::Matrix3f MultiFrame::GetRelativePoseTlr_rotation(){
    return mTlr.rotationMatrix();
}

Eigen::Vector3f MultiFrame::GetRelativePoseTlr_translation() {
    return mTlr.translation();
}


bool MultiFrame::isInFrustum(int cam, MapPoint *pMP, float viewingCosLimit)
{
    if(Nleft == -1){
        pMP->mvbTrackInView[cam] = false;
        pMP->mvTrackProjX[cam] = -1;
        pMP->mvTrackProjY[cam] = -1;

        // 3D in absolute coordinates
        Eigen::Matrix<float,3,1> P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const Eigen::Matrix<float,3,1> Pc = mTcw[cam] * P;
        const float Pc_dist = Pc.norm();

        // Check positive depth
        const float &PcZ = Pc(2);
        const float invz = 1.0f/PcZ;
        if(PcZ<0.0f)
            return false;

        const Eigen::Vector2f uv = mvpCamera[cam]->project(Pc);

        if(uv(0)<mvMinX[cam] || uv(0)>mvMaxX[cam])
            return false;
        if(uv(1)<mvMinY[cam] || uv(1)>mvMaxY[cam])
            return false;

        pMP->mvTrackProjX[cam] = uv(0);
        pMP->mvTrackProjY[cam] = uv(1);

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const Eigen::Vector3f PO = P - mOw;
        const float dist = PO.norm();

        if(dist<minDistance || dist>maxDistance)
            return false;

        // Check viewing angle
        Eigen::Vector3f Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn)/dist;

        if(viewCos<viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // Data used by the tracking
        pMP->mvbTrackInView[cam] = true;
        pMP->mvTrackProjX[cam] = uv(0);
        pMP->mvTrackProjXR[cam] = uv(0) - mbf*invz;

        pMP->mvTrackDepth[cam] = Pc_dist;

        pMP->mvTrackProjY[cam] = uv(1);
        pMP->mvnTrackScaleLevel[cam] = nPredictedLevel;
        pMP->mvTrackViewCos[cam] = viewCos;

        return true;
    }
    // else{
    //     pMP->mbTrackInView = false;
    //     pMP->mbTrackInViewR = false;
    //     pMP -> mnTrackScaleLevel = -1;
    //     pMP -> mnTrackScaleLevelR = -1;

    //     pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
    //     pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);

    //     return pMP->mbTrackInView || pMP->mbTrackInViewR;
    // }
}

bool MultiFrame::ProjectPointDistort(int cam, MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = mTcw[cam] * P;
    const float &PcX = Pc(0);
    const float &PcY= Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=mvfx[cam]*PcX*invz+mvcx[cam];
    v=mvfy[cam]*PcY*invz+mvcy[cam];

    if(u<mvMinX[cam] || u>mvMaxX[cam])
        return false;
    if(v<mvMinY[cam] || v>mvMaxY[cam])
        return false;

    float u_distort, v_distort;

    float x = (u - mvcx[cam]) * minvfx[cam];
    float y = (v - mvcy[cam]) * minvfy[cam];
    float r2 = x * x + y * y;
    float k1 = mvDistCoef[cam].at<float>(0);
    float k2 = mvDistCoef[cam].at<float>(1);
    float p1 = mvDistCoef[cam].at<float>(2);
    float p2 = mvDistCoef[cam].at<float>(3);
    float k3 = 0;
    if(mvDistCoef[cam].total() == 5)
    {
        k3 = mvDistCoef[cam].at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * mvfx[cam] + mvcx[cam];
    v_distort = y_distort * mvfy[cam] + mvcy[cam];


    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

Eigen::Vector3f MultiFrame::inRefCoordinates(int cam, Eigen::Vector3f pCw)
{
    return mTcw[cam] * pCw;
}

vector<size_t> MultiFrame::GetFeaturesInArea(int cam, 
                        const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mvMinX[cam]-factorX)*mvGridElementWidthInv[cam]));
    if(nMinCellX>=FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mvMinX[cam]+factorX)*mvGridElementWidthInv[cam]));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mvMinY[cam]-factorY)*mvGridElementHeightInv[cam]));
    if(nMinCellY>=FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mvMinY[cam]+factorY)*mvGridElementHeightInv[cam]));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mvGrid[cam][ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool MultiFrame::PosInGrid(int cam, const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mvMinX[cam])*mvGridElementWidthInv[cam]);
    posY = round((kp.pt.y-mvMinY[cam])*mvGridElementHeightInv[cam]);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void MultiFrame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mvDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void MultiFrame::UndistortKeyPoints(int cam)
{
    if (cam == nCamera) return;
    if(mvDistCoef[cam].at<float>(0)==0.0)
    {
        mvpKeysUnCams[cam]=mvpKeysCams[cam];
        return;
    }

    int num = mvpKeysCams[cam].size();
    // Fill matrix with points
    cv::Mat mat(num,2,CV_32F);

    for(int i=0; i<num; i++)
    {
        mat.at<float>(i,0)=mvpKeysCams[cam][i].pt.x;
        mat.at<float>(i,1)=mvpKeysCams[cam][i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mvpCamera[cam])->toK(),mvDistCoef[cam],cv::Mat(),mvK[cam]);
    mat=mat.reshape(1);


    // Fill undistorted keypoint vector
    mvpKeysUnCams[cam].resize(num);
    for(int i=0; i<num; i++)
    {
        cv::KeyPoint kp = mvpKeysCams[cam][i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvpKeysUnCams[cam][i]=kp;
    }

}

void MultiFrame::ComputeImageBounds(const cv::Mat &imLeft, int camID)
{
    if(mvDistCoef[camID].at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mvpCamera[camID])->toK(),mvDistCoef[camID], cv::Mat(), mvK[camID]);
        mat=mat.reshape(1);

        // Undistort corners
        mvMinX[camID] = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mvMaxX[camID] = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mvMinY[camID] = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mvMaxY[camID] = max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        mvMinX[camID] = 0.0f;
        mvMaxX[camID] = imLeft.cols;
        mvMinY[camID] = 0.0f;
        mvMaxY[camID] = imLeft.rows;
    }
}

void MultiFrame::ComputeStereoMatches()
{
    int num = mvpKeysCams[nCamera-1].size();
    mvuRight = vector<float>(num,-1.0f);
    mvDepth = vector<float>(num,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mvpORBextractor[nCamera-1]->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int numr = mvpKeysCams[nCamera].size();

    for(int iR=0; iR<numr; iR++)
    {
        const cv::KeyPoint &kp = mvpKeysCams[nCamera][iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvpKeysCams[nCamera][iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(num);

    for(int iL=0; iL<num; iL++)
    {
        const cv::KeyPoint &kpL = mvpKeysCams[nCamera-1][iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mvDescriptors[nCamera-1].row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvpKeysCams[nCamera][iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mvDescriptors[nCamera].row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvpKeysCams[nCamera][bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mvpORBextractor[nCamera-1]->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mvpORBextractor[nCamera]->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mvpORBextractor[nCamera]->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void MultiFrame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

bool MultiFrame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[mmpGlobalToLocalID[i]];
    if(z>0) {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-mvcx[nCamera-1])*z*minvfx[nCamera-1];
        const float y = (v-mvcy[nCamera-1])*z*minvfy[nCamera-1];
        Eigen::Vector3f x3Dc(x, y, z);
        x3D = mTwc[nCamera-1] * x3Dc;
        return true;
    } else
        return false;
}

bool MultiFrame::imuIsPreintegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

void MultiFrame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

// void MultiFrame::ComputeStereoFishEyeMatches() {
//     //Speed it up by matching keypoints in the lapping area
//     vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
//     vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

//     cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
//     cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

//     mvLeftToRightMatch = vector<int>(Nleft,-1);
//     mvRightToLeftMatch = vector<int>(Nright,-1);
//     mvDepth = vector<float>(Nleft,-1.0f);
//     mvuRight = vector<float>(Nleft,-1);
//     mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
//     mnCloseMPs = 0;

//     //Perform a brute force between Keypoint in the left and right image
//     vector<vector<cv::DMatch>> matches;

//     BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

//     int nMatches = 0;
//     int descMatches = 0;

//     //Check matches using Lowe's ratio
//     for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it){
//         if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7){
//             //For every good match, check parallax and reprojection error to discard spurious matches
//             Eigen::Vector3f p3D;
//             descMatches++;
//             float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
//             float depth = static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2,mvKeys[(*it)[0].queryIdx + monoLeft],mvKeysRight[(*it)[0].trainIdx + monoRight],mRlr,mtlr,sigma1,sigma2,p3D);
//             if(depth > 0.0001f){
//                 mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
//                 mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
//                 mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D;
//                 mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
//                 nMatches++;
//             }
//         }
//     }
// }

// bool MultiFrame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
//     // 3D in absolute coordinates
//     Eigen::Vector3f P = pMP->GetWorldPos();

//     Eigen::Matrix3f mR;
//     Eigen::Vector3f mt, twc;
//     if(bRight){
//         Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
//         Eigen::Vector3f trl = mTrl.translation();
//         mR = Rrl * mRbw;
//         mt = Rrl * mtbw + trl;
//         twc = mRwb * mTlr.translation() + mOw;
//     }
//     else{
//         mR = mRbw;
//         mt = mtbw;
//         twc = mOw;
//     }

//     // 3D in camera coordinates
//     Eigen::Vector3f Pc = (mR * P + mt);
//     const float Pc_dist = Pc.norm();
//     const float &PcZ = Pc(2);

//     // Check positive depth
//     if(PcZ<0.0f)
//         return false;

//     // Project in image and check it is not outside
//     Eigen::Vector2f uv;
//     if(bRight) uv = mpCamera2->project(Pc);
//     else uv = mpCamera->project(Pc);

//     if(uv(0)<mnMinX || uv(0)>mnMaxX)
//         return false;
//     if(uv(1)<mnMinY || uv(1)>mnMaxY)
//         return false;

//     // Check distance is in the scale invariance region of the MapPoint
//     const float maxDistance = pMP->GetMaxDistanceInvariance();
//     const float minDistance = pMP->GetMinDistanceInvariance();
//     const Eigen::Vector3f PO = P - twc;
//     const float dist = PO.norm();

//     if(dist<minDistance || dist>maxDistance)
//         return false;

//     // Check viewing angle
//     Eigen::Vector3f Pn = pMP->GetNormal();

//     const float viewCos = PO.dot(Pn) / dist;

//     if(viewCos<viewingCosLimit)
//         return false;

//     // Predict scale in the image
//     const int nPredictedLevel = pMP->PredictScale(dist,this);

//     if(bRight){
//         pMP->mTrackProjXR = uv(0);
//         pMP->mTrackProjYR = uv(1);
//         pMP->mnTrackScaleLevelR= nPredictedLevel;
//         pMP->mTrackViewCosR = viewCos;
//         pMP->mTrackDepthR = Pc_dist;
//     }
//     else{
//         pMP->mTrackProjX = uv(0);
//         pMP->mTrackProjY = uv(1);
//         pMP->mnTrackScaleLevel= nPredictedLevel;
//         pMP->mTrackViewCos = viewCos;
//         pMP->mTrackDepth = Pc_dist;
//     }

//     return true;
// }

Eigen::Vector3f MultiFrame::UnprojectStereoFishEye(const int &i){
    return mRwb * mvStereo3Dpoints[i] + mOw;
}

} //namespace ORB_SLAM
