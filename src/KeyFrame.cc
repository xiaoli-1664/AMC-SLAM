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

#include "KeyFrame.h"
#include "Converter.h"
#include "ImuTypes.h"
#include<mutex>

namespace ORB_SLAM3
{

long unsigned int MultiKeyFrame::nNextId=0;
std::vector<Sophus::SE3f> MultiKeyFrame::mTbc;

MultiKeyFrame::MultiKeyFrame():
        mnFrameId(0),  mTimeStamp(0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
        mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnBALocalForMerge(0),
        mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnMergeQuery(0), mnMergeWords(0), mnBAGlobalForKF(0),
        mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0),
        mbf(0), mb(0), mThDepth(0), N(0), mvKeys(), mvKeysUn(),
        mvuRight(), mvDepth(), mnScaleLevels(0), mfScaleFactor(0),
        mfLogScaleFactor(0), mvScaleFactors(), mvLevelSigma2(), mvInvLevelSigma2(),
        mPrevKF(static_cast<MultiKeyFrame*>(NULL)), mNextKF(static_cast<MultiKeyFrame*>(NULL)), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
        mbToBeErased(false), mbBad(false), mHalfBaseline(0), mbCurrentPlaceRecognition(false), mnMergeCorrectedForKF(0),
        NLeft(0),NRight(0), mnNumberOfOpt(0), mbHasVelocity(false)
{

}

MultiKeyFrame::MultiKeyFrame(MultiFrame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    bImu(pMap->isImuInitialized()), nCamera(F.nCamera), mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mvTimeStamps(F.mvTimeStamps), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mvfGridElementWidthInv(F.mvGridElementWidthInv), mvfGridElementHeightInv(F.mvGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnBALocalForMerge(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0), mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0),
    mvfx(F.mvfx), mvfy(F.mvfy), mvcx(F.mvcx), mvcy(F.mvcy), minvfx(F.minvfx), minvfy(F.minvfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn), mmpKeyToCam(F.mmpKeyToCam), mmpGlobalToLocalID(F.mmpGlobalToLocalID),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth),
    mBowVec(F.mBowVec), mBowVecs(F.mvBowVecs), mFeatVec(F.mFeatVec), mFeatVecs(F.mvFeatVecs), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mvMinX(F.mvMinX), mvMinY(F.mvMinY), mvMaxX(F.mvMaxX),
    mvMaxY(F.mvMaxY), mvK_(F.mvK_), mPrevKF(NULL), mNextKF(NULL), mpImuPreintegrated(F.mpImuPreintegrated),
    mImuCalib(F.mImuCalib), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false), mnDataset(F.mnDataset),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap), mbCurrentPlaceRecognition(false), mNameFile(F.mNameFile), mnMergeCorrectedForKF(0),
    mvpCamera(F.mvpCamera),
    mvLeftToRightMatch(F.mvLeftToRightMatch),mvRightToLeftMatch(F.mvRightToLeftMatch), mTlr(F.GetRelativePoseTlr()),
    NLeft(F.Nleft), NRight(F.Nright), mTrl(F.GetRelativePoseTrl()), mnNumberOfOpt(0), mbHasVelocity(false)
{
    if (nNextId == 0) mTbc = MultiFrame::mTbc;
    mnId=nNextId++;

    mpGP = F.mpGP;
    mvDescriptors.resize(nCamera);
    mvGrid.resize(nCamera);
    mvDistCoef.resize(nCamera);
    mTwc.resize(nCamera);
    mTcw.resize(nCamera);
#pragma omp parallel for num_threads(nCamera)
    for (int c = 0; c < nCamera; ++c) {
        mvDescriptors[c] = F.mvDescriptors[c].clone();
        mvDistCoef[c] = F.mvDistCoef[c].clone();
        mvGrid[c].resize(mnGridCols);
        for (int i = 0; i < mnGridCols; ++i) {
            mvGrid[c][i].resize(mnGridRows);
            for (int j = 0; j < mnGridRows; ++j) {
                mvGrid[c][i][j] = F.mvGrid[c][i][j];
            }
        }
    }


    if(!F.HasVelocity()) {
        mVw.setZero();
        mbHasVelocity = false;
    }
    else
    {
        mVw = F.GetVelocity();
        mbHasVelocity = true;
    }

    mImuBias = F.mImuBias;
    SetPose(F.GetPose(), F.mTwc);

    mnOriginMapId = pMap->GetId();
}

void MultiKeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mvDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
    mBowVecs.resize(nCamera);
    mFeatVecs.resize(nCamera);
}

void MultiKeyFrame::SetPose(const Sophus::SE3f &Tbw)
{
    unique_lock<mutex> lock(mMutexPose);

    mTbw = Tbw;
    mRbw = mTbw.rotationMatrix();
    mTwb = mTbw.inverse();
    mRwb = mTwb.rotationMatrix();

    mTcw[nCamera-1] = mTbc[nCamera-1].inverse() * mTbw;
    mTwc[nCamera-1] = mTcw[nCamera-1].inverse();

    if (mnId == 0) return;
    if (!mPrevKF) {
        std::cerr << "No previous frame" << std::endl;
        return;
    }
    Sophus::SE3f prevTwb = mPrevKF->GetPose().inverse(), Twb_c;
    double t1 = mPrevKF->mTimeStamp, t2 = mTimeStamp, t = 0;
    for (int c = 0; c < nCamera - 1; ++c) {
        t = mvTimeStamps[c];
        Twb_c = mpGP->QueryPose(prevTwb.cast<double>(), mTwb.cast<double>(), mPrevKF->GetVelocity().cast<double>(), mVw.cast<double>(), t1, t2, t).cast<float>();
        mTwc[c] = Twb_c * mTbc[c];
        mTcw[c] = mTwc[c].inverse();
    }
        // if (mImuCalib.mbIsSet) // TODO Use a flag instead of the OpenCV matrix
    // {
    //     mOwb = mRwb * mImuCalib.mTcb.translation() + mTwb.translation();
    // }
}

void MultiKeyFrame::SetPose(const Sophus::SE3f &Tbw, const std::vector<Sophus::SE3f> &Twc) {
    unique_lock<mutex> lock(mMutexPose);

    mTbw = Tbw;
    mRbw = mTbw.rotationMatrix();
    mTwb = mTbw.inverse();
    mRwb = mTwb.rotationMatrix();

    for (int c = 0; c < nCamera; ++c) {
        mTwc[c] = Twc[c];
        mTcw[c] = Twc[c].inverse();
    }
}

void MultiKeyFrame::SetVelocity(const Eigen::VectorXf &Vw)
{
    unique_lock<mutex> lock(mMutexPose);
    mVw = Vw;
    mbHasVelocity = true;
}

Sophus::SE3f MultiKeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTbw;
}

Sophus::SE3f MultiKeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwb;
}

Eigen::Vector3f MultiKeyFrame::GetFrameCenter(){
    unique_lock<mutex> lock(mMutexPose);
    return mTwb.translation();
}

Sophus::SE3f MultiKeyFrame::GetCameraPose(int cam)
{
    unique_lock<mutex> lock(mMutexPose);
    return mTcw[cam];
}

Sophus::SE3f MultiKeyFrame::GetCameraPoseInverse(int cam)
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwc[cam];
}

std::vector<Sophus::SE3f> MultiKeyFrame::GetCameraPoses()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTcw;
}

Eigen::Vector3f MultiKeyFrame::GetCameraCenter(int cam)
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwc[cam].translation();
}

Eigen::Vector3f MultiKeyFrame::GetImuPosition()
{
    unique_lock<mutex> lock(mMutexPose);
    return mOwb;
}

Eigen::Matrix3f MultiKeyFrame::GetImuRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return (mTwb * mImuCalib.mTcb).rotationMatrix();
}

Sophus::SE3f MultiKeyFrame::GetImuPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwb * mImuCalib.mTcb;
}

Eigen::Matrix3f MultiKeyFrame::GetRotation(){
    unique_lock<mutex> lock(mMutexPose);
    return mRbw;
}

Eigen::Vector3f MultiKeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTbw.translation();
}

Eigen::VectorXf MultiKeyFrame::GetVelocity()
{
    unique_lock<mutex> lock(mMutexPose);
    return mVw;
}

bool MultiKeyFrame::isVelocitySet()
{
    unique_lock<mutex> lock(mMutexPose);
    return mbHasVelocity;
}

void MultiKeyFrame::AddConnection(MultiKeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void MultiKeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,MultiKeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<MultiKeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        vPairs.push_back(make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end());
    list<MultiKeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        if(!vPairs[i].second->isBad())
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }
    }

    mvpOrderedConnectedKeyFrames = vector<MultiKeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

set<MultiKeyFrame*> MultiKeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<MultiKeyFrame*> s;
    for(map<MultiKeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

vector<MultiKeyFrame*> MultiKeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<MultiKeyFrame*> MultiKeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<MultiKeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

vector<MultiKeyFrame*> MultiKeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
    {
        return vector<MultiKeyFrame*>();
    }

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,MultiKeyFrame::weightComp);

    if(it==mvOrderedWeights.end() && mvOrderedWeights.back() < w)
    {
        return vector<MultiKeyFrame*>();
    }
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<MultiKeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int MultiKeyFrame::GetWeight(MultiKeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

int MultiKeyFrame::GetNumberMPs()
{
    unique_lock<mutex> lock(mMutexFeatures);
    int numberMPs = 0;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        numberMPs++;
    }
    return numberMPs;
}

void MultiKeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void MultiKeyFrame::EraseMapPointMatch(const int &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

// TODO: 要不要锁?
void MultiKeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    unique_lock<mutex> lock(mMutexFeatures);
    vector<int> indexes = pMP->GetIndexInKeyFrame(this);
    for (int c = 0; c < nCamera; ++c) {
        if(indexes[c] != -1)
            mvpMapPoints[indexes[c]] = static_cast<MapPoint*>(NULL);
    }
    // size_t leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    // if(leftIndex != -1)
    //     mvpMapPoints[leftIndex]=static_cast<MapPoint*>(NULL);
    // if(rightIndex != -1)
    //     mvpMapPoints[rightIndex]=static_cast<MapPoint*>(NULL);
}

void MultiKeyFrame::EraseMapPointMatch(MapPoint* pMP, int cam)
{
    unique_lock<mutex> lock(mMutexFeatures);
    int index = pMP->GetIndexInKeyFrame(this)[cam];
    if(index != -1)
        mvpMapPoints[index] = static_cast<MapPoint*>(NULL);
}


void MultiKeyFrame::ReplaceMapPointMatch(const int &idx, MapPoint* pMP)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> MultiKeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int MultiKeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

vector<MapPoint*> MultiKeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* MultiKeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void MultiKeyFrame::UpdateConnections(bool upParent)
{
    map<MultiKeyFrame*,int> KFcounter;

    vector<MapPoint*> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        map<MultiKeyFrame*, vector<int>> observations = pMP->GetObservations();

        for(map<MultiKeyFrame*, vector<int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId || mit->first->isBad() || mit->first->GetMap() != mpMap)
                continue;
            KFcounter[mit->first]++;

        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    MultiKeyFrame* pKFmax=NULL;
    int th = 15;

    vector<pair<int,MultiKeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    if(!upParent)
        cout << "UPDATE_CONN: current KF " << mnId << endl;
    for(map<MultiKeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(!upParent)
            cout << "  UPDATE_CONN: KF " << mit->first->mnId << " ; num matches: " << mit->second << endl;
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th)
        {
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second);
        }
    }

    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    sort(vPairs.begin(),vPairs.end());
    list<MultiKeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<MultiKeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());


        if(mbFirstConnection && mnId!=mpMap->GetInitKFid())
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

void MultiKeyFrame::AddChild(MultiKeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void MultiKeyFrame::EraseChild(MultiKeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void MultiKeyFrame::ChangeParent(MultiKeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    if(pKF == this)
    {
        cout << "ERROR: Change parent KF, the parent and child are the same KF" << endl;
        throw std::invalid_argument("The parent and child can not be the same");
    }

    mpParent = pKF;
    pKF->AddChild(this);
}

set<MultiKeyFrame*> MultiKeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

MultiKeyFrame* MultiKeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool MultiKeyFrame::hasChild(MultiKeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void MultiKeyFrame::SetFirstConnection(bool bFirst)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbFirstConnection=bFirst;
}

void MultiKeyFrame::AddLoopEdge(MultiKeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<MultiKeyFrame*> MultiKeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void MultiKeyFrame::AddMergeEdge(MultiKeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspMergeEdges.insert(pKF);
}

set<MultiKeyFrame*> MultiKeyFrame::GetMergeEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspMergeEdges;
}

void MultiKeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void MultiKeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

void MultiKeyFrame::SetBadFlag()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==mpMap->GetInitKFid())
        {
            return;
        }
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for(map<MultiKeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
    {
        mit->first->EraseConnection(this);
    }

    for(size_t i=0; i<mvpMapPoints.size(); i++)
    {
        if(mvpMapPoints[i])
        {
            mvpMapPoints[i]->EraseObservation(this);
        }
    }

    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<MultiKeyFrame*> sParentCandidates;
        if(mpParent)
            sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            MultiKeyFrame* pC;
            MultiKeyFrame* pP;

            for(set<MultiKeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                MultiKeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<MultiKeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<MultiKeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if(!mspChildrens.empty())
        {
            for(set<MultiKeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }
        }

        if(mpParent){
            mpParent->EraseChild(this);
            mTbp = mTbw * mpParent->GetPoseInverse();
        }
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool MultiKeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void MultiKeyFrame::EraseConnection(MultiKeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}


vector<size_t> MultiKeyFrame::GetFeaturesInArea(int cam, const float &x, const float &y, const float &r, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mvMinX[cam]-factorX)*mvfGridElementWidthInv[cam]));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mvMinX[cam]+factorX)*mvfGridElementWidthInv[cam]));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mvMinY[cam]-factorY)*mvfGridElementHeightInv[cam]));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mvMinY[cam]+factorY)*mvfGridElementHeightInv[cam]));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mvGrid[cam][ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool MultiKeyFrame::IsInImage(int cam, const float &x, const float &y) const
{
    return (x>=mvMinX[cam] && x<mvMaxX[cam] && y>=mvMinY[cam] && y<mvMaxY[cam]);
}

bool MultiKeyFrame::UnprojectStereo(int i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[mmpGlobalToLocalID[i]];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-mvcx[nCamera-1])*z*minvfx[nCamera-1];
        const float y = (v-mvcy[nCamera-1])*z*minvfy[nCamera-1];
        Eigen::Vector3f x3Dc(x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        x3D = mRwb * x3Dc + mTwb.translation();
        return true;
    }
    else
        return false;
}

float MultiKeyFrame::ComputeSceneMedianDepth(const int q)
{
    if(N==0)
        return -1.0;

    vector<MapPoint*> vpMapPoints;
    std::vector<Sophus::SE3f> Tcw;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw = mTcw;
    }

    int z = 0, cam;
    Eigen::Vector3f x3Dc;
    vector<float> vDepths;
    vDepths.reserve(N);
    for(int i=0; i<N; i++) {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            Eigen::Vector3f x3Dw = pMP->GetWorldPos();
            cam = mmpKeyToCam[i];
            x3Dc = Tcw[cam] * x3Dw;
            z = x3Dc.z();
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

void MultiKeyFrame::SetNewBias(const IMU::Bias &b)
{
    unique_lock<mutex> lock(mMutexPose);
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

Eigen::Vector3f MultiKeyFrame::GetGyroBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return Eigen::Vector3f(mImuBias.bwx, mImuBias.bwy, mImuBias.bwz);
}

Eigen::Vector3f MultiKeyFrame::GetAccBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return Eigen::Vector3f(mImuBias.bax, mImuBias.bay, mImuBias.baz);
}

IMU::Bias MultiKeyFrame::GetImuBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return mImuBias;
}

Map* MultiKeyFrame::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void MultiKeyFrame::UpdateMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void MultiKeyFrame::PreSave(set<MultiKeyFrame*>& spKF,set<MapPoint*>& spMP, set<GeometricCamera*>& spCam)
{
    // Save the id of each MapPoint in this KF, there can be null pointer in the vector
    mvBackupMapPointsId.clear();
    mvBackupMapPointsId.reserve(N);
    for(int i = 0; i < N; ++i)
    {

        if(mvpMapPoints[i] && spMP.find(mvpMapPoints[i]) != spMP.end()) // Checks if the element is not null
            mvBackupMapPointsId.push_back(mvpMapPoints[i]->mnId);
        else // If the element is null his value is -1 because all the id are positives
            mvBackupMapPointsId.push_back(-1);
    }
    // Save the id of each connected KF with it weight
    mBackupConnectedKeyFrameIdWeights.clear();
    for(std::map<MultiKeyFrame*,int>::const_iterator it = mConnectedKeyFrameWeights.begin(), end = mConnectedKeyFrameWeights.end(); it != end; ++it)
    {
        if(spKF.find(it->first) != spKF.end())
            mBackupConnectedKeyFrameIdWeights[it->first->mnId] = it->second;
    }

    // Save the parent id
    mBackupParentId = -1;
    if(mpParent && spKF.find(mpParent) != spKF.end())
        mBackupParentId = mpParent->mnId;

    // Save the id of the childrens KF
    mvBackupChildrensId.clear();
    mvBackupChildrensId.reserve(mspChildrens.size());
    for(MultiKeyFrame* pKFi : mspChildrens)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupChildrensId.push_back(pKFi->mnId);
    }

    // Save the id of the loop edge KF
    mvBackupLoopEdgesId.clear();
    mvBackupLoopEdgesId.reserve(mspLoopEdges.size());
    for(MultiKeyFrame* pKFi : mspLoopEdges)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupLoopEdgesId.push_back(pKFi->mnId);
    }

    // Save the id of the merge edge KF
    mvBackupMergeEdgesId.clear();
    mvBackupMergeEdgesId.reserve(mspMergeEdges.size());
    for(MultiKeyFrame* pKFi : mspMergeEdges)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupMergeEdgesId.push_back(pKFi->mnId);
    }

    //Camera data
    // TODO:存储多相机数据
    // mnBackupIdCamera = -1;
    // if(mvpCamera && spCam.find(mpCamera) != spCam.end())
    //     mnBackupIdCamera = mpCamera->GetId();

    // mnBackupIdCamera2 = -1;
    // if(mpCamera2 && spCam.find(mpCamera2) != spCam.end())
    //     mnBackupIdCamera2 = mpCamera2->GetId();

    //Inertial data
    mBackupPrevKFId = -1;
    if(mPrevKF && spKF.find(mPrevKF) != spKF.end())
        mBackupPrevKFId = mPrevKF->mnId;

    mBackupNextKFId = -1;
    if(mNextKF && spKF.find(mNextKF) != spKF.end())
        mBackupNextKFId = mNextKF->mnId;

    if(mpImuPreintegrated)
        mBackupImuPreintegrated.CopyFrom(mpImuPreintegrated);
}

void MultiKeyFrame::PostLoad(map<long unsigned int, MultiKeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid, map<unsigned int, GeometricCamera*>& mpCamId){
    // Rebuild the empty variables

    // Pose
    SetPose(mTbw);

    mTrl = mTlr.inverse();

    // Reference reconstruction
    // Each MapPoint sight from this KeyFrame
    mvpMapPoints.clear();
    mvpMapPoints.resize(N);
    for(int i=0; i<N; ++i)
    {
        if(mvBackupMapPointsId[i] != -1)
            mvpMapPoints[i] = mpMPid[mvBackupMapPointsId[i]];
        else
            mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
    }

    // Conected KeyFrames with him weight
    mConnectedKeyFrameWeights.clear();
    for(map<long unsigned int, int>::const_iterator it = mBackupConnectedKeyFrameIdWeights.begin(), end = mBackupConnectedKeyFrameIdWeights.end();
        it != end; ++it)
    {
        MultiKeyFrame* pKFi = mpKFid[it->first];
        mConnectedKeyFrameWeights[pKFi] = it->second;
    }

    // Restore parent KeyFrame
    if(mBackupParentId>=0)
        mpParent = mpKFid[mBackupParentId];

    // KeyFrame childrens
    mspChildrens.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupChildrensId.begin(), end = mvBackupChildrensId.end(); it!=end; ++it)
    {
        mspChildrens.insert(mpKFid[*it]);
    }

    // Loop edge KeyFrame
    mspLoopEdges.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupLoopEdgesId.begin(), end = mvBackupLoopEdgesId.end(); it != end; ++it)
    {
        mspLoopEdges.insert(mpKFid[*it]);
    }

    // Merge edge KeyFrame
    mspMergeEdges.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupMergeEdgesId.begin(), end = mvBackupMergeEdgesId.end(); it != end; ++it)
    {
        mspMergeEdges.insert(mpKFid[*it]);
    }

    //Camera data
    // TODO:加载多相机数据
    // if(mnBackupIdCamera >= 0)
    // {
    //     mpCamera = mpCamId[mnBackupIdCamera];
    // }
    // else
    // {
    //     cout << "ERROR: There is not a main camera in KF " << mnId << endl;
    // }
    // if(mnBackupIdCamera2 >= 0)
    // {
    //     mpCamera2 = mpCamId[mnBackupIdCamera2];
    // }

    //Inertial data
    if(mBackupPrevKFId != -1)
    {
        mPrevKF = mpKFid[mBackupPrevKFId];
    }
    if(mBackupNextKFId != -1)
    {
        mNextKF = mpKFid[mBackupNextKFId];
    }
    mpImuPreintegrated = &mBackupImuPreintegrated;


    // Remove all backup container
    mvBackupMapPointsId.clear();
    mBackupConnectedKeyFrameIdWeights.clear();
    mvBackupChildrensId.clear();
    mvBackupLoopEdgesId.clear();

    UpdateBestCovisibles();
}

// TODO:没有用的函数注释掉？
bool MultiKeyFrame::ProjectPointDistort(int cam, MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    // Eigen::Vector3f Pc = mTbc[cam].inverse() * (mRbw * P + mTbw.translation());
    // TODO:加入线程锁取mTcw
    Eigen::Vector3f Pc = mTcw[cam] * P;
    float &PcX = Pc(0);
    float &PcY = Pc(1);
    float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    float invz = 1.0f/PcZ;
    u=mvfx[cam]*PcX*invz+mvcx[cam];
    v=mvfy[cam]*PcY*invz+mvcy[cam];

    // cout << "c";

    if(u<mvMinX[cam] || u>mvMaxX[cam])
        return false;
    if(v<mvMinY[cam] || v>mvMaxY[cam])
        return false;

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

    float u_distort = x_distort * mvfx[cam] + mvcx[cam];
    float v_distort = y_distort * mvfy[cam] + mvcy[cam];

    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

bool MultiKeyFrame::ProjectPointUnDistort(int cam, MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    // Eigen::Vector3f Pc = mTbc[cam].inverse() * (mRbw * P + mTbw.translation());
    // TODO:加入线程锁取mTcw
    Eigen::Vector3f Pc = mTcw[cam] * P;
    float &PcX = Pc(0);
    float &PcY= Pc(1);
    float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u = mvfx[cam] * PcX * invz + mvcx[cam];
    v = mvfy[cam] * PcY * invz + mvcy[cam];

    if(u<mvMinX[cam] || u>mvMaxX[cam])
        return false;
    if(v<mvMinY[cam] || v>mvMaxY[cam])
        return false;

    kp = cv::Point2f(u, v);

    return true;
}

Sophus::SE3f MultiKeyFrame::GetRelativePoseTrl()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTrl;
}

Sophus::SE3f MultiKeyFrame::GetRelativePoseTlr()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTlr;
}

Sophus::SE3<float> MultiKeyFrame::GetRightPose() {
    unique_lock<mutex> lock(mMutexPose);

    return mTrl * mTbw;
}

Sophus::SE3<float> MultiKeyFrame::GetRightPoseInverse() {
    unique_lock<mutex> lock(mMutexPose);

    return mTwb * mTlr;
}

Eigen::Vector3f MultiKeyFrame::GetRightCameraCenter() {
    unique_lock<mutex> lock(mMutexPose);

    return (mTwb * mTlr).translation();
}

Eigen::Matrix<float,3,3> MultiKeyFrame::GetRightRotation() {
    unique_lock<mutex> lock(mMutexPose);

    return (mTrl.so3() * mTbw.so3()).matrix();
}

Eigen::Vector3f MultiKeyFrame::GetRightTranslation() {
    unique_lock<mutex> lock(mMutexPose);
    return (mTrl * mTbw).translation();
}

void MultiKeyFrame::SetORBVocabulary(ORBVocabulary* pORBVoc)
{
    mpORBvocabulary = pORBVoc;
}

void MultiKeyFrame::SetKeyFrameDatabase(KeyFrameDatabase* pKFDB)
{
    mpKeyFrameDB = pKFDB;
}

} //namespace ORB_SLAM
