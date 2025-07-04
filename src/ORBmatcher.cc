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


#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM3
{

    const int ORBmatcher::TH_HIGH = 100;
    const int ORBmatcher::TH_LOW = 50;
    const int ORBmatcher::HISTO_LENGTH = 30;

    ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
    {
    }

    int ORBmatcher::SearchByProjection(MultiFrame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
    {
        int nmatches=0, left = 0, right = 0;
        int nmatches_mono = 0;

        const bool bFactor = th!=1.0;

        for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
        {
            MapPoint* pMP = vpMapPoints[iMP];
            // TODO: 并行化
            for (int c = 0; c < F.nCamera; ++c) {
                if(!pMP->mvbTrackInView[c] && !pMP->mvbTrackInViewR[c])
                    continue;

                if(bFarPoints && pMP->mvTrackDepth[c]>thFarPoints)
                    continue;

                if(pMP->isBad())
                    continue;

                if(pMP->mvbTrackInView[c])
                {
                    const int &nPredictedLevel = pMP->mvnTrackScaleLevel[c];

                    // The size of the window will depend on the viewing direction
                    float r = RadiusByViewingCos(pMP->mvTrackViewCos[c]);

                    if(bFactor)
                        r*=th;

                    const vector<size_t> vIndices =
                            F.GetFeaturesInArea(c, pMP->mvTrackProjX[c],pMP->mvTrackProjY[c],r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

                    if(!vIndices.empty()){
                        const cv::Mat MPdescriptor = pMP->GetDescriptor();

                        int bestDist=256;
                        int bestLevel= -1;
                        int bestDist2=256;
                        int bestLevel2 = -1;
                        int bestIdx =-1 ;

                        // Get best and second matches with near keypoints
                        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                        {
                            const size_t idx = *vit;

                            if(F.mvpMapPoints[idx])
                                if(F.mvpMapPoints[idx]->Observations()>0)
                                    continue;

                            if(F.Nleft == -1 && c == F.nCamera-1 && F.mvuRight[F.mmpGlobalToLocalID[idx]]>0)
                            {
                                const float er = fabs(pMP->mvTrackProjXR[c]-F.mvuRight[F.mmpGlobalToLocalID[idx]]);
                                if(er>r*F.mvScaleFactors[nPredictedLevel])
                                    continue;
                            }

                            const cv::Mat &d = F.mvDescriptors[c].row(F.mmpGlobalToLocalID[idx]);

                            const int dist = DescriptorDistance(MPdescriptor,d);

                            if(dist<bestDist)
                            {
                                bestDist2=bestDist;
                                bestDist=dist;
                                bestLevel2 = bestLevel;
                                bestLevel = F.mvKeysUn[idx].octave;
                                bestIdx=idx;
                            }
                            else if(dist<bestDist2)
                            {
                                bestLevel2 = F.mvKeysUn[idx].octave;
                                bestDist2=dist;
                            }
                        }

                        // Apply ratio to second match (only if best and second are in the same scale level)
                        if(bestDist<=TH_HIGH)
                        {
                            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                                continue;

                            if(bestLevel!=bestLevel2 || bestDist<=mfNNratio*bestDist2){
                                F.mvpMapPoints[bestIdx]=pMP;

                                // if(F.Nleft != -1 && F.mvLeftToRightMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                                //     F.mvpMapPoints[F.mvLeftToRightMatch[bestIdx] + F.Nleft] = pMP;
                                //     nmatches++;
                                //     right++;
                                // }

                                nmatches++;
                                left++;
                                if (c < F.nCamera - 1) {
                                    nmatches_mono++;
                                }
                            }
                        }
                    }
                }
            }

            // if(F.Nleft != -1 && pMP->mbTrackInViewR){
            //     const int &nPredictedLevel = pMP->mnTrackScaleLevelR;
            //     if(nPredictedLevel != -1){
            //         float r = RadiusByViewingCos(pMP->mTrackViewCosR);

            //         const vector<size_t> vIndices =
            //                 F.GetFeaturesInArea(pMP->mTrackProjXR,pMP->mTrackProjYR,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel,true);

            //         if(vIndices.empty())
            //             continue;

            //         const cv::Mat MPdescriptor = pMP->GetDescriptor();

            //         int bestDist=256;
            //         int bestLevel= -1;
            //         int bestDist2=256;
            //         int bestLevel2 = -1;
            //         int bestIdx =-1 ;

            //         // Get best and second matches with near keypoints
            //         for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            //         {
            //             const size_t idx = *vit;

            //             if(F.mvpMapPoints[idx + F.Nleft])
            //                 if(F.mvpMapPoints[idx + F.Nleft]->Observations()>0)
            //                     continue;


            //             const cv::Mat &d = F.mDescriptors.row(idx + F.Nleft);

            //             const int dist = DescriptorDistance(MPdescriptor,d);

            //             if(dist<bestDist)
            //             {
            //                 bestDist2=bestDist;
            //                 bestDist=dist;
            //                 bestLevel2 = bestLevel;
            //                 bestLevel = F.mvKeysRight[idx].octave;
            //                 bestIdx=idx;
            //             }
            //             else if(dist<bestDist2)
            //             {
            //                 bestLevel2 = F.mvKeysRight[idx].octave;
            //                 bestDist2=dist;
            //             }
            //         }

            //         // Apply ratio to second match (only if best and second are in the same scale level)
            //         if(bestDist<=TH_HIGH)
            //         {
            //             if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
            //                 continue;

            //             if(F.Nleft != -1 && F.mvRightToLeftMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
            //                 F.mvpMapPoints[F.mvRightToLeftMatch[bestIdx]] = pMP;
            //                 nmatches++;
            //                 left++;
            //             }


            //             F.mvpMapPoints[bestIdx + F.Nleft]=pMP;
            //             nmatches++;
            //             right++;
            //         }
            //     }
            // }
        }
        // std::cout << "mono: " << nmatches_mono << std::endl;
        return nmatches;
    }

    float ORBmatcher::RadiusByViewingCos(const float &viewCos)
    {
        if(viewCos>0.998)
            return 2.5;
        else
            return 4.0;
    }

    int ORBmatcher::SearchByBoW(MultiKeyFrame* pKF,MultiFrame &F, vector<MapPoint*> &vpMapPointMatches)
    {
        const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

        vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

        const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

        int nmatches=0;

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
        DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
        DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
        DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
        DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

        while(KFit != KFend && Fit != Fend)
        {
            if(KFit->first == Fit->first)
            {
                const vector<unsigned int> vIndicesKF = KFit->second;
                const vector<unsigned int> vIndicesF = Fit->second;

                for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
                {
                    const unsigned int realIdxKF = vIndicesKF[iKF];

                    MapPoint* pMP = vpMapPointsKF[realIdxKF];

                    if(!pMP)
                        continue;

                    if(pMP->isBad())
                        continue;

                    const cv::Mat &dKF= pKF->mvDescriptors[pKF->mmpKeyToCam[realIdxKF]].row(pKF->mmpGlobalToLocalID[realIdxKF]);

                    int bestDist1=256;
                    int bestIdxF =-1 ;
                    int bestDist2=256;

                    int bestDist1R=256;
                    int bestIdxFR =-1 ;
                    int bestDist2R=256;

                    for(size_t iF=0; iF<vIndicesF.size(); iF++)
                    {
                        if(F.Nleft == -1){
                            const unsigned int realIdxF = vIndicesF[iF];

                            if(vpMapPointMatches[realIdxF])
                                continue;

                            const cv::Mat &dF = F.mvDescriptors[F.mmpKeyToCam[realIdxF]].row(F.mmpGlobalToLocalID[realIdxF]);

                            const int dist =  DescriptorDistance(dKF,dF);

                            if(dist<bestDist1)
                            {
                                bestDist2=bestDist1;
                                bestDist1=dist;
                                bestIdxF=realIdxF;
                            }
                            else if(dist<bestDist2)
                            {
                                bestDist2=dist;
                            }
                        }
                        // else{
                        //     const unsigned int realIdxF = vIndicesF[iF];

                        //     if(vpMapPointMatches[realIdxF])
                        //         continue;

                        //     const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                        //     const int dist =  DescriptorDistance(dKF,dF);

                        //     if(realIdxF < F.Nleft && dist<bestDist1){
                        //         bestDist2=bestDist1;
                        //         bestDist1=dist;
                        //         bestIdxF=realIdxF;
                        //     }
                        //     else if(realIdxF < F.Nleft && dist<bestDist2){
                        //         bestDist2=dist;
                        //     }

                        //     if(realIdxF >= F.Nleft && dist<bestDist1R){
                        //         bestDist2R=bestDist1R;
                        //         bestDist1R=dist;
                        //         bestIdxFR=realIdxF;
                        //     }
                        //     else if(realIdxF >= F.Nleft && dist<bestDist2R){
                        //         bestDist2R=dist;
                        //     }
                        // }

                    }

                    if(bestDist1<=TH_LOW)
                    {
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                        {
                            vpMapPointMatches[bestIdxF]=pMP;

                            const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                            if(mbCheckOrientation)
                            {
                                // TODO: 去畸变后的特征点方向检查
                                cv::KeyPoint &Fkp = F.mvKeys[bestIdxF];

                                float rot = kp.angle-Fkp.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxF);
                            }
                            nmatches++;
                        }

                        // if(bestDist1R<=TH_LOW)
                        // {
                        //     if(static_cast<float>(bestDist1R)<mfNNratio*static_cast<float>(bestDist2R) || true)
                        //     {
                        //         vpMapPointMatches[bestIdxFR]=pMP;

                        //         const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                        //         if(mbCheckOrientation)
                        //         {
                        //             cv::KeyPoint &Fkp = F.mvKeys[bestIdxFR];
                        //                     (!F.mpCamera2) ? F.mvKeys[bestIdxFR] :
                        //                     (bestIdxFR >= F.Nleft) ? F.mvKeysRight[bestIdxFR - F.Nleft]
                        //                                            : F.mvKeys[bestIdxFR];

                        //             float rot = kp.angle-Fkp.angle;
                        //             if(rot<0.0)
                        //                 rot+=360.0f;
                        //             int bin = round(rot*factor);
                        //             if(bin==HISTO_LENGTH)
                        //                 bin=0;
                        //             assert(bin>=0 && bin<HISTO_LENGTH);
                        //             rotHist[bin].push_back(bestIdxFR);
                        //         }
                        //         nmatches++;
                        //     }
                        // }
                    }

                }

                KFit++;
                Fit++;
            }
            else if(KFit->first < Fit->first)
            {
                KFit = vFeatVecKF.lower_bound(Fit->first);
            }
            else
            {
                Fit = F.mFeatVec.lower_bound(KFit->first);
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }

        return nmatches;
    }

    int ORBmatcher::SearchByProjection(MultiKeyFrame* pKF, Sophus::Sim3f &Sbw, const vector<MapPoint*> &vpPoints,
                                       vector<MapPoint*> &vpMatched, int th, float ratioHamming)
    {
        // Get Calibration Parameters for later projection
        float fx, fy, cx, cy;

        MultiKeyFrame* prevKF = pKF->mPrevKF;

        Sophus::SE3f Tbw = Sophus::SE3f(Sbw.rotationMatrix(),Sbw.translation()/Sbw.scale());
        Sophus::SE3f Twb = Tbw.inverse();
        Eigen::Vector3f Ow = Tbw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

        int nmatches=0;

        if (!pKF->mPrevKF) return 0;

        Eigen::VectorXf Vw_prev = prevKF->GetVelocity();
        Eigen::VectorXf Vw = pKF->GetVelocity();
        Sophus::SE3f Tcl = pKF->GetPose() * prevKF->GetPose().inverse();
        Sophus::SE3f Twl = Twb * Tcl;

        vector<Sophus::SE3f> vTcw;
        for (int c = 0; c < pKF->nCamera; ++c)
        {
            Sophus::SE3f Twb_c = pKF->mpGP->QueryPose(Twl.cast<double>(), Twb.cast<double>(), Vw_prev.cast<double>(), Vw.cast<double>(), prevKF->mTimeStamp, pKF->mTimeStamp, pKF->mvTimeStamps[c]).cast<float>();
            vTcw.push_back((Twb_c * pKF->mTbc[c]).inverse());
        }

        // For each Candidate MapPoint Project and Match
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            for (int c = 0; c < pKF->nCamera; ++c) {
                fx = pKF->mvfx[c];
                fy = pKF->mvfy[c];
                cx = pKF->mvcx[c];
                cy = pKF->mvcy[c];
                // Transform into Camera Coords.
                Eigen::Vector3f p3Dc = vTcw[c] * p3Dw;

                // Depth must be positive
                if(p3Dc(2)<0.0)
                    continue;

                // Project into Image
                const Eigen::Vector2f uv = pKF->mvpCamera[c]->project(p3Dc);

                // Point must be inside the image
                if(!pKF->IsInImage(c, uv(0),uv(1)))
                    continue;

                // Depth must be inside the scale invariance region of the point
                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();
                Eigen::Vector3f PO = p3Dw - pKF->GetCameraCenter(c);
                const float dist = PO.norm();

                if(dist<minDistance || dist>maxDistance)
                    continue;

                // Viewing angle must be less than 60 deg
                Eigen::Vector3f Pn = pMP->GetNormal();

                if(PO.dot(Pn)<0.5*dist)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist,pKF);

                // Search in a radius
                const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices = pKF->GetFeaturesInArea(c, uv(0),uv(1),radius);

                if(vIndices.empty())
                    continue;

                // Match to the most similar keypoint in the radius
                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx = -1;
                for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                {
                    const size_t idx = *vit;
                    if(vpMatched[idx])
                        continue;

                    const int &kpLevel= pKF->mvKeysUn[idx].octave;

                    if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                        continue;

                    const cv::Mat &dKF = pKF->mvDescriptors[c].row(pKF->mmpGlobalToLocalID[idx]);

                    const int dist = DescriptorDistance(dMP,dKF);

                    if(dist<bestDist)
                    {
                        bestDist = dist;
                        bestIdx = idx;
                    }
                }

                if(bestDist<=TH_LOW*ratioHamming)
                {
                    vpMatched[bestIdx]=pMP;
                    nmatches++;
                }

            }
        }

        return nmatches;
    }

    int ORBmatcher::SearchByProjection(MultiKeyFrame* pKF, Sophus::Sim3<float> &Sbw, const std::vector<MapPoint*> &vpPoints, const std::vector<MultiKeyFrame*> &vpPointsKFs,
                                       std::vector<MapPoint*> &vpMatched, std::vector<MultiKeyFrame*> &vpMatchedKF, int th, float ratioHamming)
    {
        // Get Calibration Parameters for later projection
        float fx, fy, cx, cy;

        MultiKeyFrame* prevKF = pKF->mPrevKF;

        Sophus::SE3f Tbw = Sophus::SE3f(Sbw.rotationMatrix(),Sbw.translation()/Sbw.scale());
        Sophus::SE3f Twb = Tbw.inverse();
        Eigen::Vector3f Ow = Tbw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

        int nmatches=0;

        if (!pKF->mPrevKF) return 0;

        Eigen::VectorXf Vw_prev = prevKF->GetVelocity();
        Eigen::VectorXf Vw = pKF->GetVelocity();
        Sophus::SE3f Tcl = pKF->GetPose() * prevKF->GetPose().inverse();
        Sophus::SE3f Twl = Twb * Tcl;

        vector<Sophus::SE3f> vTcw;
        for (int c = 0; c < pKF->nCamera; ++c)
        {
            Sophus::SE3f Twb_c = pKF->mpGP->QueryPose(Twl.cast<double>(), Twb.cast<double>(), Vw_prev.cast<double>(), Vw.cast<double>(), prevKF->mTimeStamp, pKF->mTimeStamp, pKF->mvTimeStamps[c]).cast<float>();
            vTcw.push_back((Twb_c * pKF->mTbc[c]).inverse());
        }

        // For each Candidate MapPoint Project and Match
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];
            MultiKeyFrame* pKFi = vpPointsKFs[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            for (int c = 0; c < pKF->nCamera; ++c) {
                fx = pKF->mvfx[c];
                fy = pKF->mvfy[c];
                cx = pKF->mvcx[c];
                cy = pKF->mvcy[c];


                // Transform into Camera Coords.
                Eigen::Vector3f p3Dc = vTcw[c] * p3Dw;

                // Depth must be positive
                if(p3Dc(2)<0.0)
                    continue;

                // Project into Image
                const float invz = 1/p3Dc(2);
                const float x = p3Dc(0)*invz;
                const float y = p3Dc(1)*invz;

                const float u = fx*x+cx;
                const float v = fy*y+cy;

                // Point must be inside the image
                if(!pKF->IsInImage(c, u,v))
                    continue;

                // Depth must be inside the scale invariance region of the point
                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();
                Eigen::Vector3f PO = p3Dw-pKF->GetCameraCenter(c);
                const float dist = PO.norm();

                if(dist<minDistance || dist>maxDistance)
                    continue;

                // Viewing angle must be less than 60 deg
                Eigen::Vector3f Pn = pMP->GetNormal();

                if(PO.dot(Pn)<0.5*dist)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist,pKF);

                // Search in a radius
                const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices = pKF->GetFeaturesInArea(c, u,v,radius);

                if(vIndices.empty())
                    continue;

                // Match to the most similar keypoint in the radius
                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx = -1;
                for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                {
                    const size_t idx = *vit;
                    if(vpMatched[idx])
                        continue;

                    const int &kpLevel= pKF->mvKeysUn[idx].octave;

                    if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                        continue;

                    const cv::Mat &dKF = pKF->mvDescriptors[c].row(pKF->mmpGlobalToLocalID[idx]);

                    const int dist = DescriptorDistance(dMP,dKF);

                    if(dist<bestDist)
                    {
                        bestDist = dist;
                        bestIdx = idx;
                    }
                }

                if(bestDist<=TH_LOW*ratioHamming)
                {
                    vpMatched[bestIdx] = pMP;
                    vpMatchedKF[bestIdx] = pKFi;
                    nmatches++;
                }


                }
        }

        return nmatches;
    }

    int ORBmatcher::SearchForInitialization(MultiFrame &F1, MultiFrame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
    {
        int nmatches=0;
        vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
        vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

        for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
        {
            cv::KeyPoint kp1 = F1.mvKeysUn[i1];
            int level1 = kp1.octave;
            if(level1>0)
                continue;

            vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

            if(vIndices2.empty())
                continue;

            int cam1 = F1.mmpKeyToCam[i1];
            cv::Mat d1 = F1.mvDescriptors[cam1].row(F1.mmpGlobalToLocalID[i1]);

            int bestDist = INT_MAX;
            int bestDist2 = INT_MAX;
            int bestIdx2 = -1;

            for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
            {
                size_t i2 = *vit;

                cv::Mat d2 = F2.mvDescriptors[F2.mmpKeyToCam[i2]].row(F2.mmpGlobalToLocalID[i2]);

                int dist = DescriptorDistance(d1,d2);

                if(vMatchedDistance[i2]<=dist)
                    continue;

                if(dist<bestDist)
                {
                    bestDist2=bestDist;
                    bestDist=dist;
                    bestIdx2=i2;
                }
                else if(dist<bestDist2)
                {
                    bestDist2=dist;
                }
            }

            if(bestDist<=TH_LOW)
            {
                if(bestDist<(float)bestDist2*mfNNratio)
                {
                    if(vnMatches21[bestIdx2]>=0)
                    {
                        vnMatches12[vnMatches21[bestIdx2]]=-1;
                        nmatches--;
                    }
                    vnMatches12[i1]=bestIdx2;
                    vnMatches21[bestIdx2]=i1;
                    vMatchedDistance[bestIdx2]=bestDist;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(i1);
                    }
                }
            }

        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    int idx1 = rotHist[i][j];
                    if(vnMatches12[idx1]>=0)
                    {
                        vnMatches12[idx1]=-1;
                        nmatches--;
                    }
                }
            }

        }

        //Update prev matched
        for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
            if(vnMatches12[i1]>=0)
                vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

        return nmatches;
    }

    int ORBmatcher::SearchByBoW(MultiKeyFrame *pKF1, MultiKeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
    {
        const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
        const vector<cv::Mat> &Descriptors1 = pKF1->mvDescriptors;

        const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
        const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
        const vector<cv::Mat> &Descriptors2 = pKF2->mvDescriptors;

        vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
        vector<bool> vbMatched2(vpMapPoints2.size(),false);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f/HISTO_LENGTH;

        int nmatches = 0;

        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while(f1it != f1end && f2it != f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];
                    if(pKF1 -> NLeft != -1 && idx1 >= pKF1 -> mvKeysUn.size()){
                        continue;
                    }

                    MapPoint* pMP1 = vpMapPoints1[idx1];
                    if(!pMP1)
                        continue;
                    if(pMP1->isBad())
                        continue;

                    const cv::Mat &d1 = Descriptors1[pKF1->mmpKeyToCam[idx1]].row(pKF1->mmpGlobalToLocalID[idx1]);

                    int bestDist1=256;
                    int bestIdx2 =-1 ;
                    int bestDist2=256;

                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        const size_t idx2 = f2it->second[i2];

                        if(pKF2 -> NLeft != -1 && idx2 >= pKF2 -> mvKeysUn.size()){
                            continue;
                        }

                        MapPoint* pMP2 = vpMapPoints2[idx2];

                        if(vbMatched2[idx2] || !pMP2)
                            continue;

                        if(pMP2->isBad())
                            continue;

                        const cv::Mat &d2 = Descriptors2[pKF2->mmpKeyToCam[idx2]].row(pKF2->mmpGlobalToLocalID[idx2]);

                        int dist = DescriptorDistance(d1,d2);

                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }

                    if(bestDist1<TH_LOW)
                    {
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                        {
                            vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                            vbMatched2[bestIdx2]=true;

                            if(mbCheckOrientation)
                            {
                                float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
                            nmatches++;
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }

        return nmatches;
    }

    int ORBmatcher::SearchForTriangulation(MultiKeyFrame *pKF1, MultiKeyFrame *pKF2,
                                           vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
    {
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
        
        std::vector<std::vector<Sophus::SE3f>> vvT12(pKF1->nCamera);
        std::vector<std::vector<Eigen::Matrix3f>> vvR12(pKF1->nCamera);
        std::vector<std::vector<Eigen::Vector3f>> vvt12(pKF1->nCamera);
        std::vector<std::vector<Eigen::Vector2f>> vvep(pKF1->nCamera);

        for (int c1 = 0; c1 < pKF1->nCamera; ++c1)
        {
            // Compute epipole in second image
            Sophus::SE3f T1cw = pKF1->GetCameraPose(c1);
            Eigen::Vector3f C1w = pKF1->GetCameraCenter(c1);
            for (int c2 = 0; c2 < pKF2->nCamera; ++c2)
            {
                Sophus::SE3f Twc2 = pKF2->GetCameraPoseInverse(c2);
                Sophus::SE3f Tc2w = pKF2->GetCameraPose(c2);
                Eigen::Vector3f C2 = Tc2w * C1w;
                vvT12[c1].push_back(T1cw * Twc2);
                vvR12[c1].push_back(vvT12[c1][c2].rotationMatrix());
                vvt12[c1].push_back(vvT12[c1][c2].translation());
                vvep[c1].push_back(pKF2->mvpCamera[c2]->project(C2));
            }
        }

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node
        int nmatches = 0;
        vector<bool> vbMatched2(pKF2->N, false);
        vector<int> vMatches12(pKF1->N, -1);

        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f / HISTO_LENGTH;

        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while (f1it != f1end && f2it != f2end)
        {
            if (f1it->first == f2it->first)
            {
                for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; ++i1)
                {
                    const size_t idx1 = f1it->second[i1];

                    int cam1 = pKF1->mmpKeyToCam[idx1];
                    MapPoint *pMP1 = pKF1->GetMapPoint(idx1);

                    // If there is already a MapPoint skip
                    if(pMP1)
                    {
                        continue;
                    }

                    const bool bStereo1 = (cam1 == pKF1->nCamera-1 && pKF1->mvuRight[pKF1->mmpGlobalToLocalID[idx1]] >= 0);
                    if (bOnlyStereo)
                        if (!bStereo1)
                            continue;

                    const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];

                    const cv::Mat &d1 = pKF1->mvDescriptors[cam1].row(pKF1->mmpGlobalToLocalID[idx1]);

                    int bestDist = TH_LOW;
                    int bestIdx2 = -1;

                    for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; ++i2)
                    {
                        size_t idx2 = f2it->second[i2];

                        MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

                        if (vbMatched2[idx2] || pMP2)
                            continue;

                        int cam2 = pKF2->mmpKeyToCam[idx2];

                        const bool bStereo2 = (cam2 == pKF2->nCamera-1 && pKF2->mvuRight[pKF2->mmpGlobalToLocalID[idx2]] >= 0);
                        if (bOnlyStereo)
                            if (!bStereo2)
                                continue;

                        const cv::Mat &d2 = pKF2->mvDescriptors[cam2].row(pKF2->mmpGlobalToLocalID[idx2]);

                        const int dist = DescriptorDistance(d1, d2);

                        if (dist > TH_LOW || dist > bestDist)
                            continue;

                        const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                        if (!bStereo1 && !bStereo2)
                        {
                            const float distex = vvep[cam1][cam2](0) - kp2.pt.x;
                            const float distey = vvep[cam1][cam2](1) - kp2.pt.y;
                            if (distex * distex + distey * distey < 100 * pKF2->mvScaleFactors[kp2.octave])
                                continue;
                        }

                        if (bCoarse || pKF1->mvpCamera[cam1]->epipolarConstrain(
                            pKF2->mvpCamera[cam2], kp1, kp2, vvR12[cam1][cam2], vvt12[cam1][cam2], 
                            pKF1->mvLevelSigma2[kp1.octave], pKF2->mvLevelSigma2[kp2.octave]
                        ))
                        {
                            bestIdx2 = idx2;
                            bestDist = dist;
                        }
                    }

                    if (bestIdx2 >= 0)
                    {
                        const cv::KeyPoint& kp2 = pKF2->mvKeysUn[bestIdx2];
                        vMatches12[idx1] = bestIdx2;
                        nmatches++;

                        if (mbCheckOrientation)
                        {
                            float rot = kp1.angle - kp2.angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                    }

                }
                f1it++;
                f2it++;
            }
            else if (f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if (mbCheckOrientation)
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; ++i)
            {
                if (i == ind1 || i == ind2 || i == ind3)
                {
                    for (size_t j = 0, jend = rotHist[i].size(); j < jend; ++j)
                    {
                        vMatches12[rotHist[i][j]] = -1;
                        nmatches--;
                    }
                }
            }
        }

        vMatchedPairs.clear();
        vMatchedPairs.reserve(nmatches);

        for (size_t i = 0, iend = vMatches12.size(); i < iend; ++i)
        {
            if (vMatches12[i] < 0)
                continue;
            vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
        }

        return nmatches;

    }

    int ORBmatcher::Fuse(MultiKeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
    {
        const vector<GeometricCamera*>& vpCamera = pKF->mvpCamera;
        vector<Sophus::SE3f> vTcw = pKF->GetCameraPoses();   
        vector<Eigen::Vector3f> vOw;

        for (int c = 0; c < pKF->nCamera; ++c)
        {
            vOw.push_back(pKF->GetCameraCenter(c));
        }

        const vector<float> &vfx = pKF->mvfx;
        const vector<float> &vfy = pKF->mvfy;
        const vector<float> &vcx = pKF->mvcx;
        const vector<float> &vcy = pKF->mvcy;
        const float &bf = pKF->mbf;

        int nFused = 0;

        const int nMPs = vpMapPoints.size();

        // For debbuging
        int count_notMP = 0, count_bad=0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal=0, count_notidx = 0, count_thcheck = 0;

        for (int i = 0; i < nMPs; ++i)
        {
            MapPoint* pMP = vpMapPoints[i];

            if (!pMP)
            {
                count_notMP++;
                continue;
            }

            if (pMP->isBad())
            {
                count_bad++;
                continue;
            }
            
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            for (int c = 0; c < pKF->nCamera; ++c)
            {
                if (pMP->IsInKFCamera(pKF, c))
                {
                    count_isinKF++;
                    continue;
                }

                Eigen::Vector3f p3Dc = vTcw[c] * p3Dw;

                // Depth must be positive
                if(p3Dc(2)<0.0f)
                {
                    count_negdepth++;
                    continue;
                }

                const float invz = 1.0f / p3Dc(2);

                const Eigen::Vector2f uv = vpCamera[c]->project(p3Dc);

                // Point must be inside the image
                if (!pKF->IsInImage(c, uv(0), uv(1)))
                {
                    count_notinim++;
                    continue;
                }

                const float ur = uv(0) - bf * invz;

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();
                Eigen::Vector3f PO = p3Dw - vOw[c];
                const float dist3D = PO.norm();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance) {
                    count_dist++;
                    continue;
                }

                Eigen::Vector3f Pn = pMP->GetNormal();

                if(PO.dot(Pn)<0.5*dist3D)
                {
                    count_normal++;
                    continue;
                }

                int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

                // Search in a radius
                const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices = pKF->GetFeaturesInArea(c, uv(0), uv(1), radius);

                if(vIndices.empty())
                {
                    count_notidx++;
                    continue;
                }

                // Match to the most similar keypoint in the radius
                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx = -1;

                for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
                {
                    size_t idx = *vit;
                    const cv::KeyPoint &kp = pKF->mvKeysUn[idx];
                    const int &kpLevel = kp.octave;

                    if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                        continue;

                    int localID = pKF->mmpGlobalToLocalID[idx];

                    if (c == pKF->nCamera-1 && pKF->mvuRight[localID] >= 0)
                    {
                        const float &kpx = kp.pt.x;
                        const float &kpy = kp.pt.y;
                        const float &kpr = pKF->mvuRight[localID];
                        const float ex = uv(0) - kpx;
                        const float ey = uv(1) - kpy;
                        const float er = ur - kpr;
                        const float e2 = ex * ex + ey * ey + er * er;

                        if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 7.8)
                            continue;
                    }
                    else
                    {
                        const float &kpx = kp.pt.x;
                        const float &kpy = kp.pt.y;
                        const float ex = uv(0) - kpx;
                        const float ey = uv(1) - kpy;
                        const float e2 = ex * ex + ey * ey;

                        if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99)
                            continue;
                    }

                    const cv::Mat &dKF = pKF->mvDescriptors[c].row(localID);

                    const int dist = DescriptorDistance(dMP, dKF);

                    if(dist<bestDist)
                    {
                        bestDist = dist;
                        bestIdx = idx;
                    }
                }
                // If there is already a MapPoint replace otherwise add new measurement
                if(bestDist<=TH_LOW)
                {
                    MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                    if(pMPinKF)
                    {
                        if(!pMPinKF->isBad())
                        {
                            if(pMPinKF->Observations()>pMP->Observations())
                                pMP->Replace(pMPinKF);
                            else
                                pMPinKF->Replace(pMP);
                        }
                    }
                    else
                    {
                        pMP->AddObservation(pKF,bestIdx);
                        pKF->AddMapPoint(pMP,bestIdx);
                    }
                    nFused++;
                }
                else
                    count_thcheck++;
            }
        }
        return nFused;

    }

    int ORBmatcher::Fuse(MultiKeyFrame *pKF, Sophus::Sim3f &Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
    {
        // Get Calibration Parameters for later projection
        const vector<float> &vfx = pKF->mvfx;
        const vector<float> &vfy = pKF->mvfy;
        const vector<float> &vcx = pKF->mvcx;
        const vector<float> &vcy = pKF->mvcy;

        // Decompose Scw
        // TODO： 直接使用尺度为1修改后的位姿
        // Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        vector<Sophus::SE3f> vTcw = pKF->GetCameraPoses();

        // Set of MapPoints already found in the KeyFrame
        const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

        int nFused=0;

        const int nPoints = vpPoints.size();

        // For each candidate MapPoint project and match
        for(int iMP=0; iMP<nPoints; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            for (int c = 0; c < pKF->nCamera-1; ++c)
            {
                // Transform into Camera Coords.
                Eigen::Vector3f p3Dc = vTcw[c] * p3Dw;

                // Depth must be positive
                if(p3Dc(2)<0.0f)
                    continue;

                // Project into Image
                const Eigen::Vector2f uv = pKF->mvpCamera[c]->project(p3Dc);

                // Point must be inside the image
                if(!pKF->IsInImage(c, uv(0),uv(1)))
                    continue;

                // Depth must be inside the scale pyramid of the image
                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();
                Eigen::Vector3f PO = p3Dw-vTcw[c].inverse().translation();
                const float dist3D = PO.norm();

                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                // Viewing angle must be less than 60 deg
                Eigen::Vector3f Pn = pMP->GetNormal();

                if(PO.dot(Pn)<0.5*dist3D)
                    continue;

                // Compute predicted scale level
                const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

                // Search in a radius
                const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices = pKF->GetFeaturesInArea(c, uv(0),uv(1),radius);

                if(vIndices.empty())
                    continue;

                // Match to the most similar keypoint in the radius

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = INT_MAX;
                int bestIdx = -1;
                for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
                {
                    const size_t idx = *vit;
                    const int &kpLevel = pKF->mvKeysUn[idx].octave;

                    if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                        continue;

                    const cv::Mat &dKF = pKF->mvDescriptors[c].row(pKF->mmpGlobalToLocalID[idx]);

                    int dist = DescriptorDistance(dMP,dKF);

                    if(dist<bestDist)
                    {
                        bestDist = dist;
                        bestIdx = idx;
                    }
                }

                // If there is already a MapPoint replace otherwise add new measurement
                if(bestDist<=TH_LOW)
                {
                    MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                    if(pMPinKF)
                    {
                        if(!pMPinKF->isBad())
                            vpReplacePoint[iMP] = pMPinKF;
                    }
                    else
                    {
                        pMP->AddObservation(pKF,bestIdx);
                        pKF->AddMapPoint(pMP,bestIdx);
                    }
                    nFused++;
                }
            }
        }

        return nFused;
    }

    int ORBmatcher::SearchByProjection(MultiFrame &CurrentFrame, const MultiFrame &LastFrame, const float th, const bool bMono)
    {
        int nmatches = 0;
        int nmatches_mono = 0;
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        vector<Sophus::SE3f>& vTcw = CurrentFrame.mTcw;
        vector<Sophus::SE3f>& vTwc = CurrentFrame.mTwc;


        for (int i = 0; i < LastFrame.N; ++i) {
            // TODO: 已经投影过的点不再投影？
            MapPoint* pMP = LastFrame.mvpMapPoints[i];
            if (pMP) {
                if (!LastFrame.mvbOutlier[i]) {
                    Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                    for (int c = 0; c < CurrentFrame.nCamera; ++c) {
                        Eigen::Vector3f x3Dc = vTcw[c] * x3Dw;

                        const float xc = x3Dc(0);
                        const float yc = x3Dc(1);
                        const float invzc = 1.0/x3Dc(2);   

                        if (invzc < 0) continue;

                        Eigen::Vector2f uv = CurrentFrame.mvpCamera[c]->project(x3Dc);

                        if (uv(0) < CurrentFrame.mvMinX[c] || uv(0) > CurrentFrame.mvMaxX[c])
                            continue;
                        if (uv(1) < CurrentFrame.mvMinY[c] || uv(1) > CurrentFrame.mvMaxY[c])
                            continue;

                        int nLastOctave = LastFrame.mvKeys[i].octave;

                        float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];

                        vector<size_t> vIndices2;

                        vIndices2 = CurrentFrame.GetFeaturesInArea(c, uv(0), uv(1), radius, nLastOctave-1, nLastOctave+1);

                        if (vIndices2.empty())
                            continue;

                        const cv::Mat dMP = pMP->GetDescriptor();

                        int bestDist = 256;
                        int bestIdx2 = -1;

                        for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit!=vend; ++vit){
                            const size_t i2 = *vit;

                            const int localID = CurrentFrame.mmpGlobalToLocalID[i2];

                            if (CurrentFrame.mvpMapPoints[i2])
                                if (CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                                    continue;

                            if (CurrentFrame.mmpKeyToCam[i2] == CurrentFrame.nCamera-1) {
                                int uRight = CurrentFrame.mvuRight[localID];
                                if (uRight > 0) {
                                    const float ur  = uv(0) - CurrentFrame.mbf * invzc;
                                    const float er = fabs(ur - uRight);
                                    if (er > radius)
                                        continue;
                                }
                            }

                            const cv::Mat &d = CurrentFrame.mvDescriptors[c].row(localID);

                            const int dist = DescriptorDistance(dMP, d);

                            if (dist < bestDist) {
                                bestDist = dist;
                                bestIdx2 = i2;
                            }
                        }

                        if (bestDist <= TH_HIGH) {
                            CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
                            // CurrentFrame.mMatchedFeatures.insert(bestIdx2);
                            nmatches++;
                            if (c < CurrentFrame.nCamera-1)
                                nmatches_mono++;

                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint kpLF = LastFrame.mvKeysUn[i];

                                cv::KeyPoint kpCF = CurrentFrame.mvKeysUn[bestIdx2];
                                float rot = kpLF.angle-kpCF.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdx2);
                            }
                        }
                    }
                }
            }
        }

        //Apply rotation consistency
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i!=ind1 && i!=ind2 && i!=ind3)
                {
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                        // CurrentFrame.mMatchedFeatures.erase(rotHist[i][j]);
                        nmatches--;
                    }
                }
            }
        }

        // std::cout << "nmatches: " << nmatches << " nmatches_mono: " << nmatches_mono << std::endl;

        return nmatches;
    }

    // int ORBmatcher::SearchByProjection(MultiFrame &CurrentFrame, MultiKeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
    // {
    //     int nmatches = 0;

    //     const Sophus::SE3f Tcw = CurrentFrame.GetPose();
    //     Eigen::Vector3f Ow = Tcw.inverse().translation();

    //     // Rotation Histogram (to check rotation consistency)
    //     vector<int> rotHist[HISTO_LENGTH];
    //     for(int i=0;i<HISTO_LENGTH;i++)
    //         rotHist[i].reserve(500);
    //     const float factor = 1.0f/HISTO_LENGTH;

    //     const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    //     for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    //     {
    //         MapPoint* pMP = vpMPs[i];

    //         if(pMP)
    //         {
    //             if(!pMP->isBad() && !sAlreadyFound.count(pMP))
    //             {
    //                 //Project
    //                 Eigen::Vector3f x3Dw = pMP->GetWorldPos();
    //                 Eigen::Vector3f x3Dc = Tcw * x3Dw;

    //                 const Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

    //                 if(uv(0)<CurrentFrame.mnMinX || uv(0)>CurrentFrame.mnMaxX)
    //                     continue;
    //                 if(uv(1)<CurrentFrame.mnMinY || uv(1)>CurrentFrame.mnMaxY)
    //                     continue;

    //                 // Compute predicted scale level
    //                 Eigen::Vector3f PO = x3Dw-Ow;
    //                 float dist3D = PO.norm();

    //                 const float maxDistance = pMP->GetMaxDistanceInvariance();
    //                 const float minDistance = pMP->GetMinDistanceInvariance();

    //                 // Depth must be inside the scale pyramid of the image
    //                 if(dist3D<minDistance || dist3D>maxDistance)
    //                     continue;

    //                 int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

    //                 // Search in a window
    //                 const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

    //                 const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), radius, nPredictedLevel-1, nPredictedLevel+1);

    //                 if(vIndices2.empty())
    //                     continue;

    //                 const cv::Mat dMP = pMP->GetDescriptor();

    //                 int bestDist = 256;
    //                 int bestIdx2 = -1;

    //                 for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
    //                 {
    //                     const size_t i2 = *vit;
    //                     if(CurrentFrame.mvpMapPoints[i2])
    //                         continue;

    //                     const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

    //                     const int dist = DescriptorDistance(dMP,d);

    //                     if(dist<bestDist)
    //                     {
    //                         bestDist=dist;
    //                         bestIdx2=i2;
    //                     }
    //                 }

    //                 if(bestDist<=ORBdist)
    //                 {
    //                     CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
    //                     nmatches++;

    //                     if(mbCheckOrientation)
    //                     {
    //                         float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
    //                         if(rot<0.0)
    //                             rot+=360.0f;
    //                         int bin = round(rot*factor);
    //                         if(bin==HISTO_LENGTH)
    //                             bin=0;
    //                         assert(bin>=0 && bin<HISTO_LENGTH);
    //                         rotHist[bin].push_back(bestIdx2);
    //                     }
    //                 }

    //             }
    //         }
    //     }

    //     if(mbCheckOrientation)
    //     {
    //         int ind1=-1;
    //         int ind2=-1;
    //         int ind3=-1;

    //         ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

    //         for(int i=0; i<HISTO_LENGTH; i++)
    //         {
    //             if(i!=ind1 && i!=ind2 && i!=ind3)
    //             {
    //                 for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
    //                 {
    //                     CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
    //                     nmatches--;
    //                 }
    //             }
    //         }
    //     }

    //     return nmatches;
    // }

    void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1=0;
        int max2=0;
        int max3=0;

        for(int i=0; i<L; i++)
        {
            const int s = histo[i].size();
            if(s>max1)
            {
                max3=max2;
                max2=max1;
                max1=s;
                ind3=ind2;
                ind2=ind1;
                ind1=i;
            }
            else if(s>max2)
            {
                max3=max2;
                max2=s;
                ind3=ind2;
                ind2=i;
            }
            else if(s>max3)
            {
                max3=s;
                ind3=i;
            }
        }

        if(max2<0.1f*(float)max1)
        {
            ind2=-1;
            ind3=-1;
        }
        else if(max3<0.1f*(float)max1)
        {
            ind3=-1;
        }
    }


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist=0;

        for(int i=0; i<8; i++, pa++, pb++)
        {
            unsigned  int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

} //namespace ORB_SLAM
