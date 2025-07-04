/**
 * 
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


#include "Optimizer.h"


#include <complex>

#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "Thirdparty/g2o/g2o/core/sparse_block_matrix.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "G2oTypes.h"
#include "Converter.h"

#include<mutex>

#include "OptimizableTypes.h"


namespace ORB_SLAM3
{
bool sortByVal(const pair<MapPoint*, int> &a, const pair<MapPoint*, int> &b)
{
    return (a.second < b.second);
}

void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<MultiKeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
}


void Optimizer::BundleAdjustment(const vector<MultiKeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    Map* pMap = vpKFs[0]->GetMap();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    solver->setUserLambdaInit(1e-5);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set Keyframe vertices
    for (size_t i = 0; i < vpKFs.size(); ++i)
    {
        MultiKeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;

        VertexPoseVel* v = new VertexPoseVel(pKF);
        v->setId(pKF->mnId);
        v->setFixed(pKF->mnId == pMap->GetInitKFid());
        optimizer.addVertex(v);
        if (pKF->mnId > maxKFid)
            maxKFid = pKF->mnId;
    } 

    // Gaussian Process constraints
    for (size_t i = 0; i < vpKFs.size(); ++i)
    {
        MultiKeyFrame* pKFi = vpKFs[i];
        g2o::HyperGraph::Vertex* v = optimizer.vertex(pKFi->mnId);

        EdgeVelocity* e = new EdgeVelocity();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v));
        e->setInformation(pKFi->mpGP->mQcInv.block<1, 1>(2, 2));
        optimizer.addEdge(e);

        if (!pKFi->mPrevKF)
        {
            Verbose::PrintMess("NOT INERTIAL LINK TO PREVIOUS FRAME!", Verbose::VERBOSITY_NORMAL);
            continue;
        }

        if (pKFi->mPrevKF && pKFi->mnId <= maxKFid)
        {
            if (pKFi->isBad() || pKFi->mPrevKF->mnId > maxKFid)
                continue;

            g2o::HyperGraph::Vertex* v1 = optimizer.vertex(pKFi->mPrevKF->mnId);
            g2o::HyperGraph::Vertex* v2 = optimizer.vertex(pKFi->mnId);

            EdgeGaussianPrior* e = new EdgeGaussianPrior();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v1));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v2));

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(21.026);

            e->setInformation(pKFi->mpGP->QiInv(pKFi->mTimeStamp - pKFi->mPrevKF->mTimeStamp));

            optimizer.addEdge(e);
        }
    }

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    const unsigned long iniMPid = maxKFid*5;

    vector<bool> vbNotIncludedMP(vpMP.size(), false);

    for (size_t i = 0; i < vpMP.size(); ++i)
    {
        MapPoint* pMP = vpMP[i];
        g2o::VertexSBAPointXYZ* vP = new g2o::VertexSBAPointXYZ();
        vP->setEstimate(pMP->GetWorldPos().cast<double>());
        unsigned long id = pMP->mnId + iniMPid + 1;
        vP->setId(id);
        vP->setMarginalized(true);
        optimizer.addVertex(vP);

        const map<MultiKeyFrame*, vector<int>>& observations = pMP->GetObservations();
        const unordered_multimap<MultiKeyFrame*, GPObs>& observationsGP = pMP->GetGPObservations();

        int nEdges = 0;
        // SET EDGES
        for (auto mit = observations.begin(); mit != observations.end(); ++mit)
        {
            MultiKeyFrame* pKFi = mit->first;
            if (pKFi->isBad() || pKFi->mnId > maxKFid)
                continue;
            if (optimizer.vertex(id) == NULL || optimizer.vertex(pKFi->mnId) == NULL)
                continue;
            nEdges++;
            int cam = 0;
            if (pKFi->mPrevKF != nullptr && pKFi->mPrevKF->mnId <= maxKFid)
                for (cam = 0; cam < mit->second.size()-1; ++cam)
                {
                    int index = mit->second[cam];
                    if (index < 0) continue;
                    cv::KeyPoint kp = pKFi->mvKeysUn[index];
                    Eigen::Vector2d obs(kp.pt.x, kp.pt.y);

                    EdgeMonoGP* e = new EdgeMonoGP(cam, pKFi->mvTimeStamps[cam], pKFi->mpGP);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mPrevKF->mnId)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kp.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                }
            cam = mit->second.size()-1;
            int index = mit->second[cam];

            if (index >= 0)
            {
                const float kp_ur = pKFi->mvuRight[pKFi->mmpGlobalToLocalID[index]];
                cv::KeyPoint kpUn = pKFi->mvKeysUn[index];
                if (kp_ur < 0)
                {
                    Eigen::Vector2d obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMono* e = new EdgeMono();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setMeasurement(obs);

                    // Add here uncertenty
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                }
                else
                {
                    Eigen::Vector3d obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    EdgeStereo* e = new EdgeStereo();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setMeasurement(obs);

                    // Add here uncertenty
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                }
            }
        }

        for (size_t i = 0; i < vpKFs.size(); ++i)
        {
            MultiKeyFrame* pKFi = vpKFs[i];
            if (pKFi->isBad() || !pKFi->mNextKF)
                continue;

            if (pKFi->mnId > maxKFid || pKFi->mNextKF->mnId > maxKFid)
                continue;

            auto mitrange = observationsGP.equal_range(pKFi);
            for (auto mit = mitrange.first, mend = mitrange.second; mit != mend; ++mit)
            {
                const GPObs& GPobs = mit->second;
                if (GPobs.ur >= 0)
                {
                    Eigen::Vector3d obs;
                    const cv::KeyPoint& kpUn = GPobs.obs;

                    obs << kpUn.pt.x, kpUn.pt.y, GPobs.ur;

                    EdgeStereoGP* e = new EdgeStereoGP(GPobs.cam, GPobs.time, pKFi->mpGP);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mNextKF->mnId)));
                    e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setMeasurement(obs);

                    // Add here uncertenty

                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                }
                else
                {
                    Eigen::Vector2d obs;
                    const cv::KeyPoint& kpUn = GPobs.obs;

                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoGP* e = new EdgeMonoGP(GPobs.cam, GPobs.time, pKFi->mpGP);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mNextKF->mnId)));
                    e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setMeasurement(obs);

                    // Add here uncertenty
                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                }

            }
        }
        if (nEdges == 0)
        {
            optimizer.removeVertex(vP);
            vbNotIncludedMP[i] = true;
        }
        else
        {
            vbNotIncludedMP[i] = false;
        }
    }

    // Optimize!
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);
    Verbose::PrintMess("BA: End of the optimization", Verbose::VERBOSITY_NORMAL);
    
    // Recover optimized data
    for (size_t i = 0; i < vpKFs.size(); ++i)
    {
        MultiKeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;

        VertexPoseVel* v = static_cast<VertexPoseVel*>(optimizer.vertex(pKF->mnId));
        if (nLoopKF == 0)
        {
            Sophus::SE3f Tbw(v->estimate().Twb.inverse().cast<float>());
            Eigen::VectorXf Vel = v->estimate().Vel.cast<float>();
            pKF->SetPose(Tbw);
            pKF->SetVelocity(Vel);
        }
        else
        {
            Sophus::SE3f Tbw(v->estimate().Twb.inverse().cast<float>());
            Eigen::VectorXf Vel = v->estimate().Vel.cast<float>();
            pKF->mTbwGBA = Tbw;
            pKF->mVwbGBA = Vel;
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    for (size_t i = 0; i < vpMP.size(); ++i)
    {
        if (vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + iniMPid + 1));

        if (nLoopKF == 0)
        {
            pMP->SetWorldPos(vPoint->estimate().cast<float>());
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA = vPoint->estimate().cast<float>();
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
}

int Optimizer::PoseGPOptimizationFromeLastFrame(MultiFrame *pFrame, bool fix)
{
    // std::cout << "before optimize frame vel: " << pFrame->GetVelocity().transpose() << std::endl;

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType* linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int nInitialMonoCorrespondences=0;
    int nInitialStereoCorrespondences=0;
    int nInitialCorrespondences=0;

    // 上一帧顶点
    VertexPoseVel* v1 = new VertexPoseVel(pFrame->mpPrevFrame);
    v1->setFixed(fix);
    v1->setId(0);
    optimizer.addVertex(v1);

    // 当前帧顶点
    VertexPoseVel* v2 = new VertexPoseVel(pFrame);
    v2->setFixed(false);
    v2->setId(1);
    optimizer.addVertex(v2);

    const int N = pFrame->N;

    vector<EdgeStereoOnlyPose*> vpEdgesStereoOnlyPose;
    vector<EdgeMonoOnlyPose*> vpEdgesMonoOnlyPose;
    vector<EdgeMonoGPOnlyPose*> vpEdgesMonoGPOnlyPose;
    vector<size_t> vnIndexEdgeMonoGPOnlyPose;
    vector<size_t> vnIndexEdgeStereoOnlyPose;
    vector<size_t> vnIndexEdgeMonoOnlyPose;
    vpEdgesStereoOnlyPose.reserve(N);
    vpEdgesMonoGPOnlyPose.reserve(N);
    vpEdgesMonoOnlyPose.reserve(N);
    vnIndexEdgeMonoGPOnlyPose.reserve(N);
    vnIndexEdgeStereoOnlyPose.reserve(N);
    vnIndexEdgeMonoOnlyPose.reserve(N);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for (int i = 0; i < N; ++i)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];
            if (pMP)
            {
                cv::KeyPoint kpUn = pFrame->mvKeysUn[i];
                int cam_idx = pFrame->mmpKeyToCam[i];
                if (cam_idx != pFrame->nCamera-1)
                {
                    nInitialMonoCorrespondences++;

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoGPOnlyPose* e = new EdgeMonoGPOnlyPose(pMP->GetWorldPos(), cam_idx, pFrame->mvTimeStamps[cam_idx],
                                                            pFrame->mpGP);
                    e->setVertex(0, v1);
                    e->setVertex(1, v2);
                    e->setMeasurement(obs);

                    const float unc2 = pFrame->mvpCamera[cam_idx]->uncertainty2(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    if (pFrame->mvbOutlier[i])
                        e->setLevel(1);
                    else
                        e->setLevel(0);

                    optimizer.addEdge(e);

                    vpEdgesMonoGPOnlyPose.push_back(e);
                    vnIndexEdgeMonoGPOnlyPose.push_back(i);
                }
                else
                {
                    nInitialStereoCorrespondences++;
                    // pFrame->mvbOutlier[i] = false;

                    const float kp_ur = pFrame->mvuRight[pFrame->mmpGlobalToLocalID[i]];
                    Eigen::Matrix<double, 3, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur; 

                    if (kp_ur < 0)
                    {
                        // Monocular observation
                        EdgeMonoOnlyPose* e = new EdgeMonoOnlyPose(pMP->GetWorldPos());

                        e->setVertex(0, v2);
                        e->setMeasurement(obs.head(2));

                        const float unc2 = pFrame->mvpCamera[cam_idx]->uncertainty2(obs.head(2));
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        if (pFrame->mvbOutlier[i])
                            e->setLevel(1);
                        else
                            e->setLevel(0);

                        optimizer.addEdge(e);

                        vpEdgesMonoOnlyPose.push_back(e);
                        vnIndexEdgeMonoOnlyPose.push_back(i);
                    }
                    else
                    {
                        // Stereo observation
                        EdgeStereoOnlyPose* e = new EdgeStereoOnlyPose(pMP->GetWorldPos());

                        e->setVertex(0, v2);
                        e->setMeasurement(obs);

                        const float unc2 = pFrame->mvpCamera[cam_idx]->uncertainty2(obs.head(2));
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave] / unc2;
                        e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        if (pFrame->mvbOutlier[i])
                            e->setLevel(1);
                        else
                            e->setLevel(0);

                        optimizer.addEdge(e);

                        vpEdgesStereoOnlyPose.push_back(e);
                        vnIndexEdgeStereoOnlyPose.push_back(i);
                    }
                }
            }
        }
    }

    nInitialCorrespondences = nInitialMonoCorrespondences + nInitialStereoCorrespondences;

    // Set Gaussian Prior
    EdgeGaussianPrior* egp = new EdgeGaussianPrior();
    egp->setVertex(0, v1);
    egp->setVertex(1, v2);
    egp->setInformation(pFrame->mpGP->QiInv(pFrame->mTimeStamp - pFrame->mpPrevFrame->mTimeStamp));
    // g2o::RobustKernel* rkp = new g2o::RobustKernelHuber;
    // e->setRobustKernel(rkp);
    // rkp->setDelta(21.026);
    optimizer.addEdge(egp);

    EdgeVelocity* ev1 = new EdgeVelocity();
    ev1->setVertex(0, v1);
    ev1->setInformation(pFrame->mpGP->mQcInv.block<1, 1>(2, 2));
    optimizer.addEdge(ev1);

    EdgeVelocity* ev2 = new EdgeVelocity();
    ev2->setVertex(0, v2);
    ev2->setInformation(pFrame->mpGP->mQcInv.block<1, 1>(2, 2));
    optimizer.addEdge(ev2);

    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={15.6f,9.8f,7.815f,7.815f};
    const int its[4]={10,10,10,10};

    int nBad=0;
    int nBadMono = 0;
    int nBadStereo = 0;
    int nInliersMono = 0;
    int nInliersStereo = 0;
    int nInliers=0;

    for (size_t it = 0; it < 4; ++it)
    {
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);
        // if (vpEdgesMonoGPOnlyPose.size() > 1)
        // {
        //     EdgeMonoGPOnlyPose* e = vpEdgesMonoGPOnlyPose[0];
        //     std::cout << "pkfid: " << pFrame->mnId << std::endl;
        //     std::cout << "cam: " << e->cam_idx << std::endl;
        //     const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(e->vertex(0));
        //     const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(e->vertex(1));
        //     std::cout << std::fixed << "point time: " << e->t << " prev time: " << v1->estimate().time << " next time: " << v2->estimate().time << std::endl;
        //     saveMatrix("/home/ljj/numerical.csv", e->GetJacobian());
        //     terminate();
        // }

        nBad=0;
        nBadMono = 0;
        nBadStereo = 0;
        nInliers=0;
        nInliersMono=0;
        nInliersStereo=0;
        float chi2close = 1.5*chi2Mono[it];

        for (size_t i = 0, iend = vpEdgesMonoGPOnlyPose.size(); i < iend; ++i)
        {
            EdgeMonoGPOnlyPose* e = vpEdgesMonoGPOnlyPose[i];

            const size_t idx = vnIndexEdgeMonoGPOnlyPose[i];
            const int cam_idx = pFrame->mmpKeyToCam[idx];
            bool bclose = pFrame->mvpMapPoints[idx]->mvTrackDepth[cam_idx]<10.f;

            if (pFrame->mvbOutlier[idx])
                e->computeError();

            const float chi2 = e->chi2();

            if ((chi2 > chi2Mono[it] && !bclose) || (bclose && chi2 > chi2close) || !e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBadMono++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
                nInliersMono++;
            }

            if (it==2)
                e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereoOnlyPose.size(); i < iend; ++i)
        {
            EdgeStereoOnlyPose* e = vpEdgesStereoOnlyPose[i];

            const size_t idx = vnIndexEdgeStereoOnlyPose[i];

            if (pFrame->mvbOutlier[idx])
                e->computeError();

            const float chi2 = e->chi2();

            if (chi2 > chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBadStereo++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
                nInliersStereo++;
            }

            if (it==2)
                e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesMonoOnlyPose.size(); i < iend; ++i)
        {
            EdgeMonoOnlyPose* e = vpEdgesMonoOnlyPose[i];

            const size_t idx = vnIndexEdgeMonoOnlyPose[i];
            bool bclose = pFrame->mvpMapPoints[idx]->mvTrackDepth[pFrame->mmpKeyToCam[idx]]<10.f;

            if (pFrame->mvbOutlier[idx])
                e->computeError();

            const float chi2 = e->chi2();

            if ((chi2 > chi2Mono[it] && !bclose) || (bclose && chi2 > chi2close) || !e->isDepthPositive())
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBadMono++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
                nInliersMono++;
            }

            if (it==2)
                e->setRobustKernel(0);
        }

        nInliers = nInliersMono + nInliersStereo;
        nBad = nBadMono + nBadStereo;
    
        if (optimizer.edges().size() < 10)
            break;
    }

    // saveMatrix("/home/ljj/numerical.csv", egp->GetJacobian());

    pFrame->SetPose(v2->estimate().Twb.inverse().cast<float>());
    pFrame->SetVelocity(v2->estimate().Vel.cast<float>());
    
    // TODO: 边缘化?

    // std::cout << "after optimize frame vel: " << pFrame->GetVelocity().transpose() << std::endl;
    return nInitialCorrespondences - nBad;
}

void Optimizer::saveMatrix(const std::string filename, const Eigen::MatrixXd matrix)
{
    std::ofstream file(filename);

    if (file.is_open())
    {
        for (int i = 0; i < matrix.rows(); ++i)
        {
            for (int j = 0; j < matrix.cols(); ++j)
            {
                file << matrix(i, j);
                if (j < matrix.cols() - 1)
                    file << ",";
            }
            file << "\n";
        }
        file.close();
        std::cout << "Matrix saved to " << filename << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

void Optimizer::LocalGPBA(MultiKeyFrame* pKF, bool *pbStopFlag, Map *pMap, int& num_fixedKF, int& num_OptKF, 
                        int& num_MPs, int& num_edges, bool bLarge, bool bExtrinsic, bool bRecInit)
{
    Map* pCurrentMap = pKF->GetMap();

    int maxOpt = 10;
    int opt_it = 10;
    if (bLarge)
    {
        maxOpt = 25;
        opt_it = 4;
    }
    const int Nd = std::min((int)pCurrentMap->KeyFramesInMap()-2, maxOpt);
    const unsigned long maxKFid = pCurrentMap->GetMaxKFid();

    vector<MultiKeyFrame*> vpOptimizableKFs;
    const vector<MultiKeyFrame*> vpNeighsKFs = pKF->GetVectorCovisibleKeyFrames();
    list<MultiKeyFrame*> lpOptVisKFs;

    vpOptimizableKFs.reserve(Nd);
    vpOptimizableKFs.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    for(int i=1; i<Nd; i++)
    {
        if(vpOptimizableKFs.back()->mPrevKF)
        {
            vpOptimizableKFs.push_back(vpOptimizableKFs.back()->mPrevKF);
            vpOptimizableKFs.back()->mnBALocalForKF = pKF->mnId;
        }
        else
        {
            break;
        }
    }

    int N = vpOptimizableKFs.size();

    // Optimizable points seen by temporal optimizable keyframes
    list<MapPoint*> lLocalMapPoints;
    for (int i = 0; i < N; ++i)
    {
        vector<MapPoint*> vpMPs = vpOptimizableKFs[i]->GetMapPointMatches();
        for (vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit!=vend; ++vit)
        {
            MapPoint* pMP = *vit;
            if (pMP)
                if (!pMP->isBad())
                    if (pMP->mnBALocalForKF != pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
        }
    }

    // Fixed Keyframe: First frame previous KF to optimization window)
    list<MultiKeyFrame*> lFixedKeyFrames;
    if (vpOptimizableKFs.back()->mPrevKF)
    {
        lFixedKeyFrames.push_back(vpOptimizableKFs.back()->mPrevKF);
        vpOptimizableKFs.back()->mPrevKF->mnBAFixedForKF = pKF->mnId;
    }
    else
    {
        vpOptimizableKFs.back()->mnBALocalForKF  = 0;
        vpOptimizableKFs.back()->mnBAFixedForKF = pKF->mnId;
        lFixedKeyFrames.push_back(vpOptimizableKFs.back());
        vpOptimizableKFs.pop_back();
    }

    // Optimizable visual KFs
    const int maxCovKF = 0;
    for (int i = 0, iend = vpNeighsKFs.size(); i < iend; ++i)
    {
        if (lpOptVisKFs.size() > maxCovKF)
            break;
        MultiKeyFrame* pKFi = vpNeighsKFs[i];
        if(pKFi->mnBALocalForKF == pKF->mnId || pKFi->mnBAFixedForKF == pKF->mnId)
            continue;
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad() && pKFi->GetMap() == pCurrentMap)
        {
            lpOptVisKFs.push_back(pKFi);

            vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
            for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
            {
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF!=pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->mnId;
                        }
            }
        }
    }

    // Fixed KFs which are not covisible optimizable
    const int maxFixKF = 50;

    for (list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit!=lend; ++lit)
    {
        map<MultiKeyFrame*, vector<int>> observations = (*lit)->GetObservations();
        for (map<MultiKeyFrame*, vector<int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; ++mit)
        {
            MultiKeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                {
                    lFixedKeyFrames.push_back(pKFi);
                    break;
                }
            }
        }
         if(lFixedKeyFrames.size()>=maxFixKF)
            break;
    }

    bool bNonFixed = (lFixedKeyFrames.size() == 0);

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);

    if (bLarge)
    {
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e-2);
        optimizer.setAlgorithm(solver);
    }
    else
    {
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setUserLambdaInit(1e0);
        optimizer.setAlgorithm(solver);
    }

    // Set Local Temporal KeyFrame vertices
    N = vpOptimizableKFs.size();
    for (int i = 0; i < N; ++i)
    {
        MultiKeyFrame* pKFi = vpOptimizableKFs[i];
        
        VertexPoseVel* v = new VertexPoseVel(pKFi);
        v->setId(pKFi->mnId);
        v->setFixed(false);
        optimizer.addVertex(v);

        EdgeVelocity* ev = new EdgeVelocity();
        ev->setVertex(0, v);
        ev->setInformation(pKFi->mpGP->mQcInv.block<1, 1>(2, 2));
        optimizer.addEdge(ev);
    }

    // Set Local visual KeyFrame vertices
    for (list<MultiKeyFrame*>::iterator it = lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it != itEnd; ++it)
    {
        MultiKeyFrame* pKFi = *it;
        VertexPoseVel* v = new VertexPoseVel(pKFi);
        v->setId(pKFi->mnId);
        v->setFixed(false);
        optimizer.addVertex(v);
    }

    // Set Fixed KeyFrame vertices
    for (list<MultiKeyFrame*>::iterator it = lFixedKeyFrames.begin(), itEnd = lFixedKeyFrames.end(); it != itEnd; ++it)
    {
        MultiKeyFrame* pKFi = *it;
        VertexPoseVel* v = new VertexPoseVel(pKFi);
        v->setId(pKFi->mnId);
        v->setFixed(true);
        optimizer.addVertex(v);
    }

    

    // Create GP constraints
    for (int i = N-1; i > 0; --i)
    {
        MultiKeyFrame* pKFi = vpOptimizableKFs[i];
        MultiKeyFrame* pKFip1 = vpOptimizableKFs[i-1];

        EdgeGaussianPrior* e = new EdgeGaussianPrior();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFip1->mnId)));
        e->setInformation(pKFi->mpGP->QiInv(pKFip1->mTimeStamp - pKFi->mTimeStamp));
        // g2o::RobustKernel* rkp = new g2o::RobustKernelHuber;
        // e->setRobustKernel(rkp);
        // rkp->setDelta(21.026);
        optimizer.addEdge(e);
    }

    // Create MapPoint vertices
    const int nExpectedSize = (N+lFixedKeyFrames.size())*lLocalMapPoints.size();

    // Mono
    vector<EdgeMono*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<MultiKeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // Mono GP in Non-KF obs
    vector<EdgeMonoGPExtrinsic*> vpEdgesMonoGP;
    vpEdgesMonoGP.reserve(nExpectedSize);

    vector<MultiKeyFrame*> vpEdgeKFMonoGP;
    vpEdgeKFMonoGP.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMonoGP;
    vpMapPointEdgeMonoGP.reserve(nExpectedSize);

    vector<GPObs> vpGPObs;
    vpGPObs.reserve(nExpectedSize);
    
    // Mono GP in KF obs
    vector<EdgeMonoGPExtrinsic*> vpEdgesMonoGPKF;
    vpEdgesMonoGPKF.reserve(nExpectedSize);

    vector<MultiKeyFrame*> vpEdgeKFMonoGPKF;
    vpEdgeKFMonoGPKF.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMonoGPKF;
    vpMapPointEdgeMonoGPKF.reserve(nExpectedSize);

    vector<int> vpCamEdgeMonoGPKF;
    vpCamEdgeMonoGPKF.reserve(nExpectedSize);

    //Stereo GP
    vector<EdgeStereoGP*> vpEdgesStereoGP;
    vpEdgesStereoGP.reserve(nExpectedSize);

    vector<MultiKeyFrame*> vpEdgeKFStereoGP;
    vpEdgeKFStereoGP.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereoGP;
    vpMapPointEdgeStereoGP.reserve(nExpectedSize);

    vector<GPObs> vpGPObsStereo;
    vpGPObsStereo.reserve(nExpectedSize);

    //Stereo
    vector<EdgeStereo*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<MultiKeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float chi2Mono2 = 5.991;
    const float thHuberStereo = sqrt(7.815);
    const float chi2Stereo2 = 7.815;

    const unsigned long iniMPid = maxKFid*5;

    // Set Extrinsics vertices
    for (int c = 0; c < pKF->nCamera - 1; ++c)
    {
        VertexExtrinsic* v = new VertexExtrinsic(MultiKeyFrame::mTbc[c].cast<double>());
        v->setId(iniMPid + c + 1);
        v->setFixed(true);
        optimizer.addVertex(v);

        EdgeExtrinsicPrior* e = new EdgeExtrinsicPrior(MultiFrame::mRbc_ini[c].cast<double>());
        e->setVertex(0, v);
        e->setInformation(MultiFrame::mRbc_ini_cov[c]);

        optimizer.addEdge(e);
    }

    // map<int, int> mVisEdges;

    // for (int i = 0; i < N; ++i)
    // {
    //     MultiKeyFrame* pKFi = vpOptimizableKFs[i];
    //     mVisEdges[pKFi->mnId] = 0;
    // }

    // for(list<MultiKeyFrame*>::iterator lit=lFixedKeyFrames.begin(), lend=lFixedKeyFrames.end(); lit!=lend; lit++)
    // {
    //     mVisEdges[(*lit)->mnId] = 0;
    // }
    //
    vector<int> cam_obs(pKF->nCamera, 0);

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->GetWorldPos().cast<double>());

        unsigned long id = pMP->mnId + iniMPid + 5 + pKF->nCamera;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        // 添加关键帧的约束和非关键帧的约观测约束
        const map<MultiKeyFrame*, vector<int>> observations = pMP->GetObservations();
        const unordered_multimap<MultiKeyFrame*, GPObs> observationsGP = pMP->GetGPObservations();

        for (int i = vpOptimizableKFs.size()-1; i > 0; --i)
        {
            MultiKeyFrame* pKFi = vpOptimizableKFs[i];
            if (!pKFi->mNextKF)
                continue;
            auto mitrange = observationsGP.equal_range(pKFi);
            for (auto mit = mitrange.first, mend = mitrange.second; mit != mend; ++mit)
            {
                const GPObs GPobs = mit->second;
                if (GPobs.ur >= 0)
                {
                    Eigen::Vector3d obs;
                    const cv::KeyPoint& kpUn = GPobs.obs;

                    obs << kpUn.pt.x, kpUn.pt.y, GPobs.ur;

                    EdgeStereoGP* e = new EdgeStereoGP(GPobs.cam, GPobs.time, pKFi->mpGP);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mNextKF->mnId)));
                    e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setMeasurement(obs);

                    // Add here uncertenty
                    const float unc2 = pKFi->mvpCamera[GPobs.cam]->uncertainty2(obs.head(2));

                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    optimizer.addEdge(e);
                    vpEdgesStereoGP.push_back(e);
                    vpEdgeKFStereoGP.push_back(pKFi);
                    vpMapPointEdgeStereoGP.push_back(pMP);
                    vpGPObsStereo.push_back(GPobs);
                }
                else
                {
                    Eigen::Vector2d obs;
                    const cv::KeyPoint& kpUn = GPobs.obs;

                    obs << kpUn.pt.x, kpUn.pt.y;

                    EdgeMonoGPExtrinsic* e = new EdgeMonoGPExtrinsic(GPobs.cam, GPobs.time, pKFi->mpGP);
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mNextKF->mnId)));
                    e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    cout << "id: " << id << endl;
                    e->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(iniMPid + GPobs.cam + 1)));
                    e->setMeasurement(obs);

                    // Add here uncertenty
                    const float unc2 = pKFi->mvpCamera[GPobs.cam]->uncertainty2(obs);

                    const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    optimizer.addEdge(e);
                    vpEdgesMonoGP.push_back(e);
                    vpEdgeKFMonoGP.push_back(pKFi);
                    vpMapPointEdgeMonoGP.push_back(pMP);
                    vpGPObs.push_back(GPobs);
                }

            }
        }


        for (auto mit = observations.begin(), mend = observations.end(); mit != mend; ++mit)
        {
            MultiKeyFrame* pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                continue;

            const vector<int>& idxs = mit->second;
            int c = 0, ncam = 0;
            for (c = 0, ncam = idxs.size(); c < ncam-1; ++c)
            {
                const int idx = idxs[c];
                if (idx < 0) continue;

                MultiKeyFrame* pKFprev = pKFi->mPrevKF;

                if (!pKFprev) continue;
                else if (pKFprev->mnBALocalForKF != pKF->mnId && pKFprev->mnBAFixedForKF != pKF->mnId)
                    continue;

                const cv::KeyPoint& kpUn = pKFi->mvKeysUn[idx];
                Eigen::Vector2d obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                EdgeMonoGPExtrinsic* e = new EdgeMonoGPExtrinsic(c, pKFi->mvTimeStamps[c], pKFi->mpGP);
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFprev->mnId)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(iniMPid + c + 1)));
                e->setMeasurement(obs);
                cam_obs[c]++;

                // Add here uncertenty
                const float unc2 = pKFi->mvpCamera[c]->uncertainty2(obs);

                const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);

                optimizer.addEdge(e);
                vpEdgesMonoGPKF.push_back(e);
                vpEdgeKFMonoGPKF.push_back(pKFi);
                vpMapPointEdgeMonoGPKF.push_back(pMP);
                vpCamEdgeMonoGPKF.push_back(c);
            }

            c =  ncam - 1;
            const int idx = idxs[c];
            if (idx < 0) continue;
            
            cv::KeyPoint kpUn = pKFi->mvKeysUn[idx];
            const float kp_ur = pKFi->mvuRight[pKFi->mmpGlobalToLocalID[idx]];

            if (kp_ur < 0)
            {
                Eigen::Vector2d obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                EdgeMono* e = new EdgeMono();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setMeasurement(obs);

                // Add here uncertenty
                const float unc2 = pKFi->mvpCamera[c]->uncertainty2(obs);

                const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);

                optimizer.addEdge(e);
                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKFi);
                vpMapPointEdgeMono.push_back(pMP);
            }
            else
            {
                Eigen::Vector3d obs;
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                EdgeStereo* e = new EdgeStereo();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setMeasurement(obs);

                // Add here uncertenty
                const float unc2 = pKFi->mvpCamera[c]->uncertainty2(obs.head(2));

                const float invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] / unc2;
                e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberStereo);

                optimizer.addEdge(e);
                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKFi);
                vpMapPointEdgeStereo.push_back(pMP);
            }
        }

    }

    //cout << "Total map points: " << lLocalMapPoints.size() << endl;
    // for(map<int,int>::iterator mit=mVisEdges.begin(), mend=mVisEdges.end(); mit!=mend; mit++)
    // {
    //     assert(mit->second>=3);
    // }
    int opt_it1 = bExtrinsic ? 10 : 10;
    int opt_it2 = opt_it;

    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    float err = optimizer.activeRobustChi2();
    optimizer.optimize(opt_it1);

    int extrin_thresh = 50;

    if (bExtrinsic)
    {
        for (int i = 0; i < pKF->nCamera - 1; ++i)
        {
            if (cam_obs[i] < extrin_thresh)
                continue;
            VertexExtrinsic* v = static_cast<VertexExtrinsic*>(optimizer.vertex(iniMPid + i + 1));
            v->setFixed(false);
        }
            optimizer.initializeOptimization();
            optimizer.computeActiveErrors();
            optimizer.optimize(opt_it2);
    }

    /*if (vpEdgesMonoGPKF.size() > 1)*/
    /*     {*/
    /*         std::cout << "monoGP size: " << vpEdgesMonoGPKF.size() << std::endl;*/
    /*         EdgeMonoGPExtrinsic* e = vpEdgesMonoGPKF[0];*/
    /*         std::cout << "cam: " << e->cam_idx << std::endl;*/
    /*         const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(e->vertex(0));*/
    /*         const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(e->vertex(1));*/
    /*         std::cout << std::fixed << "point time: " << e->t << " prev time: " << v1->estimate().time << " next time: " << v2->estimate().time << std::endl;*/
    /*         saveMatrix("/home/ljj/numerical.csv", e->GetJacobian());*/
    /*         terminate();*/
    /*     }*/
    float err_end = optimizer.activeRobustChi2();
    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);
    
    // Check inlier observations
    vector<tuple<MultiKeyFrame*, MapPoint*, GPObs>> vToEraseNKF;
    vector<tuple<MultiKeyFrame*, MapPoint*, int>> vToErase;
    vToEraseNKF.reserve(vpEdgesMonoGP.size() + vpEdgesStereoGP.size());
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size() + vpEdgesMonoGPKF.size());

    for (size_t i = 0, iend = vpEdgesMonoGP.size(); i < iend; ++i)
    {
        EdgeMonoGPExtrinsic* e = vpEdgesMonoGP[i];
        MapPoint* pMP = vpMapPointEdgeMonoGP[i];
        const GPObs GPobs = vpGPObs[i];
        bool bClose = pMP->mvTrackDepth[GPobs.cam] < 10.f;

        if (pMP->isBad())
            continue;
        
        if ((e->chi2() > chi2Mono2 && !bClose) || (e->chi2() > 1.5f * chi2Mono2 && bClose) || !e->isDepthPositive())
        {
            MultiKeyFrame* pKFi = vpEdgeKFMonoGP[i];    
            vToEraseNKF.push_back(make_tuple(pKFi, pMP, GPobs));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereoGP.size(); i < iend; ++i)
    {
        EdgeStereoGP* e = vpEdgesStereoGP[i];
        MapPoint* pMP = vpMapPointEdgeStereoGP[i];
        const GPObs GPobs = vpGPObsStereo[i];

        if (pMP->isBad())
            continue;
        
        if (e->chi2() > chi2Stereo2)
        {
            MultiKeyFrame* pKFi = vpEdgeKFStereoGP[i];
            vToEraseNKF.push_back(make_tuple(pKFi, pMP, GPobs));
        }
    }

    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; ++i)
    {
        EdgeMono* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];
        MultiKeyFrame* pKFi = vpEdgeKFMono[i];
        int c = pKFi->nCamera-1;

        bool bClose = pMP->mvTrackDepth[c] < 10.f;

        if (pMP->isBad())
            continue;

        if ((e->chi2() > chi2Mono2 && !bClose) || (e->chi2() > 1.5f * chi2Mono2 && bClose) || !e->isDepthPositive())
        {
            vToErase.push_back(make_tuple(pKFi, pMP, c));
        }
    }
    
    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; ++i)
    {
        EdgeStereo* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];
        MultiKeyFrame* pKFi = vpEdgeKFStereo[i];
        int c = pKFi->nCamera-1;

        if (pMP->isBad())
            continue;

        if (e->chi2() > chi2Stereo2)
        {
            vToErase.push_back(make_tuple(pKFi, pMP, c));
        }
    }


    for (size_t i = 0, iend = vpEdgesMonoGPKF.size(); i < iend; ++i)
    {
        EdgeMonoGPExtrinsic* e = vpEdgesMonoGPKF[i];
        MapPoint* pMP = vpMapPointEdgeMonoGPKF[i];
        MultiKeyFrame* pKFi = vpEdgeKFMonoGPKF[i];
        int c = vpCamEdgeMonoGPKF[i];

        bool bClose = pMP->mvTrackDepth[c] < 10.f;

        if (pMP->isBad())
            continue;

        if ((e->chi2() > chi2Mono2 && !bClose) || (e->chi2() > 1.5f * chi2Mono2 && bClose) || !e->isDepthPositive())
        {
            --cam_obs[c];
            vToErase.push_back(make_tuple(pKFi, pMP, c));
        }
    }

    // Get Map Mutex and erase outliers
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // TODO: Some convergence problems have been detected here
    if((2*err < err_end || isnan(err) || isnan(err_end)) && !bLarge) //bGN)
    {
        cout << "FAIL LOCAL-GP BA!!!!" << endl;
        return;
    }

    if (!vToEraseNKF.empty())
    {
        for (size_t i = 0, iend = vToEraseNKF.size(); i < iend; ++i)
        {
            MultiKeyFrame* pKFi = std::get<0>(vToEraseNKF[i]);
            MapPoint* pMP = std::get<1>(vToEraseNKF[i]);
            const GPObs GPobs = std::get<2>(vToEraseNKF[i]);
            pMP->EraseGPObservation(pKFi, GPobs);
        }
    }

    if (!vToErase.empty())
    {
        for (size_t i = 0, iend = vToErase.size(); i < iend; ++i)
        {
            MultiKeyFrame* pKFi = std::get<0>(vToErase[i]);
            MapPoint* pMP = std::get<1>(vToErase[i]);
            int c = std::get<2>(vToErase[i]);
            
            pKFi->EraseMapPointMatch(pMP, c);
            pMP->EraseObservation(pKFi, c);
        }
    }

    for (auto lit = lFixedKeyFrames.begin(), lend = lFixedKeyFrames.end(); lit != lend; lit++)
        (*lit)->mnBAFixedForKF = 0;

    N = vpOptimizableKFs.size();

    // Recover optimized data
    // Local temporal Keyframes
    for (int i = N-1; i >= 0; --i)
    {
        MultiKeyFrame* pKFi = vpOptimizableKFs[i];
        
        VertexPoseVel* v = static_cast<VertexPoseVel*>(optimizer.vertex(pKFi->mnId));
        pKFi->SetPose(v->estimate().Twb.inverse().cast<float>());
        pKFi->mnBALocalForKF = 0;
    }


    // Local visual KeyFrame
    for (auto it = lpOptVisKFs.begin(), itEnd = lpOptVisKFs.end(); it != itEnd; it++)
    {
        MultiKeyFrame* pKFi = *it;
        VertexPoseVel* v = static_cast<VertexPoseVel*>(optimizer.vertex(pKFi->mnId));
        pKFi->SetPose(v->estimate().Twb.inverse().cast<float>());
        pKFi->mnBALocalForKF = 0;
    }

    //Points
    for (auto lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; ++lit)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + iniMPid + 5 + pKF->nCamera));
        pMP->SetWorldPos(vPoint->estimate().cast<float>());
        pMP->UpdateNormalAndDepth();
    }

    for (int c = 0; c < pKF->nCamera - 1; ++c)
    {
        if (cam_obs[c] < extrin_thresh)
        {
            continue;
        }
        VertexExtrinsic* v = static_cast<VertexExtrinsic*>(optimizer.vertex(iniMPid + c + 1));
        MultiKeyFrame::mTbc[c] = v->estimate().cast<float>();
        MultiFrame::mTbc[c] = v->estimate().cast<float>();
    }

    pMap->IncreaseChangeIndex();

}

void Optimizer::OptimizeEssentialGraph(Map* pMap, MultiKeyFrame* pLoopKF, MultiKeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<MultiKeyFrame *, set<MultiKeyFrame *> > &LoopConnections, const bool &bFixScale)
{   
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<MultiKeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    vector<Eigen::Vector3d> vZvectors(nMaxKFid+1); // For debugging
    Eigen::Vector3d z_vec;
    z_vec << 0.0, 0.0, 1.0;

    const int minFeat = 100;

    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        MultiKeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Sophus::SE3d Tcw = pKF->GetPose().cast<double>();
            g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF->mnId==pMap->GetInitKFid())
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);
        vZvectors[nIDi]=vScw[nIDi].rotation()*z_vec; // For debugging

        vpVertices[nIDi]=VSim3;
    }


    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    int count_loop = 0;
    for(map<MultiKeyFrame *, set<MultiKeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        MultiKeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<MultiKeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<MultiKeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);
            count_loop++;
            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        MultiKeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        MultiKeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);
            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        // GetLoopEdges还没有加入这次的loopedge,所以不会加入本次回环的边，因此需要使用校正前的相对位姿作为观测
        const set<MultiKeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<MultiKeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            MultiKeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<MultiKeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<MultiKeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            MultiKeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) /*&& !sLoopEdges.count(pKFn)*/)
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }

        // TODO: 位姿图优化加入高斯过程
        // if(pKF->bImu && pKF->mPrevKF)
        // {
        //     g2o::Sim3 Spw;
        //     LoopClosing::KeyFrameAndPose::const_iterator itp = NonCorrectedSim3.find(pKF->mPrevKF);
        //     if(itp!=NonCorrectedSim3.end())
        //         Spw = itp->second;
        //     else
        //         Spw = vScw[pKF->mPrevKF->mnId];

        //     g2o::Sim3 Spi = Spw * Swi;
        //     g2o::EdgeSim3* ep = new g2o::EdgeSim3();
        //     ep->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mPrevKF->mnId)));
        //     ep->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
        //     ep->setMeasurement(Spi);
        //     ep->information() = matLambda;
        //     optimizer.addEdge(ep);
        // }
    }


    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    optimizer.optimize(20);
    optimizer.computeActiveErrors();
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        MultiKeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        double s = CorrectedSiw.scale();

        Sophus::SE3f Tiw(CorrectedSiw.rotation().cast<float>(), CorrectedSiw.translation().cast<float>() / s);
        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            MultiKeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        Eigen::Matrix<double,3,1> eigP3Dw = pMP->GetWorldPos().cast<double>();
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));
        pMP->SetWorldPos(eigCorrectedP3Dw.cast<float>());

        pMP->UpdateNormalAndDepth();
    }

    // TODO Check this changeindex
    pMap->IncreaseChangeIndex();
}

void Optimizer::OptimizeEssentialGraph(MultiKeyFrame* pCurKF, vector<MultiKeyFrame*> &vpFixedKFs, vector<MultiKeyFrame*> &vpFixedCorrectedKFs,
                                       vector<MultiKeyFrame*> &vpNonFixedKFs, vector<MapPoint*> &vpNonCorrectedMPs)
{
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedKFs.size()) + " KFs fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpFixedCorrectedKFs.size()) + " KFs fixed in the old map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonFixedKFs.size()) + " KFs non-fixed in the merged map", Verbose::VERBOSITY_DEBUG);
    Verbose::PrintMess("Opt_Essential: There are " + to_string(vpNonCorrectedMPs.size()) + " MPs non-corrected in the merged map", Verbose::VERBOSITY_DEBUG);

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    Map* pMap = pCurKF->GetMap();
    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    vector<bool> vpGoodPose(nMaxKFid+1);
    vector<bool> vpBadPose(nMaxKFid+1);

    const int minFeat = 100;

    for(MultiKeyFrame* pKFi : vpFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vCorrectedSwc[nIDi]=Siw.inverse();
        VSim3->setEstimate(Siw);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = true;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = false;
    }
    Verbose::PrintMess("Opt_Essential: vpFixedKFs loaded", Verbose::VERBOSITY_DEBUG);

    set<unsigned long> sIdKF;
    for(MultiKeyFrame* pKFi : vpFixedCorrectedKFs)
    {
        if(pKFi->isBad())
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKFi->mnId;

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vCorrectedSwc[nIDi]=Siw.inverse();
        VSim3->setEstimate(Siw);

        Sophus::SE3d Tcw_bef = pKFi->mTBwBefMerge.cast<double>();
        vScw[nIDi] = g2o::Sim3(Tcw_bef.unit_quaternion(),Tcw_bef.translation(),1.0);

        VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = true;
        vpBadPose[nIDi] = true;
    }

    for(MultiKeyFrame* pKFi : vpNonFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        if(sIdKF.count(nIDi)) // It has already added in the corrected merge KFs
            continue;

        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        Sophus::SE3d Tcw = pKFi->GetPose().cast<double>();
        g2o::Sim3 Siw(Tcw.unit_quaternion(),Tcw.translation(),1.0);

        vScw[nIDi] = Siw;
        VSim3->setEstimate(Siw);

        VSim3->setFixed(false);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;

        sIdKF.insert(nIDi);

        vpGoodPose[nIDi] = false;
        vpBadPose[nIDi] = true;
    }

    vector<MultiKeyFrame*> vpKFs;
    vpKFs.reserve(vpFixedKFs.size() + vpFixedCorrectedKFs.size() + vpNonFixedKFs.size());
    vpKFs.insert(vpKFs.end(),vpFixedKFs.begin(),vpFixedKFs.end());
    vpKFs.insert(vpKFs.end(),vpFixedCorrectedKFs.begin(),vpFixedCorrectedKFs.end());
    vpKFs.insert(vpKFs.end(),vpNonFixedKFs.begin(),vpNonFixedKFs.end());
    set<MultiKeyFrame*> spKFs(vpKFs.begin(), vpKFs.end());

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    for(MultiKeyFrame* pKFi : vpKFs)
    {
        int num_connections = 0;
        const int nIDi = pKFi->mnId;

        g2o::Sim3 correctedSwi;
        g2o::Sim3 Swi;

        if(vpGoodPose[nIDi])
            correctedSwi = vCorrectedSwc[nIDi];
        if(vpBadPose[nIDi])
            Swi = vScw[nIDi].inverse();

        MultiKeyFrame* pParentKFi = pKFi->GetParent();

        // Spanning tree edge
        if(pParentKFi && spKFs.find(pParentKFi) != spKFs.end())
        {
            int nIDj = pParentKFi->mnId;

            g2o::Sim3 Sjw;
            bool bHasRelation = false;

            if(vpGoodPose[nIDi] && vpGoodPose[nIDj])
            {
                Sjw = vCorrectedSwc[nIDj].inverse();
                bHasRelation = true;
            }
            else if(vpBadPose[nIDi] && vpBadPose[nIDj])
            {
                Sjw = vScw[nIDj];
                bHasRelation = true;
            }

            if(bHasRelation)
            {
                g2o::Sim3 Sji = Sjw * Swi;

                g2o::EdgeSim3* e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                e->setMeasurement(Sji);

                e->information() = matLambda;
                optimizer.addEdge(e);
                num_connections++;
            }

        }

        // Loop edges
        const set<MultiKeyFrame*> sLoopEdges = pKFi->GetLoopEdges();
        for(set<MultiKeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            MultiKeyFrame* pLKF = *sit;
            if(spKFs.find(pLKF) != spKFs.end() && pLKF->mnId<pKFi->mnId)
            {
                g2o::Sim3 Slw;
                bool bHasRelation = false;

                if(vpGoodPose[nIDi] && vpGoodPose[pLKF->mnId])
                {
                    Slw = vCorrectedSwc[pLKF->mnId].inverse();
                    bHasRelation = true;
                }
                else if(vpBadPose[nIDi] && vpBadPose[pLKF->mnId])
                {
                    Slw = vScw[pLKF->mnId];
                    bHasRelation = true;
                }


                if(bHasRelation)
                {
                    g2o::Sim3 Sli = Slw * Swi;
                    g2o::EdgeSim3* el = new g2o::EdgeSim3();
                    el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                    el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    el->setMeasurement(Sli);
                    el->information() = matLambda;
                    optimizer.addEdge(el);
                    num_connections++;
                }
            }
        }

        // Covisibility graph edges
        const vector<MultiKeyFrame*> vpConnectedKFs = pKFi->GetCovisiblesByWeight(minFeat);
        for(vector<MultiKeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            MultiKeyFrame* pKFn = *vit;
            if(pKFn && pKFn!=pParentKFi && !pKFi->hasChild(pKFn) && !sLoopEdges.count(pKFn) && spKFs.find(pKFn) != spKFs.end())
            {
                if(!pKFn->isBad() && pKFn->mnId<pKFi->mnId)
                {

                    g2o::Sim3 Snw =  vScw[pKFn->mnId];
                    bool bHasRelation = false;

                    if(vpGoodPose[nIDi] && vpGoodPose[pKFn->mnId])
                    {
                        Snw = vCorrectedSwc[pKFn->mnId].inverse();
                        bHasRelation = true;
                    }
                    else if(vpBadPose[nIDi] && vpBadPose[pKFn->mnId])
                    {
                        Snw = vScw[pKFn->mnId];
                        bHasRelation = true;
                    }

                    if(bHasRelation)
                    {
                        g2o::Sim3 Sni = Snw * Swi;

                        g2o::EdgeSim3* en = new g2o::EdgeSim3();
                        en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                        en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                        en->setMeasurement(Sni);
                        en->information() = matLambda;
                        optimizer.addEdge(en);
                        num_connections++;
                    }
                }
            }
        }

        if(num_connections == 0 )
        {
            Verbose::PrintMess("Opt_Essential: KF " + to_string(pKFi->mnId) + " has 0 connections", Verbose::VERBOSITY_DEBUG);
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(MultiKeyFrame* pKFi : vpNonFixedKFs)
    {
        if(pKFi->isBad())
            continue;

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        double s = CorrectedSiw.scale();
        Sophus::SE3d Tiw(CorrectedSiw.rotation(),CorrectedSiw.translation() / s);

        pKFi->mTBwBefMerge = pKFi->GetPose();
        pKFi->mTwBBefMerge = pKFi->GetPoseInverse();
        pKFi->SetPose(Tiw.cast<float>());
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(MapPoint* pMPi : vpNonCorrectedMPs)
    {
        if(pMPi->isBad())
            continue;

        MultiKeyFrame* pRefKF = pMPi->GetReferenceKeyFrame();
        while(pRefKF->isBad())
        {
            if(!pRefKF)
            {
                Verbose::PrintMess("MP " + to_string(pMPi->mnId) + " without a valid reference KF", Verbose::VERBOSITY_DEBUG);
                break;
            }

            pMPi->EraseObservation(pRefKF);
            pRefKF = pMPi->GetReferenceKeyFrame();
        }

        if(vpBadPose[pRefKF->mnId])
        {
            Sophus::SE3f TNonCorrectedwr = pRefKF->mTwBBefMerge;
            Sophus::SE3f Twr = pRefKF->GetPoseInverse();

            Eigen::Vector3f eigCorrectedP3Dw = Twr * TNonCorrectedwr.inverse() * pMPi->GetWorldPos();
            pMPi->SetWorldPos(eigCorrectedP3Dw);

            pMPi->UpdateNormalAndDepth();
        }
        else
        {
            cout << "ERROR: MapPoint has a reference KF from another map" << endl;
        }

    }
}

int Optimizer::OptimizeSim3(MultiKeyFrame *pKF1, MultiKeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2,
                            const bool bFixScale, Eigen::Matrix<double,7,7> &mAcumHessian, const bool bAllPoints)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Camera poses
    // std::cout << "OptimizeSim3: Camera poses" << std::endl;
    vector<Eigen::Matrix3f> vR1w;
    vector<Eigen::Vector3f> vt1w;
    vector<Eigen::Matrix3f> vR2w;
    vector<Eigen::Vector3f> vt2w;
    
    vector<g2o::SE3Quat> vTc1b;
    vector<g2o::SE3Quat> vTc2b;

    const vector<Sophus::SE3f> &vT1w = pKF1->GetCameraPoses();
    const vector<Sophus::SE3f> &vT2w = pKF2->GetCameraPoses();
    const Sophus::SE3f &T1bw = pKF1->GetPose();
    const Sophus::SE3f &T2bw = pKF2->GetPose();
    // std::cout << "T1 size: " << vT1w.size() << " T2 size: " << vT2w.size() << std::endl;
    for (int i = 0; i < vT1w.size(); ++i)
    {
        vR1w.push_back(vT1w[i].rotationMatrix());
        vt1w.push_back(vT1w[i].translation());
        vR2w.push_back(vT2w[i].rotationMatrix());
        vt2w.push_back(vT2w[i].translation());
        Sophus::SE3d dT1 = (vT1w[i] * T1bw.inverse()).cast<double>();
        Sophus::SE3d dT2 = (vT2w[i] * T2bw.inverse()).cast<double>();
        // vTc1b.emplace_back(dT1.rotationMatrix(), dT1.translation());
        // vTc2b.emplace_back(dT2.rotationMatrix(), dT2.translation());
        vTc1b.push_back(g2o::SE3Quat(dT1.rotationMatrix(), dT1.translation()));
        vTc2b.push_back(g2o::SE3Quat(dT2.rotationMatrix(), dT2.translation()));
    }

    // std::cout << "OptimizeSim3: Set vertices" << std::endl;
    // Set Sim3 vertex
    ORB_SLAM3::VertexSim3Expmap * vSim3 = new ORB_SLAM3::VertexSim3Expmap();
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->vpCamera1 = pKF1->mvpCamera;
    vSim3->vpCamera2 = pKF2->mvpCamera;
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<ORB_SLAM3::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<ORB_SLAM3::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;
    vector<bool> vbIsInKF2;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);
    vbIsInKF2.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;
    int nBadMPs = 0;
    int nInKF2 = 0;
    int nOutKF2 = 0;
    int nMatchWithoutMP = 0;

    vector<int> vIdsOnlyInKF2;

    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        int cam1 = pKF1->mmpKeyToCam[i];
        int cam2 = 0;

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        int i2 = -1;
        vector<int> i2Indexs = pMP2->GetIndexInKeyFrame(pKF2);

        for (int j = 0; j < i2Indexs.size(); ++j)
        {
            if (i2Indexs[j] >= 0)
            {
                cam2 = j;
                i2 = i2Indexs[j];
                break;
            }
        }

        if (i2 < 0) continue;

        Eigen::Vector3f P3D1c;
        Eigen::Vector3f P3D2c;

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad())
            {
                // std::cout << "OptimizeSim3: MP1 and MP2 are not bad" << std::endl;
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D1w = pMP1->GetWorldPos();
                P3D1c = vR1w[cam1]*P3D1w + vt1w[cam1];
                vPoint1->setEstimate(P3D1c.cast<double>());
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
                P3D2c = vR2w[cam2]*P3D2w + vt2w[cam2];
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
            {
                nBadMPs++;
                continue;
            }
        }
        else
        {
            nMatchWithoutMP++;

            //TODO The 3D position in KF1 doesn't exist
            // std::cout << "OptimizeSim3: MP1 or MP2 is bad" << std::endl;
            if(!pMP2->isBad())
            {
                // std::cout << "OptimizeSim3: MP2 is not bad" << std::endl;
                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                Eigen::Vector3f P3D2w = pMP2->GetWorldPos();
                P3D2c = vR2w[cam2] * P3D2w + vt2w[cam2];
                vPoint2->setEstimate(P3D2c.cast<double>());
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);

                vIdsOnlyInKF2.push_back(id2);
            }
            continue;
        }

        if(i2<0 && !bAllPoints)
        {
            // Verbose::PrintMess("    Remove point -> i2: " + to_string(i2) + "; bAllPoints: " + to_string(bAllPoints), Verbose::VERBOSITY_DEBUG);
            continue;
        }

        if(P3D2c(2) < 0)
        {
            // Verbose::PrintMess("Sim3: Z coordinate is negative", Verbose::VERBOSITY_DEBUG);
            continue;
        }

        nCorrespondences++;

        // Set edge x1 = S12*X2
        // std::cout << "OptimizeSim3: Set edge x1 = S12*X2" << std::endl;
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = new ORB_SLAM3::EdgeSim3ProjectXYZ(cam1, cam2, vTc1b, vTc2b);

        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        // std::cout << "OptimizeSim3: Set edge x2 = S21*X1" << std::endl;
        Eigen::Matrix<double,2,1> obs2;
        cv::KeyPoint kpUn2;
        bool inKF2;
        if(i2 >= 0)
        {
            kpUn2 = pKF2->mvKeysUn[i2];
            obs2 << kpUn2.pt.x, kpUn2.pt.y;
            inKF2 = true;

            nInKF2++;
        }
        else
        {
            float invz = 1/P3D2c(2);
            float x = P3D2c(0)*invz;
            float y = P3D2c(1)*invz;

            obs2 << x, y;
            kpUn2 = cv::KeyPoint(cv::Point2f(x, y), pMP2->mvnTrackScaleLevel[cam2]);

            inKF2 = false;
            nOutKF2++;
        }

        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = new ORB_SLAM3::EdgeInverseSim3ProjectXYZ(cam1, cam2, vTc1b, vTc2b);

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);

        vbIsInKF2.push_back(inKF2);
    }

    // Optimize!
    // std::cout << "OptimizeSim3: Optimize!" << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad=0;
    int nBadOutKF2 = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<ORB_SLAM3::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<ORB_SLAM3::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;

            if(!vbIsInKF2[i])
            {
                nBadOutKF2++;
            }
            continue;
        }

        //Check if remove the robust adjustment improve the result
        e12->setRobustKernel(0);
        e21->setRobustKernel(0);
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers
    // std::cout << "OptimizeSim3: Optimize again only with inliers" << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    mAcumHessian = Eigen::MatrixXd::Zero(7, 7);
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        ORB_SLAM3::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        ORB_SLAM3::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        e12->computeError();
        e21->computeError();

        if(e12->chi2()>th2 || e21->chi2()>th2){
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else{
            nIn++;
        }
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}

int Optimizer::OptimizeVel(MultiFrame* pF1, MultiFrame* pF2, std::vector<int>& indices, std::unordered_set<int>& samples, Eigen::Matrix<double, 6, 1>& Vel, std::vector<bool>& vbInliers, double threshold)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int inliers = 0;

    // Set Vel Vertex
    VertexVel* vVel = new VertexVel();
    vVel->setEstimate(pF1->GetVelocity().cast<double>());
    vVel->setId(0);
    vVel->setFixed(false);
    optimizer.addVertex(vVel);

    std::vector<EdgeVelReproj*> vpEdges;

    // Set Obs Edges
    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for (auto &index : indices)
        {
            MapPoint* pMP = pF1->mvpMapPoints[index];
            int cam = pF1->mmpKeyToCam[index];

            const cv::KeyPoint &kpUn = pF1->mvKeysUn[index];
            Eigen::Vector2d obs;
            obs << kpUn.pt.x, kpUn.pt.y;

            EdgeVelReproj* e = new EdgeVelReproj(pF2->GetPoseW().cast<double>(), 
                                pF1->mvTimeStamps[cam] - pF2->mTimeStamp,
                                pMP->GetWorldPos().cast<double>(), cam, pF1);

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs);
            e->setInformation(Eigen::Matrix2d::Identity() * pF1->mvInvLevelSigma2[kpUn.octave]);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(5.991);

            if (samples.count(index))
                e->setLevel(0);
            else
                e->setLevel(1);

            optimizer.addEdge(e);
            vpEdges.push_back(e);
        }
    }

    optimizer.initializeOptimization(0);
    optimizer.optimize(40);

    for (int i = 0; i < vpEdges.size(); ++i)
    {
        vpEdges[i]->computeError();

        const float chi2 = vpEdges[i]->chi2();

        if (vpEdges[i]->error().norm() <= threshold)
        {
            inliers++;
            vbInliers[i] = true;
        }
        else
        {
            vbInliers[i] = false;
        }
    }

    VertexVel* vVel_recov = static_cast<VertexVel*>(optimizer.vertex(0));
    Vel = vVel_recov->estimate();

    return inliers;

}



} //namespace ORB_SLAM
