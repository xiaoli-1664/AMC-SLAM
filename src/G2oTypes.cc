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

#include "G2oTypes.h"
#include "ImuTypes.h"
#include "Converter.h"
namespace ORB_SLAM3
{

PoseVelocity::PoseVelocity(MultiKeyFrame* pKF) {
    Twb = pKF->GetPoseInverse().cast<double>();
    Vel = pKF->GetVelocity().cast<double>();
    time = pKF->mTimeStamp;
    bf = pKF->mbf;
    vpCameras = pKF->mvpCamera;
}

PoseVelocity::PoseVelocity(MultiFrame* pF) {
    Twb = pF->GetPoseW().cast<double>();
    Vel = pF->GetVelocity().cast<double>();
    time = pF->mTimeStamp;
    bf = pF->mbf;
    vpCameras = pF->mvpCamera;
}

void PoseVelocity::Update(const double* pu) {
    Eigen::Map<const Vector6d> pose_update(pu);
    Eigen::Map<const Vector6d> vel_update(pu+6);
    Twb = Twb * Sophus::SE3d::exp(pose_update);
    Vel += vel_update;
}

Eigen::Vector3d PoseVelocity::ProjectStereo(const Eigen::Vector3d& Xw) const {
    Eigen::Vector3d Xc = (Twb * MultiKeyFrame::mTbc.back().cast<double>()).inverse() * Xw;
    Eigen::Vector3d xc;
    double invZ = 1/Xc(2);
    xc.head(2) = vpCameras.back()->project(Xc);
    xc(2) = xc(0) - bf * invZ;
    return xc;
}

Eigen::Vector2d PoseVelocity::Project(const Eigen::Vector3d& Xw) const {
    Eigen::Vector3d Xc = (Twb * MultiKeyFrame::mTbc.back().cast<double>()).inverse() * Xw;
    Eigen::Vector2d xc;
    double invZ = 1/Xc(2);
    xc = vpCameras.back()->project(Xc);
    return xc;
}

bool PoseVelocity::isDepthPositive(const Eigen::Vector3d& Xw, int cam_idx) const
{
    Eigen::Vector3d Xc = (Twb * MultiKeyFrame::mTbc[cam_idx].cast<double>()).inverse() * Xw;
    return Xc(2) > 0;
}

bool PoseVelocity::isDepthPositive(const Eigen::Vector3d& Xw) const
{
    Eigen::Vector3d Xc = (Twb * MultiKeyFrame::mTbc.back().cast<double>()).inverse() * Xw;
    return Xc(2) > 0;
}

bool PoseVelocity::isDepthPositive(const Eigen::Vector3d& Xw, int cam_idx, Sophus::SE3d Tbc) const
{
    Eigen::Vector3d Xc = (Twb * Tbc).inverse() * Xw;
    return Xc(2) > 0;
}

// void EdgeGaussianPrior::linearizeOplus()
// {
//     const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
//     const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
//     Sophus::SE3d T = v2->estimate().Twb * v1->estimate().Twb.inverse();
//     Vector6d xi = T.log();
//     _jacobianOplusXi.block<6, 6>(0, 0) = -LeftJacobianPose3Inv(-xi);
//     std::vector<Eigen::Matrix<double, 6, 6>> J = jacobianNumercialDiff(LeftJacobianPose3Inv, T, v2->estimate().Vel);
//     _jacobianOplusXi.block<6, 6>(6, 0) = J[0];
//     _jacobianOplusXi.block<12, 6>(0, 6) << -(v2->estimate().time - v1->estimate().time)
//                                             * Eigen::Matrix<double, 6, 6>::Identity(), -Eigen::Matrix<double, 6, 6>::Identity();
//     Eigen::Matrix<double, 6, 6> lJxi = LeftJacobianPose3Inv(xi);
//     _jacobianOplusXj.block<6, 6>(0, 0) = lJxi;
//     _jacobianOplusXj.block<6, 6>(6, 0) = J[1];
//     _jacobianOplusXj.block<12, 6>(0, 6) << Eigen::Matrix<double, 6, 6>::Zero(), lJxi;
// }

void EdgeGaussianPrior::linearizeOplus()
{
    const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
    const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
    Sophus::SE3d T = v1->estimate().Twb.inverse() * v2->estimate().Twb;
    Vector6d xi = T.log();
    Eigen::Matrix<double ,6, 6> Jr_inv_T = RightJacobianPose3Inv(xi);

    Eigen::Matrix<double, 6, 6> ad_v2 = se3Adj(v2->estimate().Vel);
    // _jacobianOplusXi.block<6, 6>(0, 0) = -Jl_inv_T * T.Adj();
    _jacobianOplusXi.block<6, 6>(0, 0) = -Jr_inv_T * T.Adj().inverse();
    _jacobianOplusXi.block<6, 6>(6, 0) = -0.5 * ad_v2 * _jacobianOplusXi.block<6, 6>(0, 0);
    _jacobianOplusXi.block<12, 6>(0, 6) << -(v2->estimate().time - v1->estimate().time)
                                            * Eigen::Matrix<double, 6, 6>::Identity(), -Eigen::Matrix<double, 6, 6>::Identity();
    
    _jacobianOplusXj.block<6, 6>(0, 0) = Jr_inv_T;
    _jacobianOplusXj.block<6, 6>(6, 0) = -0.5 * ad_v2 * _jacobianOplusXj.block<6, 6>(0, 0);
    _jacobianOplusXj.block<12, 6>(0, 6) << Eigen::Matrix<double, 6, 6>::Zero(), Jr_inv_T;
}

void EdgeMonoOnlyPose::linearizeOplus()
{
    const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);

    const Sophus::SE3d Tcb = MultiKeyFrame::mTbc.back().cast<double>().inverse();
    const Eigen::Vector3d Xb = v1->estimate().Twb.inverse() * Xw;
    const Eigen::Vector3d Xc = Tcb * Xb;

    Eigen::Matrix<double, 2, 3> proj_jac;
    proj_jac = v1->estimate().vpCameras.back()->projectJac(Xc);

    Eigen::Matrix<double, 3, 6> SE3deriv;
    Eigen::Matrix<double, 3, 3> Rcb = Tcb.rotationMatrix();
    SE3deriv << -Rcb, Rcb * Skew(Xb);

    _jacobianOplusXi.block<2, 6>(0, 0) = -proj_jac * SE3deriv;
    _jacobianOplusXi.block<2, 6>(0, 6) = Eigen::Matrix<double, 2, 6>::Zero();
}

void EdgeStereoOnlyPose::linearizeOplus()
{
    const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);

    const Sophus::SE3d Tcb = MultiKeyFrame::mTbc.back().cast<double>().inverse();
    const Eigen::Vector3d Xb = v1->estimate().Twb.inverse() * Xw;
    const Eigen::Vector3d Xc = Tcb * Xb;
    const double bf = v1->estimate().bf;
    const double inv_z2 = 1.0 / (Xc(2) * Xc(2));

    Eigen::Matrix<double, 3, 3> proj_jac;
    proj_jac.block<2, 3>(0, 0) = v1->estimate().vpCameras.back()->projectJac(Xc);
    proj_jac.block<1, 3>(2, 0) = proj_jac.block<1, 3>(0, 0);
    proj_jac(2, 2) += bf * inv_z2;

    Eigen::Matrix<double, 3, 6> SE3deriv;
    Eigen::Matrix<double, 3, 3> Rcb = Tcb.rotationMatrix();
    SE3deriv << -Rcb, Rcb * Skew(Xb);

    _jacobianOplusXi.block<3, 6>(0, 0) = -proj_jac * SE3deriv;
    _jacobianOplusXi.block<3, 6>(0, 6) = Eigen::Matrix<double, 3, 6>::Zero();
}

void EdgeMonoGPOnlyPose::computeError()
{
    const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
    const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
    const PoseVelocity& f1 = v1->estimate();
    const PoseVelocity& f2 = v2->estimate(); 

    const Sophus::SE3d Twb = gp->QueryPose(f1.Twb, f2.Twb, f1.Vel, f2.Vel, f1.time, f2.time, t);
    const Sophus::SE3d Twc = Twb * MultiKeyFrame::mTbc[cam_idx].cast<double>();
    const Eigen::Vector3d Xc = Twc.inverse() * Xw;
    const Eigen::Vector2d xc = f1.vpCameras[cam_idx]->project(Xc);
    const Eigen::Vector2d obs(_measurement);
    _error = obs - xc;
}

void EdgeMonoGPOnlyPose::linearizeOplus()
{
    const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
    const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
    const PoseVelocity& f1 = v1->estimate();
    const PoseVelocity& f2 = v2->estimate(); 

    Eigen::VectorXd xi12;
    Eigen::Matrix<double, 6, 12> At1, Pt1;
    Sophus::SE3d dT, Twb;
    Twb = gp->QueryPose(f1.Twb, f2.Twb, f1.Vel, f2.Vel, f1.time, f2.time, t, At1, Pt1, dT, xi12);

    const Sophus::SE3d Tcb = MultiKeyFrame::mTbc[cam_idx].cast<double>().inverse();
    const Eigen::Vector3d Xb = Twb.inverse() * Xw;
    const Eigen::Vector3d Xc = Tcb * Xb;

    Eigen::Matrix<double, 2, 3> proj_jac;
    proj_jac = f1.vpCameras[cam_idx]->projectJac(Xc);

    Eigen::Matrix<double, 3, 6> SE3deriv;
    Eigen::Matrix<double, 3, 3> Rcb = Tcb.rotationMatrix();
    SE3deriv << -Rcb, Rcb * Skew(Xb);

    Eigen::Matrix<double, 2, 6> J1 = -proj_jac * SE3deriv;
    Eigen::Matrix<double, 6, 1> dxi = dT.log();
    Eigen::Matrix<double, 6, 6> Ad_dT = Sophus::SE3d::exp(-dxi).Adj();
    Eigen::Matrix<double, 6, 6> Jr_dxi = RightJacobianPose3(dxi);
    Eigen::Matrix<double, 6, 6> Jr_inv_xi12 = RightJacobianPose3Inv(xi12);

    Eigen::Matrix<double, 6, 6> ad_v2 = se3Adj(f2.Vel);
    Eigen::Matrix<double,6 ,6> ad_T12 = Sophus::SE3d::exp(xi12).Adj();

    Eigen::Matrix<double, 12, 6> JinT1, JinV1, JinT2, JinV2;
    JinT1.block<6, 6>(0, 0) = -Jr_inv_xi12 * ad_T12.inverse();
    JinT1.block<6, 6>(6, 0) = -0.5 * ad_v2 * JinT1.block<6, 6>(0, 0);
    JinV1 << Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 6>::Identity();
    JinT2.block<6, 6>(0, 0) = Jr_inv_xi12;
    JinT2.block<6, 6>(6, 0) = -0.5 * ad_v2 * JinT2.block<6, 6>(0, 0);
    JinV2 << Eigen::Matrix<double, 6, 6>::Zero(), Jr_inv_xi12;

    _jacobianOplusXi.block<2, 6>(0, 0) = J1 * (Jr_dxi * Pt1 * JinT1 + Ad_dT);
    _jacobianOplusXi.block<2, 6>(0, 6) = J1 * Jr_dxi * At1 * JinV1;

    Eigen::Matrix<double, 2, 12> Jj1 = J1 * Jr_dxi * Pt1;
    _jacobianOplusXj.block<2, 6>(0, 0) = Jj1 * JinT2;
    _jacobianOplusXj.block<2, 6>(0, 6) = Jj1 * JinV2;
}

void EdgeMonoGP::computeError()
{
    const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
    const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* vpMP = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[2]);
    const PoseVelocity& f1 = v1->estimate();
    const PoseVelocity& f2 = v2->estimate(); 

    const Sophus::SE3d Twb = gp->QueryPose(f1.Twb, f2.Twb, f1.Vel, f2.Vel, f1.time, f2.time, t);
    const Sophus::SE3d Twc = Twb * MultiKeyFrame::mTbc[cam_idx].cast<double>();
    const Eigen::Vector3d Xc = Twc.inverse() * vpMP->estimate();
    const Eigen::Vector2d xc = f1.vpCameras[cam_idx]->project(Xc);
    const Eigen::Vector2d obs(_measurement);
    _error = obs - xc;
}

void EdgeMonoGPExtrinsic::computeError()
{
    const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
    const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* vpMP = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[2]);
    const VertexExtrinsic* v3 = static_cast<const VertexExtrinsic*>(_vertices[3]);
    const PoseVelocity& f1 = v1->estimate();
    const PoseVelocity& f2 = v2->estimate(); 
    
    const Sophus::SE3d Twb = gp->QueryPose(f1.Twb, f2.Twb, f1.Vel, f2.Vel, f1.time, f2.time, t);
    const Sophus::SE3d Twc = Twb * v3->estimate();
    const Eigen::Vector3d Xc = Twc.inverse() * vpMP->estimate();
    const Eigen::Vector2d xc = f1.vpCameras[cam_idx]->project(Xc);
    const Eigen::Vector2d obs(_measurement);
    _error = obs - xc;
}

void EdgeMonoGPExtrinsic::linearizeOplus()
{
    const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
    const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* vpMP = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[2]);
    const VertexExtrinsic* v3 = static_cast<const VertexExtrinsic*>(_vertices[3]);
    const PoseVelocity& f1 = v1->estimate();
    const PoseVelocity& f2 = v2->estimate(); 
    const Eigen::Vector3d& Xw = vpMP->estimate();

    Eigen::VectorXd xi12;
    Eigen::Matrix<double, 6, 12> At1, Pt1;
    Sophus::SE3d dT, Twb;
    Twb = gp->QueryPose(f1.Twb, f2.Twb, f1.Vel, f2.Vel, f1.time, f2.time, t, At1, Pt1, dT, xi12);

    const Sophus::SE3d Tcb = v3->estimate().inverse();
    const Eigen::Matrix3d Rbw = Twb.rotationMatrix().transpose();
    const Eigen::Vector3d Xb = Twb.inverse() * Xw;
    const Eigen::Vector3d Xc = Tcb * Xb;

    Eigen::Matrix<double, 2, 3> proj_jac;
    proj_jac = f1.vpCameras[cam_idx]->projectJac(Xc);

    Eigen::Matrix<double, 3, 6> SE3deriv;
    Eigen::Matrix<double, 3, 3> Rcb = Tcb.rotationMatrix();
    SE3deriv << -Rcb, Rcb * Skew(Xb);

    Eigen::Matrix<double, 2, 6> J1 = -proj_jac * SE3deriv;
    Eigen::Matrix<double, 6, 1> dxi = dT.log();
    Eigen::Matrix<double, 6, 6> Ad_dT = Sophus::SE3d::exp(-dxi).Adj();
    Eigen::Matrix<double, 6, 6> Jr_dxi = RightJacobianPose3(dxi);
    Eigen::Matrix<double, 6, 6> Jr_inv_xi12 = RightJacobianPose3Inv(xi12);

    Eigen::Matrix<double, 6, 6> ad_v2 = se3Adj(f2.Vel);
    Eigen::Matrix<double,6 ,6> ad_T12 = Sophus::SE3d::exp(xi12).Adj();

    Eigen::Matrix<double, 12, 6> JinT1, JinV1, JinT2, JinV2;
    JinT1.block<6, 6>(0, 0) = -Jr_inv_xi12 * ad_T12.inverse();
    JinT1.block<6, 6>(6, 0) = -0.5 * ad_v2 * JinT1.block<6, 6>(0, 0);
    JinV1 << Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 6>::Identity();
    JinT2.block<6, 6>(0, 0) = Jr_inv_xi12;
    JinT2.block<6, 6>(6, 0) = -0.5 * ad_v2 * JinT2.block<6, 6>(0, 0);
    JinV2 << Eigen::Matrix<double, 6, 6>::Zero(), Jr_inv_xi12;

    _jacobianOplus[0].block<2, 6>(0, 0) = J1 * (Jr_dxi * Pt1 * JinT1 + Ad_dT);
    _jacobianOplus[0].block<2, 6>(0, 6) = J1 * Jr_dxi * At1 * JinV1;

    Eigen::Matrix<double, 2, 12> Jj1 = J1 * Jr_dxi * Pt1;
    _jacobianOplus[1].block<2, 6>(0, 0) = Jj1 * JinT2;
    _jacobianOplus[1].block<2, 6>(0, 6) = Jj1 * JinV2;

    _jacobianOplus[2] = -proj_jac * Rcb * Rbw;

    Eigen::Matrix<double, 3, 6> SE3deriv2;
    SE3deriv2  << -Eigen::Matrix3d::Identity(), Skew(Xc);
    _jacobianOplus[3] = -proj_jac * SE3deriv2;
}

void EdgeMonoGP::linearizeOplus()
{
    const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
    const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* vpMP = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[2]);
    const PoseVelocity& f1 = v1->estimate();
    const PoseVelocity& f2 = v2->estimate(); 
    const Eigen::Vector3d& Xw = vpMP->estimate();

    Eigen::VectorXd xi12;
    Eigen::Matrix<double, 6, 12> At1, Pt1;
    Sophus::SE3d dT, Twb;
    Twb = gp->QueryPose(f1.Twb, f2.Twb, f1.Vel, f2.Vel, f1.time, f2.time, t, At1, Pt1, dT, xi12);

    const Sophus::SE3d Tcb = MultiKeyFrame::mTbc[cam_idx].cast<double>().inverse();
    const Eigen::Matrix3d Rbw = Twb.rotationMatrix().transpose();
    const Eigen::Vector3d Xb = Twb.inverse() * Xw;
    const Eigen::Vector3d Xc = Tcb * Xb;

    Eigen::Matrix<double, 2, 3> proj_jac;
    proj_jac = f1.vpCameras[cam_idx]->projectJac(Xc);

    Eigen::Matrix<double, 3, 6> SE3deriv;
    Eigen::Matrix<double, 3, 3> Rcb = Tcb.rotationMatrix();
    SE3deriv << -Rcb, Rcb * Skew(Xb);

    Eigen::Matrix<double, 2, 6> J1 = -proj_jac * SE3deriv;
    Eigen::Matrix<double, 6, 1> dxi = dT.log();
    Eigen::Matrix<double, 6, 6> Ad_dT = Sophus::SE3d::exp(-dxi).Adj();
    Eigen::Matrix<double, 6, 6> Jr_dxi = RightJacobianPose3(dxi);
    Eigen::Matrix<double, 6, 6> Jr_inv_xi12 = RightJacobianPose3Inv(xi12);

    Eigen::Matrix<double, 6, 6> ad_v2 = se3Adj(f2.Vel);
    Eigen::Matrix<double,6 ,6> ad_T12 = Sophus::SE3d::exp(xi12).Adj();

    Eigen::Matrix<double, 12, 6> JinT1, JinV1, JinT2, JinV2;
    JinT1.block<6, 6>(0, 0) = -Jr_inv_xi12 * ad_T12.inverse();
    JinT1.block<6, 6>(6, 0) = -0.5 * ad_v2 * JinT1.block<6, 6>(0, 0);
    JinV1 << Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 6>::Identity();
    JinT2.block<6, 6>(0, 0) = Jr_inv_xi12;
    JinT2.block<6, 6>(6, 0) = -0.5 * ad_v2 * JinT2.block<6, 6>(0, 0);
    JinV2 << Eigen::Matrix<double, 6, 6>::Zero(), Jr_inv_xi12;

    _jacobianOplus[0].block<2, 6>(0, 0) = J1 * (Jr_dxi * Pt1 * JinT1 + Ad_dT);
    _jacobianOplus[0].block<2, 6>(0, 6) = J1 * Jr_dxi * At1 * JinV1;

    Eigen::Matrix<double, 2, 12> Jj1 = J1 * Jr_dxi * Pt1;
    _jacobianOplus[1].block<2, 6>(0, 0) = Jj1 * JinT2;
    _jacobianOplus[1].block<2, 6>(0, 6) = Jj1 * JinV2;

    _jacobianOplus[2] = -proj_jac * Rcb * Rbw;
}

void EdgeStereoGP::computeError()
{
    const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
    const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* vpMP = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[2]);
    const PoseVelocity& f1 = v1->estimate();
    const PoseVelocity& f2 = v2->estimate(); 

    const Sophus::SE3d Twb = gp->QueryPose(f1.Twb, f2.Twb, f1.Vel, f2.Vel, f1.time, f2.time, t);
    const Sophus::SE3d Twc = Twb * MultiKeyFrame::mTbc[cam_idx].cast<double>();
    const Eigen::Vector3d Xc = Twc.inverse() * vpMP->estimate();
    Eigen::Vector3d xc;
    double invZ = 1/Xc(2);
    xc.head(2) = f1.vpCameras[cam_idx]->project(Xc);
    xc(2) = xc(0) - f1.bf * invZ;
    const Eigen::Vector3d obs(_measurement);
    _error = obs - xc;
}

void EdgeStereoGP::linearizeOplus()
{
    const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
    const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
    const g2o::VertexSBAPointXYZ* vpMP = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[2]);
    const PoseVelocity& f1 = v1->estimate();
    const PoseVelocity& f2 = v2->estimate(); 
    const Eigen::Vector3d& Xw = vpMP->estimate();

    Eigen::VectorXd xi12;
    Eigen::Matrix<double, 6, 12> At1, Pt1;
    Sophus::SE3d dT, Twb;
    Twb = gp->QueryPose(f1.Twb, f2.Twb, f1.Vel, f2.Vel, f1.time, f2.time, t, At1, Pt1, dT, xi12);

    const Sophus::SE3d Tcb = MultiKeyFrame::mTbc[cam_idx].cast<double>().inverse();
    const Eigen::Matrix3d Rbw = Twb.rotationMatrix().transpose();
    Eigen::Vector3d Xb = Twb.inverse() * Xw;
    const Eigen::Vector3d Xc = Tcb * Xb;
    const double bf = f1.bf;
    const double inv_z2 = 1.0 / (Xc(2) * Xc(2));

    Eigen::Matrix<double, 3, 3> proj_jac;
    proj_jac.block<2, 3>(0, 0) = f1.vpCameras[cam_idx]->projectJac(Xc);
    proj_jac.block<1, 3>(2, 0) = proj_jac.block<1, 3>(0, 0);
    proj_jac(2, 2) += bf * inv_z2;

    Eigen::Matrix<double, 3, 6> SE3deriv;
    Eigen::Matrix<double, 3, 3> Rcb = Tcb.rotationMatrix();
    SE3deriv << -Rcb, Rcb * Skew(Xb);

    Eigen::Matrix<double, 3, 6> J1 = -proj_jac * SE3deriv;
    Eigen::Matrix<double, 6, 1> dxi = dT.log();
    Eigen::Matrix<double, 6, 6> Ad_dT = Sophus::SE3d::exp(-dxi).Adj();
    Eigen::Matrix<double, 6, 6> Jr_dxi = RightJacobianPose3(dxi);
    Eigen::Matrix<double, 6, 6> Jr_inv_xi12 = RightJacobianPose3Inv(xi12);

    Eigen::Matrix<double, 6, 6> ad_v2 = se3Adj(f2.Vel);
    Eigen::Matrix<double,6 ,6> ad_T12 = Sophus::SE3d::exp(xi12).Adj();

    Eigen::Matrix<double, 12, 6> JinT1, JinV1, JinT2, JinV2;
    JinT1.block<6, 6>(0, 0) = -Jr_inv_xi12 * ad_T12.inverse();
    JinT1.block<6, 6>(6, 0) = -0.5 * ad_v2 * JinT1.block<6, 6>(0, 0);
    JinV1 << Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 6>::Identity();
    JinT2.block<6, 6>(0, 0) = Jr_inv_xi12;
    JinT2.block<6, 6>(6, 0) = -0.5 * ad_v2 * JinT2.block<6, 6>(0, 0);
    JinV2 << Eigen::Matrix<double, 6, 6>::Zero(), Jr_inv_xi12;

    _jacobianOplus[0].block<3, 6>(0, 0) = J1 * (Jr_dxi * Pt1 * JinT1 + Ad_dT);
    _jacobianOplus[0].block<3, 6>(0, 6) = J1 * Jr_dxi * At1 * JinV1;

    Eigen::Matrix<double, 3, 12> Jj1 = J1 * Jr_dxi * Pt1;
    _jacobianOplus[1].block<3, 6>(0, 0) = Jj1 * JinT2;
    _jacobianOplus[1].block<3, 6>(0, 6) = Jj1 * JinV2;

    _jacobianOplus[2] = -proj_jac * Rcb * Rbw;
}

void EdgeMono::linearizeOplus()
{
    const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
    const g2o::VertexSBAPointXYZ* vpMP = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[1]);
    const PoseVelocity& f1 = v1->estimate();
    const Eigen::Vector3d& Xw = vpMP->estimate();

    const Sophus::SE3d Tcb = MultiKeyFrame::mTbc.back().cast<double>().inverse();
    const Eigen::Matrix3d Rbw = f1.Twb.rotationMatrix().transpose();
    const Eigen::Vector3d Xb = f1.Twb.inverse() * Xw;
    const Eigen::Vector3d Xc = Tcb * Xb;

    Eigen::Matrix<double, 2, 3> proj_jac;
    proj_jac = f1.vpCameras.back()->projectJac(Xc);

    Eigen::Matrix<double, 3, 6> SE3deriv;
    Eigen::Matrix<double, 3, 3> Rcb = Tcb.rotationMatrix();
    SE3deriv << -Rcb, Rcb * Skew(Xb);

    _jacobianOplusXi.block<2, 6>(0, 0) = -proj_jac * SE3deriv;
    _jacobianOplusXi.block<2, 6>(0, 6) = Eigen::Matrix<double, 2, 6>::Zero();
    _jacobianOplusXj = -proj_jac * Rcb * Rbw;
}

void EdgeStereo::linearizeOplus()
{
    const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
    const g2o::VertexSBAPointXYZ* vpMP = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[1]);
    const PoseVelocity& f1 = v1->estimate();
    const Eigen::Vector3d& Xw = vpMP->estimate();

    const Sophus::SE3d Tcb = MultiKeyFrame::mTbc.back().cast<double>().inverse();
    const Eigen::Matrix3d Rbw = f1.Twb.rotationMatrix().transpose();
    const Eigen::Vector3d Xb = f1.Twb.inverse() * Xw;
    const Eigen::Vector3d Xc = Tcb * Xb;
    const double bf = f1.bf;
    const double inv_z2 = 1.0 / (Xc(2) * Xc(2));

    Eigen::Matrix<double, 3, 3> proj_jac;
    proj_jac.block<2, 3>(0, 0) = f1.vpCameras.back()->projectJac(Xc);
    proj_jac.block<1, 3>(2, 0) = proj_jac.block<1, 3>(0, 0);
    proj_jac(2, 2) += bf * inv_z2;

    Eigen::Matrix<double, 3, 6> SE3deriv;
    Eigen::Matrix<double, 3, 3> Rcb = Tcb.rotationMatrix();
    SE3deriv << -Rcb, Rcb * Skew(Xb);

    _jacobianOplusXi.block<3, 6>(0, 0) = -proj_jac * SE3deriv;
    _jacobianOplusXi.block<3, 6>(0, 6) = Eigen::Matrix<double, 3, 6>::Zero();
    _jacobianOplusXj = -proj_jac * Rcb * Rbw;
}

void EdgeVelReproj::linearizeOplus()
{
    const VertexVel* v = static_cast<const VertexVel*>(_vertices[0]);
    const Eigen::Matrix<double, 6, 1> dxi = v->estimate() * dt;
    const Sophus::SE3d Tcb1 = MultiKeyFrame::mTbc[cam].cast<double>().inverse() * Sophus::SE3d::exp(-dxi);
    const Eigen::Vector3d Xb = T.inverse() * Xw;
    const Eigen::Vector3d Xc = Tcb1 * Xb;
    Eigen::Matrix<double, 2, 3> proj_jac = pf->mvpCamera[cam]->projectJac(Xc);

    Eigen::Matrix<double, 4, 6> SE3deriv;
    SE3deriv = -Tcb1.matrix() * CircleDot(Xb)* RightJacobianPose3(-dxi) * dt;

    _jacobianOplusXi = -proj_jac * SE3deriv.block<3, 6>(0, 0);
}

// SO3 FUNCTIONS
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w)
{
    return ExpSO3(w[0],w[1],w[2]);
}

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        Eigen::Matrix3d res = Eigen::Matrix3d::Identity() + W +0.5*W*W;
        return NormalizeRotation(res);
    }
    else
    {
        Eigen::Matrix3d res =Eigen::Matrix3d::Identity() + W*sin(d)/d + W*W*(1.0-cos(d))/d2;
        return NormalizeRotation(res);
    }
}

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
{
    const double tr = R(0,0)+R(1,1)+R(2,2);
    Eigen::Vector3d w;
    w << (R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
    const double costheta = (tr-1.0)*0.5f;
    if(costheta>1 || costheta<-1)
        return w;
    const double theta = acos(costheta);
    const double s = sin(theta);
    if(fabs(s)<1e-5)
        return w;
    else
        return theta*w/s;
}

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v)
{
    return InverseRightJacobianSO3(v[0],v[1],v[2]);
}

Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
        return Eigen::Matrix3d::Identity();
    else
        return Eigen::Matrix3d::Identity() + W/2 + W*W*(1.0/d2 - (1.0+cos(d))/(2.0*d*sin(d)));
}

Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v)
{
    return RightJacobianSO3(v[0],v[1],v[2]);
}

Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z)
{
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);

    Eigen::Matrix3d W;
    W << 0.0, -z, y,z, 0.0, -x,-y,  x, 0.0;
    if(d<1e-5)
    {
        return Eigen::Matrix3d::Identity();
    }
    else
    {
        return Eigen::Matrix3d::Identity() - W*(1.0-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
    }
}

Eigen::Matrix3d Skew(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d W;
    W << 0.0, -w[2], w[1],w[2], 0.0, -w[0],-w[1],  w[0], 0.0;
    return W;
}

}
