#include "GaussianProcess.h"

namespace ORB_SLAM3 {

Sophus::SE3d GaussianProcess::QueryPose(const Sophus::SE3d& pose1, const Sophus::SE3d& pose2, const Eigen::VectorXd& v1, 
                                        const Eigen::VectorXd& v2, double t1, double t2, double t) const {
    Eigen::Matrix<double, 12, 12> Pt = Qi(t - t1) * Transition(t, t2).transpose() * QiInv(t2 - t1);
    Eigen::Matrix<double, 12, 12> At = Transition(t1, t) - Pt * Transition(t1, t2);
    Eigen::Matrix<double, 6, 12> At1 = At.block<6, 12>(0, 0);
    Eigen::Matrix<double, 6, 12> Pt1 = Pt.block<6, 12>(0, 0);
    Eigen::Matrix<double, 12, 1> x1, x2;
    x1.block<6, 1>(0, 0).setZero();
    x1.block<6, 1>(6, 0) = v1;
    Sophus::SE3d dp = pose1.inverse() * pose2;
    // 位移在前，旋转在后
    Eigen::VectorXd xi = dp.log();
    x2.block<6, 1>(0, 0) = xi;
    x2.block<6, 1>(6, 0) = RightJacobianPose3Inv(xi) * v2;
    Sophus::SE3d T = pose1 * Sophus::SE3d::exp(At1 * x1 + Pt1 * x2);
    return T;
}

Sophus::SE3d GaussianProcess::QueryPose(const Sophus::SE3d& pose1, const Sophus::SE3d& pose2, const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, double t1, double t2, double t,
                            Eigen::Matrix<double, 6, 12>& At1, Eigen::Matrix<double, 6, 12>& Pt1, 
                            Sophus::SE3d& dT, Eigen::VectorXd& xi12) const 
{
    Eigen::Matrix<double, 12, 12> Pt = Qi(t - t1) * Transition(t, t2).transpose() * QiInv(t2 - t1);
    Eigen::Matrix<double, 12, 12> At = Transition(t1, t) - Pt * Transition(t1, t2);
    At1 = At.block<6, 12>(0, 0);
    Pt1 = Pt.block<6, 12>(0, 0);
    Eigen::VectorXd x1(12), x2(12);
    x1.block<6, 1>(0, 0).setZero();
    x1.block<6, 1>(6, 0) = v1;
    Sophus::SE3d dp = pose1.inverse() * pose2;
    // 位移在前，旋转在后
    xi12 = dp.log();
    x2.block<6, 1>(0, 0) = xi12;
    x2.block<6, 1>(6, 0) = RightJacobianPose3Inv(xi12) * v2;
    dT = Sophus::SE3d::exp(At1 * x1 + Pt1 * x2);
    Sophus::SE3d T = pose1 * dT;
    return T;
}

} // namespace ORB_SLAM3