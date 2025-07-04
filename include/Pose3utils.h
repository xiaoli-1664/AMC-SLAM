#pragma once

#include <boost/function.hpp>

#include <Eigen/Core>
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"

namespace ORB_SLAM3 {

Eigen::Matrix<double, 6, 6> LeftJacobianPose3(const Eigen::VectorXd& xi);

Eigen::Matrix<double, 6, 6> RightJacobianPose3(const Eigen::VectorXd& xi);

Eigen::Matrix<double, 6, 6> LeftJacobianPose3Inv(const Eigen::VectorXd& xi);

Eigen::Matrix<double, 6, 6> RightJacobianPose3Inv(const Eigen::VectorXd& xi);

Eigen::Matrix3d LeftJacobianPose3Q(const Eigen::VectorXd& xi);

Eigen::Matrix3d LeftJacobianRot3(const Eigen::Vector3d& omega);

Eigen::Matrix3d LeftJacobianRot3Inv(const Eigen::Vector3d& omega);

Eigen::Matrix<double, 4, 6> CircleDot(const Eigen::Vector3d& p);

std::vector<Eigen::Matrix<double, 6, 6>> jacobianNumercialDiff(
    boost::function<Eigen::Matrix<double, 6, 6>(const Eigen::Matrix<double, 6, 1>)> func,
    const Sophus::SE3d& dT, const Eigen::Matrix<double, 6, 1>& w,
    double dxi=1e-6);

Eigen::Matrix<double, 6, 6> se3Adj(const Eigen::Matrix<double, 6, 1>& v);

} // namespace ORB_SLAM3