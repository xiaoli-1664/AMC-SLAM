#include "Pose3utils.h"

namespace ORB_SLAM3 {

Eigen::Matrix3d LeftJacobianPose3Q(const Eigen::VectorXd& xi) {
    const Eigen::Vector3d omega = xi.tail<3>(), rho = xi.head<3>();
    const double theta = omega.norm();
    const Eigen::Matrix3d X = Sophus::SO3d::hat(omega), Y = Sophus::SO3d::hat(rho);

    const Eigen::Matrix3d XY = X * Y, YX = Y * X, XYX = X * YX;
    if (fabs(theta) > 1e-5) {
        const double sin_theta = sin(theta), cos_theta = cos(theta);
        const double theta2 = theta * theta, theta3 = theta2 * theta, theta4 = theta3 * theta, theta5 = theta4 * theta;

        return 0.5 * Y + (theta - sin_theta) / (theta3) * (XY + YX + XYX) -
                (1.0 - 0.5 * theta2 - cos_theta) / theta4 * (X * XY + YX * X - 3.0 * XYX)
                - 0.5 * ((1.0 - 0.5 * theta2 - cos_theta) / theta4 - 3.0 * (theta - sin_theta - theta3 / 6.0) / theta5) * (XYX * X + X * XYX);
    } else {
        return 0.5 * Y + 1.0 / 6.0 * (XY + YX + XYX) - 1.0 / 24.0 * (X * XY + YX * X - 3.0 * XYX)
                - 0.5 * (1.0 / 24.0 + 3.0 / 120.0) * (XYX * X + X * XYX);
    }
}

Eigen::Matrix<double, 6, 6> LeftJacobianPose3(const Eigen::VectorXd& xi) {
    const Eigen::Vector3d omega = xi.tail<3>();
    const Eigen::Matrix3d Q = LeftJacobianPose3Q(xi);
    const Eigen::Matrix3d J = LeftJacobianRot3(omega);

    return (Eigen::Matrix<double, 6, 6>() << J, Q, Eigen::Matrix3d::Zero(), J).finished();
}

Eigen::Matrix<double, 6, 6> RightJacobianPose3(const Eigen::VectorXd& xi) {
    return LeftJacobianPose3(-xi);
}

Eigen::Matrix<double, 6, 6> LeftJacobianPose3Inv(const Eigen::VectorXd& xi) {
    const Eigen::Vector3d omega = xi.tail<3>();
    const Eigen::Matrix3d Q = LeftJacobianPose3Q(xi);
    const Eigen::Matrix3d Jinv = LeftJacobianRot3Inv(omega);

    return (Eigen::Matrix<double, 6, 6>() << Jinv, -Jinv * Q * Jinv, Eigen::Matrix3d::Zero(), Jinv).finished();
}

Eigen::Matrix<double, 6, 6> RightJacobianPose3Inv(const Eigen::VectorXd& xi) {
    return LeftJacobianPose3Inv(-xi);
}

Eigen::Matrix3d LeftJacobianRot3(const Eigen::Vector3d& omega) {
    double theta2 = omega.dot(omega);
    if (theta2 <= std::numeric_limits<double>::epsilon()) return Eigen::Matrix3d::Identity();
    const double theta = sqrt(theta2);
    const Eigen::Vector3d dir = omega / theta;

    const double sin_theta = sin(theta);
    const Eigen::Matrix3d A = Sophus::SO3d::hat(omega) / theta;

    return sin_theta / theta * Eigen::Matrix3d::Identity() + (1.0 - sin_theta / theta) * (dir * dir.transpose())
            + (1.0 - cos(theta)) / theta * A; 
}

Eigen::Matrix3d LeftJacobianRot3Inv(const Eigen::Vector3d& omega) {
    double theta2 = omega.dot(omega);
    if (theta2 <= std::numeric_limits<double>::epsilon()) return Eigen::Matrix3d::Identity();
    const double theta = sqrt(theta2);
    const Eigen::Vector3d dir = omega / theta;

    const double theta_2 = theta / 2.0;
    const double cot_theta_2 = 1.0 / tan(theta_2);
    const Eigen::Matrix3d A = Sophus::SO3d::hat(omega) / theta;

    return theta_2 * cot_theta_2 * Eigen::Matrix3d::Identity() + (1.0 - theta_2 * cot_theta_2) * (dir * dir.transpose())
            - theta_2 * A;
}

Eigen::Matrix<double, 4, 6> CircleDot(const Eigen::Vector3d& p) {
    Eigen::Matrix<double, 4, 6> res = Eigen::Matrix<double, 4, 6>::Zero();
    res.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    res.block<3, 3>(0, 3) = -Sophus::SO3d::hat(p);
    return res;
}

std::vector<Eigen::Matrix<double, 6, 6>> jacobianNumercialDiff(
    boost::function<Eigen::Matrix<double, 6, 6>(const Eigen::Matrix<double, 6, 1>)> func,
    const Sophus::SE3d& dT, const Eigen::Matrix<double, 6, 1>& w, double dxi)
{
    std::vector<Eigen::Matrix<double, 6, 6>> Diff(2, Eigen::Matrix<double, 6, 6>::Zero());
    Eigen::Matrix<double, 6, 1> dxi_plus, dxi_minus, xi_plus, xi_minus;
    Eigen::Matrix<double, 6, 6> J_plus, J_minus;

    for (int i = 0; i < 6; ++i) {
        dxi_plus = Eigen::Matrix<double, 6, 1>::Zero();
        dxi_plus(i) = dxi;
        dxi_minus = Eigen::Matrix<double, 6, 1>::Zero();
        dxi_minus(i) = -dxi;

        xi_plus = (dT * Sophus::SE3d::exp(dxi_plus).inverse()).log();
        xi_minus = (dT * Sophus::SE3d::exp(dxi_minus).inverse()).log();
        J_plus = func(xi_plus);
        J_minus = func(xi_minus);
        Diff[0].col(i) = (J_plus - J_minus) * w / (2 * dxi);

        xi_plus = (Sophus::SE3d::exp(dxi_plus) * dT).log();
        xi_minus = (Sophus::SE3d::exp(dxi_minus) * dT).log();
        J_plus = func(xi_plus);
        J_minus = func(xi_minus);
        Diff[1].col(i) = (J_plus - J_minus) * w / (2 * dxi);
    }
    return Diff;
}

Eigen::Matrix<double, 6, 6> se3Adj(const Eigen::Matrix<double, 6, 1>& v)
{
    Eigen::Matrix<double, 6, 6> Adj;
    Adj.setZero();
    Adj.block<3, 3>(0, 0) = Sophus::SO3d::hat(v.tail<3>());
    Adj.block<3, 3>(0, 3) = Sophus::SO3d::hat(v.head<3>());
    Adj.block<3, 3>(3, 3) = Sophus::SO3d::hat(v.tail<3>());
    return Adj;
}

} // namespace ORB_SLAM3