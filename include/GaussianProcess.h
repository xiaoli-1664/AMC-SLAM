#pragma once
#include <vector>
#include <cmath>

#include "Eigen/Core"
#include "sophus/se3.hpp"

#include "Pose3utils.h"

namespace ORB_SLAM3 {

class GaussianProcess {
public:

    GaussianProcess(): mQc(Eigen::Matrix<double, 6, 6>::Identity()), mQcInv(mQc) {}
    GaussianProcess(const Eigen::Matrix<double, 6, 6>& Qc): mQc(Qc), mQcInv(Qc.inverse()) {}

    inline void SetQc(const Eigen::Matrix<double, 6, 6>& Qc) { mQc = Qc; }

    Eigen::Matrix<double, 12, 12> Qi(double dt) const {
        Eigen::Matrix<double, 12, 12> xQi = Eigen::Matrix<double, 12, 12>::Zero();
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        xQi.block<6, 6>(0, 0) = 1.0 / 3.0 * dt3 *mQc;
        xQi.block<6, 6>(0, 6) = 1.0 / 2.0 * dt2 * mQc;
        xQi.block<6, 6>(6, 0) = 1.0 / 2.0 * dt2 * mQc;
        xQi.block<6, 6>(6, 6) = dt * mQc;
        return xQi;
    }

    Eigen::Matrix<double, 12, 12> QiInv(double dt) const {
        assert(fabs(dt) > 1e-6);
        Eigen::Matrix<double, 12, 12> xQiInv = Eigen::Matrix<double, 12, 12>::Zero();
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        xQiInv.block<6, 6>(0, 0) = 12.0 / dt3 * mQcInv;
        xQiInv.block<6, 6>(0, 6) = -6.0 / dt2 * mQcInv;
        xQiInv.block<6, 6>(6, 0) = -6.0 / dt2 * mQcInv;
        xQiInv.block<6, 6>(6, 6) = 4.0 / dt * mQcInv;
        return xQiInv;
    }

    // 高斯过程状态转移矩阵
    Eigen::Matrix<double, 12, 12> Transition(double t1, double t2) const {
        Eigen::Matrix<double, 12, 12> T = Eigen::Matrix<double, 12, 12>::Identity();
        T.block<6, 6>(0, 6) = (t2 - t1) * Eigen::Matrix<double, 6, 6>::Identity();
        return T;
    }

    // 高斯过程插值函数，查询两个位姿之间的位姿
    Sophus::SE3d QueryPose(const Sophus::SE3d& pose1, const Sophus::SE3d& pose2, const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, double t1, double t2, double t) const;
    Sophus::SE3d QueryPose(const Sophus::SE3d& pose1, const Sophus::SE3d& pose2, 
                            const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, double t1, double t2, double t,
                            Eigen::Matrix<double, 6, 12>& At1, Eigen::Matrix<double, 6, 12>& Pt1,
                            Sophus::SE3d& dT, Eigen::VectorXd& xi12) const;

public:
    Eigen::Matrix<double, 6, 6> mQc;
    Eigen::Matrix<double, 6, 6> mQcInv;
};

} // namespace ORB_SLAM3