#ifndef G2OTYPES_H
#define G2OTYPES_H

#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_sba.h"
#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"

#include<opencv2/core/core.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <Frame.h>
#include <KeyFrame.h>

#include"Converter.h"
#include "Pose3utils.h"
#include "sophus/se3.hpp"

#include <math.h>

namespace ORB_SLAM3
{

class MultiKeyFrame;
class MultiFrame;
class GeometricCamera;
class GaussianProcess;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 12, 1> Vector12d;
typedef Eigen::Matrix<double, 15, 1> Vector15d;
typedef Eigen::Matrix<double, 12, 12> Matrix12d;
typedef Eigen::Matrix<double, 15, 15> Matrix15d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z);
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w);

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z);

Eigen::Matrix3d Skew(const Eigen::Vector3d &w);
Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z);

template<typename T = double>
Eigen::Matrix<T,3,3> NormalizeRotation(const Eigen::Matrix<T,3,3> &R) {
    Eigen::JacobiSVD<Eigen::Matrix<T,3,3>> svd(R,Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

class PoseVelocity
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PoseVelocity(){}
    PoseVelocity(MultiKeyFrame* pKF);
    PoseVelocity(MultiFrame* pF);
    void Update(const double *pu);
    Eigen::Vector3d ProjectStereo(const Eigen::Vector3d &Xw) const;
    Eigen::Vector2d Project(const Eigen::Vector3d &Xw) const;
    bool isDepthPositive(const Eigen::Vector3d &Xw, int cam_idx) const;
    bool isDepthPositive(const Eigen::Vector3d &Xw) const;
    bool isDepthPositive(const Eigen::Vector3d &Xw, int cam_idx, Sophus::SE3d Tbc) const; 

public:
    Sophus::SE3d Twb;
    // 速度表示在世界坐标系下
    Vector6d Vel;

    double time, bf;
    std::vector<GeometricCamera*> vpCameras;
};


class VertexExtrinsic : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexExtrinsic() {}

    VertexExtrinsic(Sophus::SE3d Tbc) {
        setEstimate(Tbc);
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
    }

    virtual void oplusImpl(const double* update_) {
        _estimate = _estimate * Sophus::SE3d::exp(Eigen::Map<const Vector6d>(update_));
    }
};

class VertexPoseVel : public g2o::BaseVertex<12, PoseVelocity>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPoseVel() {}
    VertexPoseVel(MultiKeyFrame* pKF) {
        setEstimate(PoseVelocity(pKF));
    }
    VertexPoseVel(MultiFrame* pF) {
        setEstimate(PoseVelocity(pF));
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const {return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_) {
        _estimate.Update(update_);
        updateCache();
    }
};

class VertexVel : public g2o::BaseVertex<6, Vector6d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexVel() {}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        _estimate.setZero();
    }

    virtual void oplusImpl(const double* update_) {
        _estimate += Eigen::Map<const Vector6d>(update_);
    }

};

class EdgeGaussianPrior : public g2o::BaseBinaryEdge<12, Vector12d, VertexPoseVel, VertexPoseVel>
{
public:
    EdgeGaussianPrior() {}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
        const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
        const PoseVelocity f1 = v1->estimate();
        const PoseVelocity f2 = v2->estimate();
        Vector6d dxi = (f1.Twb.inverse() * f2.Twb).log();
        _error.block<6, 1>(0, 0) = dxi - (f2.time - f1.time) * f1.Vel;
        _error.block<6, 1>(6, 0) = RightJacobianPose3Inv(dxi) * f2.Vel - f1.Vel;
    }

    virtual void linearizeOplus();

    Eigen::Matrix<double, 24, 24> GetHessian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 12, 24> J;
        J.block<12, 12>(0, 0) = _jacobianOplusXi;
        J.block<12, 12>(0, 12) = _jacobianOplusXj;
        return J.transpose() * _information * J;
    }

    Eigen::Matrix<double, 12, 24> GetJacobian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 12, 24> J;
        J.block<12, 12>(0, 0) = _jacobianOplusXi;
        J.block<12, 12>(0, 12) = _jacobianOplusXj;
        return J;
    }
};

class EdgeMonoOnlyPose : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPoseVel>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeMonoOnlyPose(const Eigen::Vector3f &Xw_): Xw(Xw_.cast<double>()) {}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexPoseVel* v = static_cast<const VertexPoseVel*>(_vertices[0]);
        const Eigen::Vector2d obs(_measurement);
        _error = obs - v->estimate().Project(Xw);
    }

    virtual void linearizeOplus();

    bool isDepthPositive()
    {
        const VertexPoseVel* v = static_cast<const VertexPoseVel*>(_vertices[0]);
        return v->estimate().isDepthPositive(Xw);
    }

    Eigen::Matrix<double, 2, 12> GetJacobian()
    {
        linearizeOplus();
        return _jacobianOplusXi;
    }
    
public:
    const Eigen::Vector3d Xw;
};

class EdgeStereoOnlyPose : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPoseVel>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeStereoOnlyPose(const Eigen::Vector3f &Xw_): Xw(Xw_.cast<double>()) {}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexPoseVel* v = static_cast<const VertexPoseVel*>(_vertices[0]);
        const Eigen::Vector3d obs(_measurement);
        _error = obs - v->estimate().ProjectStereo(Xw);
    }

    virtual void linearizeOplus();

    Eigen::Matrix<double, 3, 12> GetJacobian()
    {
        linearizeOplus();
        return _jacobianOplusXi;
    }
    
public:
    const Eigen::Vector3d Xw;
};

class EdgeMonoGPOnlyPose : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPoseVel, VertexPoseVel>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeMonoGPOnlyPose(const Eigen::Vector3f &Xw_, int cam_idx_, double t_, GaussianProcess* gp_): 
                    Xw(Xw_.cast<double>()), cam_idx(cam_idx_), t(t_), gp(gp_) {}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();

    virtual void linearizeOplus();

    bool isDepthPositive()
    {
        const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
        const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);

        return v1->estimate().isDepthPositive(Xw, cam_idx) && v2->estimate().isDepthPositive(Xw, cam_idx);
    }

    Eigen::Matrix<double, 2, 24> GetJacobian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 2, 24> J;
        J.block<2, 12>(0, 0) = _jacobianOplusXi;
        J.block<2, 12>(0, 12) = _jacobianOplusXj;
        const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
        const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
        std::cout << "prev v: " << v1->estimate().Vel.transpose() << std::endl;
        std::cout << "next v: " << v2->estimate().Vel.transpose() << std::endl;
        std::cout << "prev pose: " << v1->estimate().Twb.matrix3x4() << std::endl;
        std::cout << "next pose: " << v2->estimate().Twb.matrix3x4() << std::endl;
        std::cout << "obs: " << _measurement.transpose() << std::endl;
        std::cout << "Xw: " << Xw.transpose() << std::endl;
        return J;
    }

public:
    const Eigen::Vector3d Xw;
    const int cam_idx;
    const double t;
    GaussianProcess* gp;
};

class EdgeMonoGPExtrinsic : public g2o::BaseMultiEdge<2, Eigen::Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeMonoGPExtrinsic(int cam_idx_, double t_, GaussianProcess* gp_): 
                    cam_idx(cam_idx_), t(t_), gp(gp_) { resize(4); }
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();

    virtual void linearizeOplus();

    bool isDepthPositive()
    {
        const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
        const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* vPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[2]);
        const VertexExtrinsic* v3 = static_cast<const VertexExtrinsic*>(_vertices[3]);

        return v1->estimate().isDepthPositive(vPoint->estimate(), cam_idx, v3->estimate()) &&
                 v2->estimate().isDepthPositive(vPoint->estimate(), cam_idx, v3->estimate());
    }

    Eigen::Matrix<double, 2, 33> GetJacobian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 2, 33> J;
        J.block<2, 12>(0, 0) = _jacobianOplus[0];
        J.block<2, 12>(0, 12) = _jacobianOplus[1];
        J.block<2, 3>(0, 24) = _jacobianOplus[2];
        J.block<2, 6>(0, 27) = _jacobianOplus[3];
        /*const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);*/
        /*const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);*/
        /*const PoseVelocity& f1 = v1->estimate();*/
        /*const PoseVelocity& f2 = v2->estimate(); */
        /*const g2o::VertexSBAPointXYZ* vPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[2]);*/
        /*const Sophus::SE3d Twb = gp->QueryPose(f1.Twb, f2.Twb, f1.Vel, f2.Vel, f1.time, f2.time, t);*/
        /*Vector6d v3 = f2.Vel + (Vector6d() << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished();*/
        /*const Sophus::SE3d Twb2 = gp->QueryPose(f1.Twb, f2.Twb, f1.Vel, v3, f1.time, f2.time, t);*/
        /*std::cout << "prev v: " << v1->estimate().Vel.transpose() << std::endl;*/
        /*std::cout << "next v: " << v2->estimate().Vel.transpose() << std::endl;*/
        /*std::cout << "prev pose: " << v1->estimate().Twb.matrix3x4() << std::endl;*/
        /*std::cout << "next pose: " << v2->estimate().Twb.matrix3x4() << std::endl;*/
        /*std::cout << "cur pose: " << Twb.matrix3x4() << std::endl;*/
        /*std::cout << "cur pose2: " << Twb2.matrix3x4() << std::endl;*/
        /*std::cout << "obs: " << _measurement.transpose() << std::endl;*/
        /*std::cout << "Xw: " << vPoint->estimate().transpose() << std::endl;*/
        return J;
    }

public:
    const int cam_idx;
    const double t;
    GaussianProcess* gp;
};

class EdgeMonoGP : public g2o::BaseMultiEdge<2, Eigen::Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeMonoGP(int cam_idx_, double t_, GaussianProcess* gp_): 
                    cam_idx(cam_idx_), t(t_), gp(gp_) { resize(3); }
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();

    virtual void linearizeOplus();

    bool isDepthPositive()
    {
        const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
        const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* vPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[2]);

        return v1->estimate().isDepthPositive(vPoint->estimate(), cam_idx) &&
                 v2->estimate().isDepthPositive(vPoint->estimate(), cam_idx);
    }

    Eigen::Matrix<double, 2, 27> GetJacobian()
    {
        linearizeOplus();
        Eigen::Matrix<double, 2, 27> J;
        J.block<2, 12>(0, 0) = _jacobianOplus[0];
        J.block<2, 12>(0, 12) = _jacobianOplus[1];
        J.block<2, 3>(0, 24) = _jacobianOplus[2];
        const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
        const VertexPoseVel* v2 = static_cast<const VertexPoseVel*>(_vertices[1]);
        const PoseVelocity& f1 = v1->estimate();
        const PoseVelocity& f2 = v2->estimate(); 
        const g2o::VertexSBAPointXYZ* vPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[2]);
        const Sophus::SE3d Twb = gp->QueryPose(f1.Twb, f2.Twb, f1.Vel, f2.Vel, f1.time, f2.time, t);
        Vector6d v3 = f2.Vel + (Vector6d() << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished();
        const Sophus::SE3d Twb2 = gp->QueryPose(f1.Twb, f2.Twb, f1.Vel, v3, f1.time, f2.time, t);
        std::cout << "prev v: " << v1->estimate().Vel.transpose() << std::endl;
        std::cout << "next v: " << v2->estimate().Vel.transpose() << std::endl;
        std::cout << "prev pose: " << v1->estimate().Twb.matrix3x4() << std::endl;
        std::cout << "next pose: " << v2->estimate().Twb.matrix3x4() << std::endl;
        std::cout << "cur pose: " << Twb.matrix3x4() << std::endl;
        std::cout << "cur pose2: " << Twb2.matrix3x4() << std::endl;
        std::cout << "obs: " << _measurement.transpose() << std::endl;
        std::cout << "Xw: " << vPoint->estimate().transpose() << std::endl;
        return J;
    }

public:
    const int cam_idx;
    const double t;
    GaussianProcess* gp;
};

class EdgeStereoGP : public g2o::BaseMultiEdge<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeStereoGP(int cam_idx_, double t_, GaussianProcess* gp_): 
                    cam_idx(cam_idx_), t(t_), gp(gp_) { resize(3); }
    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();

    virtual void linearizeOplus();

public:
    const int cam_idx;
    const double t;
    GaussianProcess* gp;
};

class EdgeMono : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPoseVel, g2o::VertexSBAPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeMono() {}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[1]);
        const Eigen::Vector2d obs(_measurement);
        _error = obs - v1->estimate().Project(v2->estimate());
    }

    bool isDepthPositive()
    {
        const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
        const g2o::VertexSBAPointXYZ* vPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[1]);

        return v1->estimate().isDepthPositive(vPoint->estimate());
    }

    virtual void linearizeOplus();
};

class EdgeStereo : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, VertexPoseVel, g2o::VertexSBAPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeStereo() {}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexPoseVel* v1 = static_cast<const VertexPoseVel*>(_vertices[0]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[1]);
        const Eigen::Vector3d obs(_measurement);
        _error = obs - v1->estimate().ProjectStereo(v2->estimate());
    }

    virtual void linearizeOplus();

};

class EdgeExtrinsicPrior: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexExtrinsic>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeExtrinsicPrior(Sophus::SO3d R): R_(R.inverse()) {}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError()
    {
        const VertexExtrinsic* v = static_cast<const VertexExtrinsic*>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        _error = (R_ * T.so3()).log();
    }

    virtual void linearizeOplus()
    {
        _jacobianOplusXi.block<3, 3>(0, 0) = Eigen::Matrix3d::Zero();
        _jacobianOplusXi.block<3, 3>(0, 3) = RightJacobianSO3(_error).inverse();
    }

public:
    Sophus::SO3d R_;
};

class EdgeVelocity : public g2o::BaseUnaryEdge<1, double, VertexPoseVel>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeVelocity() { A << 0, 0, 1, 0, 0, 0; }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError()
    {
        const VertexPoseVel* v = static_cast<const VertexPoseVel*>(_vertices[0]);
        _error = A * v->estimate().Vel;
    }

    virtual void linearizeOplus()
    {
        _jacobianOplusXi.block<1, 6>(0, 0) = Eigen::Matrix<double, 1, 6>::Zero();
        _jacobianOplusXi.block<1, 6>(0, 6) = A;
    }

public:
    Eigen::Matrix<double, 1, 6> A;
};

class EdgeVelReproj : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexVel>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeVelReproj(Sophus::SE3d T_, double dt_, Eigen::Vector3d Xw_, int cam_, MultiFrame* pf_): T(T_), dt(dt_), Xw(Xw_), cam(cam_), pf(pf_) {}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError()
    {
        const Eigen::Vector2d obs(_measurement);
        const VertexVel* v = static_cast<const VertexVel*>(_vertices[0]);
        Eigen::Vector3d Xc = (T * Sophus::SE3d::exp(v->estimate() * dt) * MultiFrame::mTbc[cam].cast<double>()).inverse() * Xw;
        _error = obs - pf->mvpCamera[cam]->project(Xc);
    }

    virtual void linearizeOplus();

public:
    Sophus::SE3d T;
    double dt;
    Eigen::Vector3d Xw;
    int cam;

    MultiFrame* pf;
};

} //namespace ORB_SLAM2

#endif // G2OTYPES_H
