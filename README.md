# Asynchronous Multi-Camera SLAM Using Sparse Gaussian Process Regression

**Abstract**: Conventional VSLAM systems frequently experience drift and tracking failures in complex environments, which limits their overall effectiveness. Multi-camera systems have enhanced VSLAM accuracy by incorporating diverse viewpoints, but they generally rely on synchronous data capture, restricting their applicability in multi-sensor setups where asynchronous data acquisition is necessary. To address this limitation, we propose a continuous-time Asynchronous Multi-Camera SLAM (AMC-SLAM) framework using sparse Gaussian process regression. Our method integrates outlier removal, continuous-time trajectory optimization, and multi-view loop closing to achieve robust pose estimation. By combining Gaussian process interpolation with bundle adjustment, our approach strengthens inter-camera data correlation and reduces state variables, while our derived analytical Jacobians enhance optimization efficiency. Experimental results on the AMV-Bench dataset demonstrate absolute translation error below 0.5% over a 10 km trajectory, indicating significant accuracy improvements over existing stereo and multi-camera SLAM systems. These findings underscore the potential of AMC-SLAM for high-precision, robust applications in challenging environments.

This repository currently contains raw results and example programs.

The full implementation will be publicly released once the paper is accepted for publication.

This project is based on ORBSLAM3 (https://github.com/UZ-SLAMLab/ORB_SLAM3) and inherits the same licensing terms.

