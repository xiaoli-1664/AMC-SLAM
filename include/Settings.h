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

#ifndef ORB_SLAM3_SETTINGS_H
#define ORB_SLAM3_SETTINGS_H


// Flag to activate the measurement of time in each process (track,localmap, place recognition).
//#define REGISTER_TIMES

#include "CameraModels/GeometricCamera.h"
#include "sophus/se3.hpp"

#include <opencv2/core/hal/interface.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

namespace ORB_SLAM3 {

    class System;

    //TODO: change to double instead of float

    class Settings {
    public:
        /*
         * Enum for the different camera types implemented
         */
        enum CameraType {
            PinHole = 0,
            Rectified = 1,
            KannalaBrandt = 2
        };

        /*
         * Delete default constructor
         */
        Settings() = delete;

        /*
         * Constructor from file
         */
        Settings(const std::string &configFile, const int& sensor);

        /*
         * Ostream operator overloading to dump settings to the terminal
         */
        friend std::ostream &operator<<(std::ostream &output, const Settings &s);

        /*
         * Getter methods
         */
        CameraType cameraType() {return cameraType_;}
        std::vector<GeometricCamera*> camera() {return calibration_;}
        /*cv::Mat camera1DistortionCoef() {return cv::Mat(vPinHoleDistorsion1_.size(),1,CV_32F,vPinHoleDistorsion1_.data());}*/
        std::vector<cv::Mat> cameraDistortionCoef() {
            std::vector<cv::Mat> ret;
            for (int i = 0; i < vPinHoleDistorsion_.size(); ++i)
            {
                ret.push_back(cv::Mat(vPinHoleDistorsion_[i].size(), 1, CV_32F, vPinHoleDistorsion_[i].data()));
            }
            return ret;
        }

        Sophus::SE3f Tlr() {return Tlr_;}
    std::vector<Sophus::SE3f> Tbc() {return Tbc_;}
        float bf() {return bf_;}
        float b() {return b_;}
        float thDepth() {return thDepth_;}
    cv::Mat mapX(int i) {return vMapX_[i];}
    cv::Mat mapY(int i) {return vMapY_[i];}

    cv::Size newImSize() {return newImSize_.back();};

        bool needToUndistort() {return bNeedToUndistort_;}

        float fps() {return fps_;}
        bool rgb() {return bRGB_;}
        bool needToResize() {return bNeedToResize1_;}
        bool needToRectify() {return bNeedToRectify_;}

        float noiseGyro() {return noiseGyro_;}
        float noiseAcc() {return noiseAcc_;}
        float gyroWalk() {return gyroWalk_;}
        float accWalk() {return accWalk_;}
        float imuFrequency() {return imuFrequency_;}
        bool insertKFsWhenLost() {return insertKFsWhenLost_;}

        float depthMapFactor() {return depthMapFactor_;}

        int nFeatures() {return nFeatures_;}
        int nLevels() {return nLevels_;}
        float initThFAST() {return initThFAST_;}
        float minThFAST() {return minThFAST_;}
        float scaleFactor() {return scaleFactor_;}

        float keyFrameSize() {return keyFrameSize_;}
        float keyFrameLineWidth() {return keyFrameLineWidth_;}
        float graphLineWidth() {return graphLineWidth_;}
        float pointSize() {return pointSize_;}
        float cameraSize() {return cameraSize_;}
        float cameraLineWidth() {return cameraLineWidth_;}
        float viewPointX() {return viewPointX_;}
        float viewPointY() {return viewPointY_;}
        float viewPointZ() {return viewPointZ_;}
        float viewPointF() {return viewPointF_;}
        float imageViewerScale() {return imageViewerScale_;}

        std::string atlasLoadFile() {return sLoadFrom_;}
        std::string atlasSaveFile() {return sSaveto_;}

        float thFarPoints() {return thFarPoints_;}

    private:
        template<typename T>
        T readParameter(cv::FileStorage& fSettings, const std::string& name, bool& found,const bool required = true){
            cv::FileNode node = fSettings[name];
            if(node.empty()){
                if(required){
                    std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                    exit(-1);
                }
                else{
                    std::cerr << name << " optional parameter does not exist..." << std::endl;
                    found = false;
                    return T();
                }

            }
            else{
                found = true;
                return T();
            }
        }

        void readCamera(cv::FileStorage& fSettings);
        void readImageInfo(cv::FileStorage& fSettings);
        void readIMU(cv::FileStorage& fSettings);
        void readRGBD(cv::FileStorage& fSettings);
        void readORB(cv::FileStorage& fSettings);
        void readViewer(cv::FileStorage& fSettings);
        void readLoadAndSave(cv::FileStorage& fSettings);
        void readOtherParameters(cv::FileStorage& fSettings);

        void precomputeRectificationMaps();

        int sensor_;
        CameraType cameraType_;     //Camera type

        /*
         * Visual stuff
         */
        /*GeometricCamera* calibration1_, *calibration2_;   //Camera calibration*/
        /*GeometricCamera* originalCalib1_, *originalCalib2_;*/
        std::vector<GeometricCamera*> calibration_;
        std::vector<GeometricCamera*> originalCalib_;
        /*std::vector<float> vPinHoleDistorsion1_, vPinHoleDistorsion2_;*/
        std::vector<std::vector<float>> vPinHoleDistorsion_;

        std::vector<cv::Size> originalImSize_, newImSize_;
        float fps_;
        bool bRGB_;

        bool bNeedToUndistort_;
        bool bNeedToRectify_;
        bool bNeedToResize1_, bNeedToResize2_;

        Sophus::SE3f Tlr_;
        std::vector<Sophus::SE3f> Tbc_;
        float thDepth_;
        float bf_, b_;

        /*
         * Rectification stuff
         */
        /*cv::Mat M1l_, M2l_;*/
        /*cv::Mat M1r_, M2r_;*/
    std::vector<cv::Mat> vMapX_;
    std::vector<cv::Mat> vMapY_;

        /*
         * Inertial stuff
         */
        float noiseGyro_, noiseAcc_;
        float gyroWalk_, accWalk_;
        float imuFrequency_;
        bool insertKFsWhenLost_;

        /*
         * RGBD stuff
         */
        float depthMapFactor_;

        /*
         * ORB stuff
         */
        int nFeatures_;
        float scaleFactor_;
        int nLevels_;
        int initThFAST_, minThFAST_;

        /*
         * Viewer stuff
         */
        float keyFrameSize_;
        float keyFrameLineWidth_;
        float graphLineWidth_;
        float pointSize_;
        float cameraSize_;
        float cameraLineWidth_;
        float viewPointX_, viewPointY_, viewPointZ_, viewPointF_;
        float imageViewerScale_;

        /*
         * Save & load maps
         */
        std::string sLoadFrom_, sSaveto_;

        /*
         * Other stuff
         */
        float thFarPoints_;
        int nCam;

    };
};


#endif //ORB_SLAM3_SETTINGS_H
