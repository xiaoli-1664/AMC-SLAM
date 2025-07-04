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

#include "Settings.h"

#include "CameraModels/Pinhole.h"
#include "CameraModels/KannalaBrandt8.h"

#include "Eigen/src/Core/Matrix.h"
#include "System.h"
#include "sophus/se3.hpp"

#include <nlohmann/json_fwd.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/eigen.hpp>

#include <iostream>

using namespace std;

namespace ORB_SLAM3 {

    template<>
    float Settings::readParameter<float>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return 0.0f;
            }
        }
        else if(!node.isReal()){
            std::cerr << name << " parameter must be a real number, aborting..." << std::endl;
            exit(-1);
        }
        else{
            found = true;
            return node.real();
        }
    }

    template<>
    int Settings::readParameter<int>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return 0;
            }
        }
        else if(!node.isInt()){
            std::cerr << name << " parameter must be an integer number, aborting..." << std::endl;
            exit(-1);
        }
        else{
            found = true;
            return node.operator int();
        }
    }

    template<>
    string Settings::readParameter<string>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return string();
            }
        }
        else if(!node.isString()){
            std::cerr << name << " parameter must be a string, aborting..." << std::endl;
            exit(-1);
        }
        else{
            found = true;
            return node.string();
        }
    }

    template<>
    cv::Mat Settings::readParameter<cv::Mat>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return cv::Mat();
            }
        }
        else{
            found = true;
            return node.mat();
        }
    }

    Settings::Settings(const std::string &configFile, const int& sensor) :
    bNeedToUndistort_(false), bNeedToRectify_(false), bNeedToResize1_(false), bNeedToResize2_(false) {
        sensor_ = sensor;

        //Open settings file
        cv::FileStorage fSettings(configFile, cv::FileStorage::READ);
        if (!fSettings.isOpened()) {
            cerr << "[ERROR]: could not open configuration file at: " << configFile << endl;
            cerr << "Aborting..." << endl;

            exit(-1);
        }
        else{
            cout << "Loading settings from " << configFile << endl;
        }

        readCamera(fSettings);
        cout << "\t-Loaded camera" << endl;

        readORB(fSettings);
        cout << "\t-Loaded ORB settings" << endl;
        readViewer(fSettings);
        cout << "\t-Loaded viewer settings" << endl;
        readLoadAndSave(fSettings);
        cout << "\t-Loaded Atlas settings" << endl;
        readOtherParameters(fSettings);
        cout << "\t-Loaded misc parameters" << endl;

        if(bNeedToRectify_){
            precomputeRectificationMaps();
            cout << "\t-Computed rectification maps" << endl;
        }

        cout << "----------------------------------" << endl;
    }

    void Settings::readCamera(cv::FileStorage &fSettings)
{
    bool found;
    nCam = readParameter<int>(fSettings, "Camera.number", found);
    cout << "Number of cameras: " << nCam << endl;

    vector<float> vCalibration;
    calibration_.resize(nCam + 1);
    originalCalib_.resize(nCam + 1);
    vPinHoleDistorsion_.resize(nCam + 1);
    originalImSize_.resize(nCam + 1);

    vector<string> vCamCalibFile;
    cv::FileNode node = fSettings["Camera.calibfile"];
    node >> vCamCalibFile;
    string dataset = fSettings["dataset"];

    for (int i = 0; i <= nCam; ++i)
    {
        vCamCalibFile[i] = dataset + vCamCalibFile[i];

        float fx, fy, cx, cy;
        std::ifstream f(vCamCalibFile[i]);
        if (!f.is_open())
        {
            std::cerr << "Cannot open the camera parameter file" << std::endl;
            return;
        }

        nlohmann::json data = nlohmann::json::parse(f);
        std::vector<float> vectorK = data["intrinsics"];
        
        fx = vectorK[0];
        fy = vectorK[4];
        cx = vectorK[2];
        cy = vectorK[5];

        vCalibration = {fx, fy, cx, cy};
        calibration_[i] = new Pinhole(vCalibration);
        originalCalib_[i] = new Pinhole(vCalibration);

        vPinHoleDistorsion_[i].resize(5);
        std::vector<float> vectorD = data["distortion_coefficients"];
        vPinHoleDistorsion_[i] = vectorD;

        int rows = data["image_size"][0];
        int cols = data["image_size"][1];
        originalImSize_[i].width = cols;
        originalImSize_[i].height = rows;
        newImSize_.push_back(originalImSize_[i]);

         std::vector<float> vectorTbc = data["sensor_to_vehicle"];

         Eigen::Matrix4f mat;
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                mat(i, j) = vectorTbc[i*4+j];
            }
        }
        Sophus::SE3f Tbc(mat);
        Tbc_.push_back(Tbc);
    }

    int newHeight = readParameter<int>(fSettings, "Camera.newHeight", found, false);
    if (found)
    {
        bNeedToResize1_ = true;
        for (int i = 0; i <= nCam; ++i)
        {
            newImSize_[i].height = newHeight;
        }
    }
    int newWidth = readParameter<int>(fSettings, "Camera.newWidth", found, false);
    if (found)
    {
        bNeedToResize1_ = true;
        for (int i = 0; i <= nCam; ++i)
        {
            newImSize_[i].width = newWidth;
        }
    }
    Tlr_ = Tbc_[nCam - 1].inverse() * Tbc_[nCam];
    std::cout << "Tlr: " << Tlr_.matrix3x4() << std::endl;
    fps_ = fSettings["Camera.fps"];
    fSettings["Camera.RGB"] >> bRGB_;

    b_ = Tlr_.translation().norm();
    bf_ = b_ * calibration_[nCam - 1]->getParameter(0);
    bNeedToRectify_ = true;

    thDepth_ = fSettings["ThDepth"];
}

    void Settings::readORB(cv::FileStorage &fSettings) {
        bool found;

        nFeatures_ = readParameter<int>(fSettings,"ORBextractor.nFeatures",found);
        scaleFactor_ = readParameter<float>(fSettings,"ORBextractor.scaleFactor",found);
        nLevels_ = readParameter<int>(fSettings,"ORBextractor.nLevels",found);
        initThFAST_ = readParameter<int>(fSettings,"ORBextractor.iniThFAST",found);
        minThFAST_ = readParameter<int>(fSettings,"ORBextractor.minThFAST",found);
    }

    void Settings::readViewer(cv::FileStorage &fSettings) {
        bool found;

        keyFrameSize_ = readParameter<float>(fSettings,"Viewer.KeyFrameSize",found);
        keyFrameLineWidth_ = readParameter<float>(fSettings,"Viewer.KeyFrameLineWidth",found);
        graphLineWidth_ = readParameter<float>(fSettings,"Viewer.GraphLineWidth",found);
        pointSize_ = readParameter<float>(fSettings,"Viewer.PointSize",found);
        cameraSize_ = readParameter<float>(fSettings,"Viewer.CameraSize",found);
        cameraLineWidth_ = readParameter<float>(fSettings,"Viewer.CameraLineWidth",found);
        viewPointX_ = readParameter<float>(fSettings,"Viewer.ViewpointX",found);
        viewPointY_ = readParameter<float>(fSettings,"Viewer.ViewpointY",found);
        viewPointZ_ = readParameter<float>(fSettings,"Viewer.ViewpointZ",found);
        viewPointF_ = readParameter<float>(fSettings,"Viewer.ViewpointF",found);
        imageViewerScale_ = readParameter<float>(fSettings,"Viewer.imageViewScale",found,false);

         if(!found)
            imageViewerScale_ = 1.0f;
    }

    void Settings::readLoadAndSave(cv::FileStorage &fSettings) {
        bool found;

        sLoadFrom_ = readParameter<string>(fSettings,"System.LoadAtlasFromFile",found,false);
        sSaveto_ = readParameter<string>(fSettings,"System.SaveAtlasToFile",found,false);
    }

    void Settings::readOtherParameters(cv::FileStorage& fSettings) {
        bool found;

        thFarPoints_ = readParameter<float>(fSettings,"System.thFarPoints",found,false);
    }

    void Settings::precomputeRectificationMaps() {
        std::vector<cv::Mat> vK;
        for (int i = 0; i <= nCam; ++i)
        {
            cv::Mat K = static_cast<Pinhole*>(calibration_[i])->toK();
            K.convertTo(K, CV_64F);
            vK.push_back(K);
        }
        
        cv::Mat cvTlr;
        cv::eigen2cv(Tlr_.inverse().matrix3x4(), cvTlr);
        cv::Mat R12 = cvTlr.rowRange(0, 3).colRange(0, 3);
        R12.convertTo(R12, CV_64F);
        cv::Mat t12 = cvTlr.rowRange(0, 3).col(3);
        t12.convertTo(t12, CV_64F);

        std::vector<cv::Mat> vR(nCam + 1);
        std::vector<cv::Mat> vP(nCam + 1);
        cv::Mat Q;

        cv::stereoRectify(vK[nCam - 1], cameraDistortionCoef()[nCam - 1], vK[nCam], cameraDistortionCoef()[nCam], newImSize_[nCam], R12, t12, vR[nCam-1], vR[nCam], vP[nCam-1], vP[nCam], Q, cv::CALIB_ZERO_DISPARITY, -1, newImSize_[nCam]);

        for (int i = 0; i < nCam - 1; ++i)
        {
             vP[i] = cv::getOptimalNewCameraMatrix(vK[i], cameraDistortionCoef()[i], newImSize_[i], 1, newImSize_[i], 0);
        }

        vMapX_.resize(nCam + 1);
        vMapY_.resize(nCam + 1);
        for (int i = 0; i <= nCam; ++i)
        {
             cv::initUndistortRectifyMap(vK[i], cameraDistortionCoef()[i], vR[i], vP[i].rowRange(0, 3).colRange(0, 3), newImSize_[i], CV_32F, vMapX_[i], vMapY_[i]);

            calibration_[i]->setParameter(vP[i].at<double>(0, 0), 0);
            calibration_[i]->setParameter(vP[i].at<double>(1, 1), 1);
            calibration_[i]->setParameter(vP[i].at<double>(0, 2), 2);
            calibration_[i]->setParameter(vP[i].at<double>(1, 2), 3);
        }

        bf_ = b_ * vP[nCam - 1].at<double>(0, 0);
}

    ostream &operator<<(std::ostream& output, const Settings& settings){
        output << "SLAM settings: " << endl;

        output << "\t-Camera 1 parameters (";
        if(settings.cameraType_ == Settings::PinHole || settings.cameraType_ ==  Settings::Rectified){
            output << "Pinhole";
        }
        else{
            output << "Kannala-Brandt";
        }
    int ncam = settings.nCam;
        output << ")" << ": [";
        for(size_t i = 0; i < settings.originalCalib_[settings.nCam-1]->size(); i++){
            output << " " << settings.originalCalib_[settings.nCam-1]->getParameter(i);
        }
        output << " ]" << endl;

        if(!settings.vPinHoleDistorsion_[ncam-1].empty()){
            output << "\t-Camera 1 distortion parameters: [ ";
            for(float d : settings.vPinHoleDistorsion_[ncam-1]){
                output << " " << d;
            }
            output << " ]" << endl;
        }

            if(!settings.vPinHoleDistorsion_[ncam-1].empty()){
                output << "\t-Camera 1 distortion parameters: [ ";
                for(float d : settings.vPinHoleDistorsion_[ncam-1]){
                    output << " " << d;
                }
                output << " ]" << endl;
            }

        output << "\t-Original image size: [ " << settings.originalImSize_[ncam-1].width << " , " << settings.originalImSize_[ncam-1].height << " ]" << endl;

        if(settings.bNeedToRectify_){
            output << "\t-Camera 1 parameters after rectification: [ ";
            for(size_t i = 0; i < settings.calibration_[ncam-1]->size(); i++){
                output << " " << settings.calibration_[ncam-1]->getParameter(i);
            }
            output << " ]" << endl;
        }
        else if(settings.bNeedToResize1_){
            output << "\t-Camera 1 parameters after resize: [ ";
            for(size_t i = 0; i < settings.calibration_[ncam-1]->size(); i++){
                output << " " << settings.calibration_[ncam-1]->getParameter(i);
            }
            output << " ]" << endl;

        }

        output << "\t-Sequence FPS: " << settings.fps_ << endl;

        output << "\t-Features per image: " << settings.nFeatures_ << endl;
        output << "\t-ORB scale factor: " << settings.scaleFactor_ << endl;
        output << "\t-ORB number of scales: " << settings.nLevels_ << endl;
        output << "\t-Initial FAST threshold: " << settings.initThFAST_ << endl;
        output << "\t-Min FAST threshold: " << settings.minThFAST_ << endl;

        return output;
    }
};
