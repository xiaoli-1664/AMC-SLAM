#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <yaml-cpp/yaml.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "System.h"

void UpdateConfigFile(const string& strConfigFIle, const string& strSeqPath)
{
    YAML::Node config = YAML::LoadFile(strConfigFIle);
    std::cout << strConfigFIle << std::endl;
    YAML::Node seq = YAML::LoadFile(strSeqPath + "orb_stereo_calib.yaml");
    std::cout << strSeqPath + "orb_stereo_calib.yaml" << std::endl;
    config["Camera.bf"] = seq["Camera.bf"];
    std::cout << "Camera.bf: " << config["Camera.bf"] << std::endl;
    config["dataset"] = strSeqPath;

    std::ofstream fout(strConfigFIle);
    fout << "%YAML:1.0" << std::endl;
    fout << config;
    fout.close();
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cerr << std::endl
                  << "Usage: ./multicam_amv ./amv path_to_vocabulary path_to_settings seq_path index" << std::endl;
        return 1;
    }


    vector<vector<string>> vstrImageFilenames;
    vector<vector<double>> vvTimestamps;

    UpdateConfigFile(argv[2], argv[3]);

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MULTICAMERA, false);

    SLAM.LoadAmvImages(vstrImageFilenames, vvTimestamps);

    std::cout << std::endl
              << "-------" << std::endl
              << "Start processing sequence ..." << std::endl
              << "-------" << std::endl;

    vector<float> vTimesTrack;
    int nImages = vstrImageFilenames.back().size();
    vTimesTrack.resize(nImages);

    vector<cv::Mat> imgs(vstrImageFilenames.size(), cv::Mat());
    vector<double> vTimestamps(vvTimestamps.size(), 0);

    for (int ni = 0; ni < nImages; ++ni)
    {
        // std::cout << std::endl
        //           << "-------" << std::endl
        //           << "Frame: " << ni << "/" << nImages << std::endl
        //           << "-------" << std::endl;
        for (size_t i = 0; i < vstrImageFilenames.size(); ++i)
        {
            if (ni == 0 && i < vstrImageFilenames.size()-2)
                continue;
            imgs[i] = cv::imread(vstrImageFilenames[i][ni], cv::IMREAD_UNCHANGED);
            if (i < vvTimestamps.size()) vTimestamps[i] = vvTimestamps[i][ni];

            if (imgs[i].empty())
            {
                std::cerr << std::endl
                          << "Failed to load image at: " << vstrImageFilenames[i][ni] << std::endl;
                return 1;
            }
        }

        double t_resize = 0.0, t_track = 0.0;

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        SLAM.TrackMultiCamera(imgs, vTimestamps);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        vTimesTrack[ni] = ttrack;

        double T = 0;
        if (ni < nImages - 1)
            T = vvTimestamps.back()[ni+1]-vvTimestamps.back()[ni];
        else if (ni > 0)
            T = vvTimestamps.back()[ni]-vvTimestamps.back()[ni-1];
        
        if (ttrack < T)
            usleep((T-ttrack)*1e6);
    }

    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;


    // Save camera trajectory
    // SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");
    int index = std::stoi(argv[4]);
    int num = index / 25;
    index = index % 25;
    // SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveTrajectoryTUM("CameraTrajectory_" + std::to_string(index) + "_" + std::to_string(num) + ".txt");

    return 0;

    
}
