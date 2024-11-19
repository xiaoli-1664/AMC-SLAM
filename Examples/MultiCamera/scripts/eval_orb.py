import subprocess
import os

run_cmd = 'evo_ape tum {} {} -va --plot --plot_mode=xy'

traj_path = '/home/ljj/source_code/ORB_SLAM3/Examples/Stereo/amv_result/CameraTrajectory_{}_{}.txt'

dataset_path = '/home/ljj/dataset/amv/val_halfrez'

seq_names = []

for i in range(0, 12):
    seq_names.append('day_no_rain_{}/'.format(i))

for i in range(0, 6):
    seq_names.append('day_rain_{}/'.format(i))

for i in range(0, 4):
    seq_names.append('hwy_no_rain_{}/'.format(i))

for i in range(0, 3):
    seq_names.append('hwy_rain_{}/'.format(i))

i = 0

for n in range(0, 3):
    i = 0
    for seq_name in seq_names:
        seq_path = ''
        seq_path = os.path.join(dataset_path, seq_name)
        seq_path += 'poses_and_times.txt'
        cmd = run_cmd.format(seq_path, traj_path.format(i, n))
        subprocess.run(cmd, shell=True, check=True)
        i += 1
