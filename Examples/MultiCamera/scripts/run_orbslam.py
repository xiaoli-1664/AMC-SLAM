import subprocess
import os

run_cmd = '/home/ljj/source_code/ORB_SLAM3/Examples/Stereo/stereo_kitti {} {} {} {}'

voc_path = '/home/ljj/source_code/ORB_SLAM3/Vocabulary/ORBvoc.txt'

config_path = '/home/ljj/dataset/amv/val_halfrez/{}orb_stereo_calib.yaml'

dataset_path = '/home/ljj/dataset/amv/val_halfrez'

seq_names = []

for i in range(2, 12):
    seq_names.append('day_no_rain_{}/'.format(i))

for i in range(0, 6):
    seq_names.append('day_rain_{}/'.format(i))

for i in range(0, 4):
    seq_names.append('hwy_no_rain_{}/'.format(i))

for i in range(0, 3):
    seq_names.append('hwy_rain_{}/'.format(i))

i = 0

for n in range(0, 3):
    for seq_name in seq_names:
        seq_path = ''
        seq_path = os.path.join(dataset_path, seq_name)
        print(seq_path)
        cmd = run_cmd.format(
            voc_path, config_path.format(seq_name), seq_path, i)
        print(cmd)
        subprocess.run(cmd, shell=True, check=False)
        i += 1
