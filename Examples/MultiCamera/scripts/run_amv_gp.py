import subprocess
import os

run_cmd = '/home/ljj/source_code/AMC-GP-SLAM/Examples/MultiCamera/MultiCamera {} {} {} {}'

voc_path = '/home/ljj/source_code/AMC-GP-SLAM/Vocabulary/ORBvoc.txt'

config_path = '/home/ljj/source_code/AMC-GP-SLAM/Examples/MultiCamera/orb_multicam.yaml'

dataset_path = '/home/ljj/dataset/amv/val_halfrez'

seq_names = []

# for i in range(0, 12):
# seq_names.append('day_no_rain_{}/'.format(i))

for i in range(4, 6):
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
        cmd = run_cmd.format(voc_path, config_path, seq_path, i)
        subprocess.run(cmd, shell=True, check=False)
        i += 1
