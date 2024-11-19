import subprocess
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Evaluate the amv-bench dataset trajectory and save the results to an Excel file.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--folder', '-f', nargs='+')
parser.add_argument('--index', '-i')
args = parser.parse_args()
folder = args.folder
print(type(folder))
index = args.index


# run_cmd = 'evo_rpe tum {} {} -va -r trans_part --delta 10 --delta_unit f --plot --plot_mode=xy'
run_cmd = 'evo_traj tum {} {} --ref {} -a --plot --plot_mode=xy'

traj_path = '/home/ljj/source_code/AMC-GP-SLAM/Examples/MultiCamera/{}/CameraTrajectory_{}_{}.txt'

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

i = int(index)
#
# for n in range(0, 3):
#     i = int(index)
#     for j in range(i, len(seq_names)):
seq_name = seq_names[i]
seq_path = ''
seq_path = os.path.join(dataset_path, seq_name)
seq_path += 'poses_and_times.txt'
cmd = run_cmd.format(traj_path.format(
    folder[0], i, 0), traj_path.format(folder[1], i, 2), seq_path)
result = subprocess.run(cmd, shell=True, check=True, text=True)
# i += 1
