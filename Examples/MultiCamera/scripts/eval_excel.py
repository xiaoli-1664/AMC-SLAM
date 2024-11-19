import subprocess
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Evaluate the amv-bench dataset trajectory and save the results to an Excel file.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--folder', '-f')
args = parser.parse_args()
folder = args.folder

seq_names = []

for i in range(0, 12):
    seq_names.append('day_no_rain_{}/'.format(i))

for i in range(0, 6):
    seq_names.append('day_rain_{}/'.format(i))

for i in range(0, 4):
    seq_names.append('hwy_no_rain_{}/'.format(i))

for i in range(0, 3):
    seq_names.append('hwy_rain_{}/'.format(i))


def calculate_mean(tum_file_path, gt_file_path, mode):
    # 使用evo_ape命令计算APE的RMSE
    ate_cmd = f"evo_ape tum {tum_file_path} {gt_file_path} -va"
    rpe_trans_cmd = f"evo_rpe tum {tum_file_path} {gt_file_path} -va -r trans_part --delta 10 --delta_unit f"
    rpe_rot_cmd = f"evo_rpe tum {tum_file_path} {gt_file_path} -va -r angle_deg --delta 10 --delta_unit f"
    length_cmd = f"evo_traj tum {gt_file_path} -v"

    try:
        # 执行命令并捕获输出
        if mode == 'ate':
            result = subprocess.run(ate_cmd, shell=True,
                                    capture_output=True, text=True, check=True)
        elif mode == 'rpe.trans':
            result = subprocess.run(rpe_trans_cmd, shell=True,
                                    capture_output=True, text=True, check=True)
        elif mode == 'rpe.rot':
            result = subprocess.run(rpe_rot_cmd, shell=True,
                                    capture_output=True, text=True, check=True)
        elif mode == 'length':
            result = subprocess.run(length_cmd, shell=True,
                                    capture_output=True, text=True, check=True)

            # 从输出中提取RMSE值
        if mode == 'length':
            for line in result.stdout.split('\n'):
                if 'length' in line:
                    return float(line.split()[-1])
        else:
            for line in result.stdout.split('\n'):
                if 'mean' in line:
                    return float(line.split()[-1])

        raise ValueError("未能从evo输出中找到mean值")

    except subprocess.CalledProcessError as e:
        print(f"执行evo命令时发生错误: {e}")
        print("错误输出:")
        print(e.stderr)
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None


def process_multiple_trajectories(trajectory_folder, output_excel):
    # 获取文件夹中所有的.txt文件
    tum_files = []
    gt_files = []
    for n in range(0, 3):
        i = 0
        for seq_name in seq_names:
            gt_files.append(os.path.join(
                gt_path, seq_name, 'poses_and_times.txt'))
            tum_files.append(trajectory_folder.format(folder, i, n))
            i += 1

    results = []

    ates = [0] * 25
    rpes_trans = [0] * 25
    rpes_rot = [0] * 25
    length = [0] * 25

    for i in range(len(tum_files)):
        ate = calculate_mean(tum_files[i], gt_files[i], 'ate')
        rpe_trans = calculate_mean(tum_files[i], gt_files[i], 'rpe.trans')
        rpe_rot = calculate_mean(tum_files[i], gt_files[i], 'rpe.rot')
        j = i % 25
        length[j] = calculate_mean(tum_files[i], gt_files[i], 'length')
        ates[j] += ate
        rpes_trans[j] += rpe_trans
        rpes_rot[j] += rpe_rot

    for i in range(0, 25):
        results.append({'Trajectory': seq_names[i].rstrip(
            '/'), 'ATE MEAN': ates[i] / 3, 'RPE trans': rpes_trans[i] / 3, 'RPE rot': rpes_rot[i] / 3, 'length': length[i]})

    # 创建DataFrame并保存到Excel
    df = pd.DataFrame(results)
    with pd.ExcelWriter(output_excel, mode='a', engine='openpyxl') as writer:
        sheet_name = folder
        if sheet_name in writer.book.sheetnames:
            writer.book.remove(writer.book[sheet_name])
        df.to_excel(writer, sheet_name=folder, index=False)
    print(f"结果已保存到 {output_excel}")

# 使用示例


traj_path = '/home/ljj/source_code/AMC-GP-SLAM/Examples/MultiCamera/{}/CameraTrajectory_{}_{}.txt'
gt_path = '/home/ljj/dataset/amv/val_halfrez'

output_excel = '/home/ljj/source_code/AMC-GP-SLAM/Examples/MultiCamera/result.xlsx'
output_excel = output_excel.format(folder, folder)

process_multiple_trajectories(traj_path, output_excel)
