import subprocess
import os

download_cmd = 'aws s3 --no-sign-request cp --recursive {} {}'

dataset_path = 's3://pit30m/amv-slam/v0.2.0/val_halfrez'

download_path = '~/dataset/amv/val_halfrez'

seq_names = ['day_rain_5', 'hwy_no_rain_0', 'hwy_no_rain_1', 'hwy_no_rain_2', 'hwy_no_rain_3', 'hwy_rain_1', 'hwy_rain_2', 'hwy_rain_0']

for seq_name in seq_names:
    seq_path = os.path.join(dataset_path, seq_name)
    download_path = os.path.join(download_path, seq_name)
    cmd = download_cmd.format(seq_path, download_path)
    subprocess.run(cmd, shell=True, check=True)