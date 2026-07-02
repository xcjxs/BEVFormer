import json
import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 禁止显示窗口，提高并行性能
import matplotlib.pyplot as plt
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
import argparse
from tqdm import tqdm
import cv2
from multiprocessing import Pool, cpu_count
from functools import partial

# 全局变量（用于单进程模式，多进程模式下在子进程中重新创建）
nusc = None
DATAROOT = None
VERSION = None


def get_color(category_name: str):
    """官方颜色映射"""
    colormap = {
        'car': [255, 158, 0],
        'truck': [255, 61, 99],
        'construction_vehicle': [0, 0, 142],
        'bus': [102, 102, 156],
        'trailer': [153, 153, 153],
        'barrier': [255, 0, 0],
        'motorcycle': [0, 0, 230],
        'bicycle': [119, 11, 32],
        'pedestrian': [220, 20, 60],
        'traffic_cone': [255, 165, 0]
    }
    return colormap.get(category_name, [0, 0, 0])


def render_bev_to_ax(sample_token, ax_gt, ax_pred, pred_data, axes_limit=40):
    """直接在Axes上绘制BEV点云和GT/Pred框（无临时文件）"""
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_path, boxes_gt, _ = nusc.get_sample_data(lidar_token, selected_anntokens=None)
    pc = LidarPointCloud.from_file(lidar_path)
    points = pc.points[:3, :]
    mask = (np.abs(points[0, :]) < axes_limit) & (np.abs(points[1, :]) < axes_limit)
    points = points[:, mask]
    colors = (points[2, :] - np.min(points[2, :])) / (np.max(points[2, :]) - np.min(points[2, :]) + 1e-6)
    ax_gt.scatter(points[0, :], points[1, :], c=colors, s=1, cmap='viridis', alpha=0.8)
    ax_pred.scatter(points[0, :], points[1, :], c=colors, s=1, cmap='viridis', alpha=0.8)

    # GT boxes (green)
    for box in boxes_gt:
        c = np.array([0, 255, 0]) / 255.0
        box.render(ax_gt, view=np.eye(4), colors=(c, c, c))

    # Pred boxes (blue)
    if sample_token in pred_data['results']:
        for record in pred_data['results'][sample_token]:
            if record['detection_score'] > 0.3:
                box = Box(record['translation'], record['size'], Quaternion(record['rotation']),
                          name=record['detection_name'], token='predicted')
                c = np.array([0, 0, 255]) / 255.0
                box.render(ax_pred, view=np.eye(4), colors=(c, c, c))

    for ax in (ax_gt, ax_pred):
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)
        ax.axis('off')
        ax.set_aspect('equal')
    ax_gt.set_title('BEV GT (Green)')
    ax_pred.set_title('BEV Pred (Blue)')


def render_camera_to_ax(sample_token, cam_name, ax, pred_data, score_thresh=0.3):
    """绘制单张相机视图（仅预测框）"""
    sample = nusc.get('sample', sample_token)
    sd_token = sample['data'][cam_name]
    sd_record = nusc.get('sample_data', sd_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    data_path = nusc.get_sample_data_path(sd_token)
    im = Image.open(data_path)
    imsize = (sd_record['width'], sd_record['height'])
    ax.imshow(im)

    if sample_token in pred_data['results']:
        for record in pred_data['results'][sample_token]:
            if record['detection_score'] > score_thresh:
                box = Box(record['translation'], record['size'], Quaternion(record['rotation']),
                          name=record['detection_name'], token='predicted')
                # 转换到相机坐标系
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)
                if box_in_image(box, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):
                    c = np.array(get_color(box.name)) / 255.0
                    box.render(ax, view=cam_intrinsic, normalize=True, colors=(c, c, c))

    ax.set_xlim(0, im.size[0])
    ax.set_ylim(im.size[1], 0)
    ax.axis('off')
    ax.set_title(f'{cam_name} (Pred)')


def render_single_sample(sample_token, pred_data, out_dir, verbose=False):
    """渲染单个样本并保存图片"""
    fig, axes = plt.subplots(3, 4, figsize=(32, 18))
    # BEV
    render_bev_to_ax(sample_token, axes[0, 0], axes[0, 1], pred_data)
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')
    # 6 Camera views
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    for idx, cam in enumerate(cams):
        row = 1 + idx // 3
        col = idx % 3
        render_camera_to_ax(sample_token, cam, axes[row, col], pred_data)
    axes[1, 3].axis('off')
    axes[2, 3].axis('off')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f'{sample_token}.png')
    plt.savefig(img_path, dpi=120, bbox_inches='tight')
    if verbose:
        plt.show()
    plt.close(fig)
    return img_path


def init_worker(dataroot, version):
    """多进程worker初始化：创建全局nusc实例"""
    global nusc
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)


def images_to_video(image_paths, output_video_path, fps=5):
    """将图片列表合成为视频，自动尝试多种编码器，使用绝对路径"""
    if not image_paths:
        print("图片列表为空，无法生成视频")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_video_path))
    os.makedirs(output_dir, exist_ok=True)

    # 读取第一帧获取尺寸
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        print(f"无法读取第一帧图片: {image_paths[0]}")
        return
    h, w, _ = first_frame.shape

    # 定义编码器候选（按优先级）
    codecs = [
        ('MJPG', '.avi'),   # 广泛支持
        ('XVID', '.avi'),
        ('mp4v', '.mp4'),
        ('avc1', '.mp4'),
        ('H264', '.mp4'),
    ]

    video_writer = None
    chosen_path = None

    for fourcc_str, ext in codecs:
        test_path = os.path.splitext(output_video_path)[0] + ext
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(test_path, fourcc, fps, (w, h))
        if writer.isOpened():
            video_writer = writer
            chosen_path = test_path
            print(f"使用编码器 {fourcc_str}，输出文件: {chosen_path}")
            break
        else:
            writer.release()  # 释放资源

    if video_writer is None:
        print("错误：所有编码器均无法创建视频文件，请检查OpenCV编译选项或安装编码库。")
        return

    # 写入所有帧
    for p in tqdm(image_paths, desc=f"生成视频 {os.path.basename(chosen_path)}"):
        img = cv2.imread(p)
        if img is not None:
            video_writer.write(img)
        else:
            print(f"警告：跳过无法读取的图片 {p}")

    video_writer.release()
    print(f"视频已成功保存至: {chosen_path}")


def compute_metrics(nusc, results, output_dir, eval_set='mini_val', config_name='detection_cvpr_2019'):
    """
    计算nuScenes检测指标（mAP, NDS等），并将结果保存到output_dir。
    """
    # 将结果写入临时JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(results, f)
        temp_path = f.name

    try:
        cfg = config_factory(config_name)
        nusc_eval = NuScenesEval(
            nusc,
            config=cfg,
            result_path=temp_path,
            eval_set=eval_set,
            output_dir=output_dir,
            verbose=True
        )
        metrics_summary = nusc_eval.main()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("\n========== 评估结果 ==========")
    print(f"mAP: {metrics_summary['mean_ap']:.4f}")
    print(f"NDS: {metrics_summary['nd_score']:.4f}")
    print("==============================\n")
    return metrics_summary

def process_scene_parallel(scene_token, pred_data, out_root, num_workers=None):
    """
    并行处理一个场景的所有样本。
    返回生成的图片路径列表（按时间顺序）。
    """
    scene = nusc.get('scene', scene_token)
    sample_tokens = []
    st = scene['first_sample_token']
    while st != '':
        sample_tokens.append(st)
        st = nusc.get('sample', st)['next']

    scene_dir = os.path.join(out_root, scene['name'])
    os.makedirs(scene_dir, exist_ok=True)

    if num_workers is None:
        num_workers = min(cpu_count(), len(sample_tokens))

    # 使用多进程池
    with Pool(processes=num_workers, initializer=init_worker, initargs=(DATAROOT, VERSION)) as pool:
        render_func = partial(render_single_sample, pred_data=pred_data, out_dir=scene_dir, verbose=False)
        img_paths = pool.map(render_func, sample_tokens)

    return img_paths


def main():
    parser = argparse.ArgumentParser(description='BEVFormer可视化优化版（支持多进程）')
    parser.add_argument('--dataroot', type=str, default='./data/nuscenes', help='nuScenes数据根目录')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='数据集版本')
    parser.add_argument('--result_json', type=str, required=True, help='预测结果JSON路径')
    parser.add_argument('--out_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--fps', type=int, default=5, help='视频帧率')
    parser.add_argument('--num_workers', type=int, default=4, help='并行进程数（设为1则单进程）')
    parser.add_argument('--eval', action='store_true', help='是否运行评估')
    args = parser.parse_args()

    # 设置全局变量（用于主进程和多进程初始化）
    global nusc, DATAROOT, VERSION
    DATAROOT = args.dataroot
    VERSION = args.version
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # 加载预测结果
    with open(args.result_json, 'r') as f:
        pred_data = json.load(f)

    # 遍历所有场景
    for scene in tqdm(nusc.scene, desc="Processing scenes"):
        scene_token = scene['token']
        img_paths = process_scene_parallel(scene_token, pred_data, args.out_dir, num_workers=args.num_workers)
        if img_paths:
            video_path = os.path.join(args.out_dir, f"{scene['name']}_video.mp4")
            images_to_video(img_paths, video_path, args.fps)

    # 评估（如果启用）
    if args.eval:
        eval_set = 'mini_val' if args.version == 'v1.0-mini' else 'val'
        compute_metrics(nusc, pred_data, args.out_dir, eval_set=eval_set)


if __name__ == '__main__':
    main()