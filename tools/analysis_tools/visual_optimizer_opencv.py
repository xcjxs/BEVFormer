import json
import os
import tempfile
import numpy as np
import cv2
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# 全局变量（多进程 worker 中重新创建）
nusc = None
DATAROOT = None
VERSION = None


def get_color(category_name: str):
    """官方颜色映射 (BGR)"""
    colormap = {
        'car': (0, 158, 255),
        'truck': (99, 61, 255),
        'construction_vehicle': (142, 0, 0),
        'bus': (156, 102, 102),
        'trailer': (153, 153, 153),
        'barrier': (0, 0, 255),
        'motorcycle': (230, 0, 0),
        'bicycle': (32, 11, 119),
        'pedestrian': (60, 20, 220),
        'traffic_cone': (0, 165, 255)
    }
    return colormap.get(category_name, (0, 0, 0))


def draw_bev_image(points, boxes, title, color_boxes, axes_limit=40, img_size=400):
    """
    绘制 BEV 图像（点云 + 框）。
    points: (3, N) 点云数组
    boxes: Box 对象列表
    title: 图像标题
    color_boxes: (B, G, R) 框颜色
    """
    # 创建白色背景
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    scale = img_size / (2 * axes_limit)
    cx, cy = img_size // 2, img_size // 2

    # 绘制点云（高度映射颜色）
    if points.shape[1] > 0:
        xs = (points[0, :] * scale + cx).astype(np.int32)
        ys = (-points[1, :] * scale + cy).astype(np.int32)  # y 轴反向
        valid = (xs >= 0) & (xs < img_size) & (ys >= 0) & (ys < img_size)
        xs, ys = xs[valid], ys[valid]
        z_vals = points[2, valid]
        if z_vals.size > 0:
            z_min, z_max = z_vals.min(), z_vals.max()
            if z_max > z_min:
                colors = ((z_vals - z_min) / (z_max - z_min) * 255).astype(np.uint8)
            else:
                colors = np.full(z_vals.shape, 128, dtype=np.uint8)
            # 逐点绘制（性能可接受）
            for x, y, c in zip(xs, ys, colors):
                cv2.circle(img, (x, y), 1, (int(c), int(c), int(c)), -1)

    # 绘制框
    for box in boxes:
        # 获取 BEV 下的 8 个角点（忽略 z）
        corners = box.corners()  # (3, 8)
        corners_bev = corners[:2, :]  # (2, 8)
        # 缩放到图像坐标
        x_pix = corners_bev[0, :] * scale + cx
        y_pix = -corners_bev[1, :] * scale + cy
        pts = np.stack([x_pix, y_pix], axis=1).astype(np.int32)
        # 绘制多边形（外轮廓）
        cv2.polylines(img, [pts], True, color_boxes, 2)
        # 绘制中心点（可选）
        cx_box, cy_box = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        cv2.circle(img, (int(cx_box), int(cy_box)), 3, color_boxes, -1)

    # 添加标题（顶部留白）
    title_h = 30
    img_with_title = np.ones((img_size + title_h, img_size, 3), dtype=np.uint8) * 255
    img_with_title[title_h:, :, :] = img
    cv2.putText(img_with_title, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return img_with_title


def render_bev_cv(sample_token, pred_data, axes_limit=40, img_size=400):
    """返回 BEV GT 和 Pred 两张图像（带标题）"""
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_path, boxes_gt, _ = nusc.get_sample_data(lidar_token, selected_anntokens=None)
    pc = LidarPointCloud.from_file(lidar_path)
    points = pc.points[:3, :]
    # 过滤范围
    mask = (np.abs(points[0, :]) < axes_limit) & (np.abs(points[1, :]) < axes_limit)
    points = points[:, mask]

    # GT 框
    gt_img = draw_bev_image(points, boxes_gt, "BEV GT (Green)", (0, 255, 0), axes_limit, img_size)

    # Pred 框（蓝色）
    pred_boxes = []
    if sample_token in pred_data['results']:
        for record in pred_data['results'][sample_token]:
            if record['detection_score'] > 0.3:
                box = Box(record['translation'], record['size'],
                          Quaternion(record['rotation']),
                          name=record['detection_name'], token='predicted')
                pred_boxes.append(box)
    pred_img = draw_bev_image(points, pred_boxes, "BEV Pred (Blue)", (255, 0, 0), axes_limit, img_size)

    return gt_img, pred_img


def render_camera_cv(sample_token, cam_name, pred_data, score_thresh=0.3, img_size=400):
    """返回相机视图（带标题和预测框）"""
    sample = nusc.get('sample', sample_token)
    sd_token = sample['data'][cam_name]
    sd_record = nusc.get('sample_data', sd_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    data_path = nusc.get_sample_data_path(sd_token)
    im = cv2.imread(data_path)
    if im is None:
        # 如果读取失败，创建灰色占位
        im = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128
    h, w = im.shape[:2]

    # 预测框绘制
    if sample_token in pred_data['results']:
        for record in pred_data['results'][sample_token]:
            if record['detection_score'] > score_thresh:
                box = Box(record['translation'], record['size'],
                          Quaternion(record['rotation']),
                          name=record['detection_name'], token='predicted')
                # 转换到相机坐标系
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

                # 检查是否在图像内
                if not box_in_image(box, cam_intrinsic, (w, h), vis_level=BoxVisibility.ANY):
                    continue

                # 获取 8 个角点并投影
                corners = box.corners()  # (3, 8)
                # 齐次坐标投影
                corners_h = np.vstack((corners, np.ones((1, 8))))  # (4, 8)
                pts_2d = cam_intrinsic @ corners_h[:3, :]  # (3, 8)
                pts_2d = pts_2d / pts_2d[2, :]  # 归一化
                pts_2d = pts_2d[:2, :].astype(np.int32).T  # (8, 2)

                # 绘制 3D 框的 12 条边（定义边索引）
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],   # 底面
                    [4, 5], [5, 6], [6, 7], [7, 4],   # 顶面
                    [0, 4], [1, 5], [2, 6], [3, 7]    # 竖边
                ]
                color = get_color(box.name)
                for edge in edges:
                    p1 = tuple(pts_2d[edge[0]])
                    p2 = tuple(pts_2d[edge[1]])
                    # 检查是否在图像内（至少有一个点在图像内即绘制，OpenCV 会自动裁剪）
                    cv2.line(im, p1, p2, color, 2)

                # 绘制标签（类别 + 分数）
                label = f"{box.name} {record['detection_score']:.2f}"
                # 在中心点上方绘制
                cx = int(np.mean(pts_2d[:, 0]))
                cy = int(np.mean(pts_2d[:, 1]))
                cv2.putText(im, label, (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # 缩放并添加标题
    title_h = 30
    target_w = img_size
    target_h = img_size
    # 保持宽高比缩放
    scale_ratio = min(target_w / w, target_h / h)
    new_w = int(w * scale_ratio)
    new_h = int(h * scale_ratio)
    if new_w > 0 and new_h > 0:
        im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        im_resized = np.ones((target_h, target_w, 3), dtype=np.uint8) * 128
    # 居中放置到目标尺寸
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 128
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = im_resized

    # 添加标题
    img_with_title = np.ones((target_h + title_h, target_w, 3), dtype=np.uint8) * 255
    img_with_title[title_h:, :, :] = canvas
    cv2.putText(img_with_title, cam_name, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return img_with_title


def assemble_frame(sample_token, pred_data, subplot_size=400):
    """
    组装 3 行 4 列的大图。
    返回 BGR 图像数组。
    """
    # 子图尺寸（含标题高度）
    title_h = 30
    content_size = subplot_size
    sub_h = content_size + title_h
    sub_w = content_size

    # 生成所有子图
    gt_bev, pred_bev = render_bev_cv(sample_token, pred_data, img_size=content_size)
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    cam_imgs = [render_camera_cv(sample_token, cam, pred_data, img_size=content_size) for cam in cams]

    # 创建空白图像（用于占位）
    blank = np.ones((sub_h, sub_w, 3), dtype=np.uint8) * 255

    # 第 0 行：GT, Pred, blank, blank
    row0 = np.hstack([gt_bev, pred_bev, blank, blank])
    # 第 1 行：3 个前相机 + blank
    row1 = np.hstack([cam_imgs[0], cam_imgs[1], cam_imgs[2], blank])
    # 第 2 行：3 个后相机 + blank
    row2 = np.hstack([cam_imgs[3], cam_imgs[4], cam_imgs[5], blank])

    # 垂直拼接
    full_img = np.vstack([row0, row1, row2])
    return full_img


def write_video_from_arrays(img_arrays, video_path, fps=5):
    """将图像数组列表写入视频"""
    if not img_arrays:
        print("没有图像，跳过视频生成。")
        return

    h, w = img_arrays[0].shape[:2]
    # 尝试多种编码器
    codecs = [
        ('MJPG', '.avi'),
        ('XVID', '.avi'),
        ('mp4v', '.mp4'),
        ('avc1', '.mp4'),
    ]
    video_writer = None
    chosen_path = None
    for fourcc_str, ext in codecs:
        test_path = os.path.splitext(video_path)[0] + ext
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(test_path, fourcc, fps, (w, h))
        if writer.isOpened():
            video_writer = writer
            chosen_path = test_path
            print(f"使用编码器 {fourcc_str}，输出: {chosen_path}")
            break
        else:
            writer.release()
    if video_writer is None:
        print("错误：无法创建视频文件。")
        return

    for img in tqdm(img_arrays, desc=f"写入视频 {os.path.basename(chosen_path)}"):
        video_writer.write(img)
    video_writer.release()
    print(f"视频保存至: {chosen_path}")


def process_scene_parallel(scene_token, pred_data, out_root, subplot_size, num_workers=None):
    """
    并行处理一个场景的所有样本，返回图像数组列表（按时间顺序）。
    同时可选保存每帧 PNG（默认不保存）。
    """
    scene = nusc.get('scene', scene_token)
    sample_tokens = []
    st = scene['first_sample_token']
    while st != '':
        sample_tokens.append(st)
        st = nusc.get('sample', st)['next']

    if num_workers is None:
        num_workers = min(cpu_count(), len(sample_tokens))

    # 使用多进程池渲染所有帧
    with Pool(processes=num_workers, initializer=init_worker, initargs=(DATAROOT, VERSION)) as pool:
        render_func = partial(assemble_frame, pred_data=pred_data, subplot_size=subplot_size)
        img_arrays = pool.map(render_func, sample_tokens)

    return img_arrays


def init_worker(dataroot, version):
    """多进程 worker 初始化"""
    global nusc
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)


def compute_metrics(nusc, results, output_dir, eval_set='mini_val', config_name='detection_cvpr_2019'):
    """计算 nuScenes 指标（与原版相同）"""
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


def main():
    parser = argparse.ArgumentParser(description='BEVFormer 可视化 (OpenCV 渲染版)')
    parser.add_argument('--dataroot', type=str, default='./data/nuscenes', help='nuScenes 数据根目录')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='数据集版本')
    parser.add_argument('--result_json', type=str, required=True, help='预测结果 JSON 路径')
    parser.add_argument('--out_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--fps', type=int, default=5, help='视频帧率')
    parser.add_argument('--num_workers', type=int, default=4, help='并行进程数')
    parser.add_argument('--subplot_size', type=int, default=400, help='每个子图的内容尺寸（像素）')
    parser.add_argument('--eval', action='store_true', help='是否运行评估')
    args = parser.parse_args()

    global nusc, DATAROOT, VERSION
    DATAROOT = args.dataroot
    VERSION = args.version
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # 加载预测结果
    with open(args.result_json, 'r') as f:
        pred_data = json.load(f)

    # 处理每个场景
    for scene in tqdm(nusc.scene, desc="Processing scenes"):
        scene_token = scene['token']
        img_arrays = process_scene_parallel(scene_token, pred_data, args.out_dir,
                                            args.subplot_size, args.num_workers)
        if img_arrays:
            video_path = os.path.join(args.out_dir, f"{scene['name']}_video.mp4")
            write_video_from_arrays(img_arrays, video_path, args.fps)

    # 评估
    if args.eval:
        eval_set = 'mini_val' if args.version == 'v1.0-mini' else 'val'
        compute_metrics(nusc, pred_data, args.out_dir, eval_set=eval_set)


if __name__ == '__main__':
    main()