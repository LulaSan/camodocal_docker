import cv2
import numpy as np
import yaml
import os
import argparse

# Define paths
default_image_path = os.getenv('IMAGE_PATH', "/path/to/images/")
default_output_path = os.getenv('OUTPUT_PATH', "/path/to/output/")
default_yaml_paths = {
    'pinhole': os.getenv('PINHOLE_YAML', "/path/to/pinhole_calib.yaml"),
    'kannala': os.getenv('KANNALA_YAML', "/path/to/kannala_calib.yaml"),
    'kannala_fisheye': os.getenv('FISHEYE_YAML', "/path/to/fisheye_calib.yaml")
}

def load_camera_params(yaml_file_path):
    """Load camera parameters from a YAML file."""
    with open(yaml_file_path) as file:
        lines = file.readlines()
    if lines[0].startswith('%YAML'):
        lines = lines[1:]
    return yaml.safe_load(''.join(lines))

def undistort_pinhole(image_path, output_path, camera_params):
    """Undistort images using Pinhole camera model."""
    fx, fy = camera_params['projection_parameters']['fx'], camera_params['projection_parameters']['fy']
    cx, cy = camera_params['projection_parameters']['cx'], camera_params['projection_parameters']['cy']
    dist = np.array([camera_params['distortion_parameters'][k] for k in ['k1', 'k2', 'p1', 'p2']])

    for filename in os.listdir(image_path):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(image_path, filename))
            h, w = img.shape[:2]
            mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            rectified = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
            x, y, w, h = roi
            rectified = rectified[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_path, f"{filename}_pinhole.png"), rectified)

def undistort_kannala(image_path, output_path, camera_params):
    """Undistort images using Kannala-Brandt model."""
    mu, mv = camera_params['projection_parameters']['mu'], camera_params['projection_parameters']['mv']
    u0, v0 = camera_params['distortion_parameters']['u0'], camera_params['distortion_parameters']['v0']
    D = np.array([camera_params['distortion_parameters'][k] for k in ['k2', 'k3', 'k4', 'k5']])

    K = np.array([[mu, 0, u0], [0, mv, v0], [0, 0, 1]])
    for filename in os.listdir(image_path):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(image_path, filename))
            h, w = img.shape[:2]
            new_camera_mtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=1)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_camera_mtx, (w, h), cv2.CV_16SC2)
            rectified_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(os.path.join(output_path, f"{filename}_kannala.png"), rectified_img)

def main(args):
    # Select calibration model and process images accordingly
    if args.model == 'pinhole':
        params = load_camera_params(args.yaml_path)
        undistort_pinhole(args.image_path, args.output_path, params)
    elif args.model == 'kannala':
        params = load_camera_params(args.yaml_path)
        undistort_kannala(args.image_path, args.output_path, params)
    else:
        print(f"Model '{args.model}' not supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera model undistortion")
    parser.add_argument("--model", choices=['pinhole', 'kannala'], required=True, help="Choose the calibration model.")
    parser.add_argument("--image_path", default=default_image_path, help="Path to the directory with input images.")
    parser.add_argument("--output_path", default=default_output_path, help="Path to save undistorted images.")
    parser.add_argument("--yaml_path", default=default_output_path, help="Path to calibration yaml file.")
    
    args = parser.parse_args()
    main(args)
