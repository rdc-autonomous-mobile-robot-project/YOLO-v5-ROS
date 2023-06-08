#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys
# from rostopic import get_topic_type

from sensor_msgs.msg import Image, CompressedImage
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox

from std_msgs.msg import Bool
from std_srvs.srv import SetBool
# from std_srvs.srv import SetBool, SetBool_Response, SetBool_Request
from rclpy.qos import QoSProfile

# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from yolov5_ros.models.common import DetectMultiBackend
from yolov5_ros.utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    scale_coords
)
from yolov5_ros.utils.plots import Annotator, colors
from yolov5_ros.utils.torch_utils import select_device
from yolov5_ros.utils.augmentations import letterbox


class Yolov5Detector(Node):
    def __init__(self):
        super().__init__("yolov5_detector")

        self.stop_pub = self.create_publisher(Bool, "/detect", QoSProfile(depth=1))
        self.stop = False
        self.restart_srv = self.create_service(SetBool, "/waypoint_manager/waypoint_server/resume_waypoint", self.callback_trigger)
        self.judge_srv = self.create_service(SetBool, '/switch_segmentation', self.callback_dl_switch)
        self.trigger = False
        self.switch = False
        self.count = 0
        self.image_store = 0
        self.camera_header = 0

        self.time_period = 0.2
        self.tmr = self.create_timer(self.time_period, self.callback_timer)

        # self.declare_parameter = ('confidence_threshold', value=0.50)
        self.declare_parameter('confidence_threshold', "0.50")
        self.declare_parameter('iou_threshold', "0.50")
        self.declare_parameter('agnostic_nms', "0.50")
        self.declare_parameter('maximum_detections', "0.50")
        self.declare_parameter('classes', "0.50")
        self.declare_parameter('line_thickness', "0.50")
        self.declare_parameter('view_image', "0.50")
        self.declare_parameter('weights', "0.50")
        self.declare_parameter('device', "0.50")
        self.declare_parameter('dnn', "0.50")
        self.declare_parameter('data', "0.50")
        self.declare_parameter('inference_size_w', "0.50")
        self.declare_parameter('inference_size_h', "0.50")
        self.declare_parameter('half', "false")
        self.declare_parameter('output_topic', "false")
        self.declare_parameter('output_image_topic', "false")
        self.declare_parameter('publish_image', "false")

        self.conf_thres = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        self.iou_thres = self.get_parameter("iou_threshold").get_parameter_value().double_value
        self.agnostic_nms = self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        self.max_det = self.get_parameter("maximum_detections").get_parameter_value().integer_value
        self.classes = self.get_parameter("classes").get_parameter_value().string_array_value
        self.line_thickness = self.get_parameter("line_thickness").get_parameter_value().integer_value
        self.view_image = self.get_parameter("view_image").get_parameter_value().bool_value
        # Initialize weights
        weights = self.get_parameter("weights").get_parameter_value().string_value
        # Initialize model
        self.device = select_device(str(self.get_parameter("device").get_parameter_value().string_value))
        self.model = DetectMultiBackend(weights, device=self.device, dnn=self.get_parameter("dnn").get_parameter_value().string_value, data=self.get_parameter("data").get_parameter_value().string_value)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )

        # Setting inference size
        self.img_size = [self.get_parameter("inference_size_w").get_parameter_value().integer_value, self.get_parameter("inference_size_h").get_parameter_value().integer_value]
        self.img_size = check_img_size(self.img_size, s=self.stride)

        # Half
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup()  # warmup

        # Initialize subscriber to Image/CompressedImage topic
        # input_image_type, input_image_topic, _ = get_topic_type(self.get_parameter("input_image_topic").get_parameter_value().string_value, blocking=True)
        # self.compressed_input = "sensor_msgs/msg/Image"
        self.compressed_input = False
        # self.compressed_input = "sensor_msgs/msg/CompressedImage"
        input_image_topic = "/camera/image_raw"

        if self.compressed_input:
            self.image_sub = self.create_subscription(
                CompressedImage,
                input_image_topic,
                self.callback,
                1,
            )
        else:
            self.image_sub = self.create_subscription(
                Image,
                input_image_topic,
                self.callback,
                1,
            )

        # Initialize prediction publisher
        self.pred_pub = self.create_publisher(
            BoundingBoxes,
            self.get_parameter("output_topic").get_parameter_value().string_value,
            QoSProfile(depth=10),
        )
        # Initialize image publisher
        self.publish_image = self.get_parameter("publish_image").get_parameter_value().bool_value
        if self.publish_image:
            self.image_pub = self.create_publisher(
                Image,
                self.get_parameter("output_image_topic").get_parameter_value().string_value,
                QoSProfile(depth=10),
            )

        # Initialize CV_Bridge
        self.bridge = CvBridge()

    def callback_trigger(self, request, response):
        self.trigger = request.data
        response.success = True
        return response

    def callback_dl_switch(self, request, response):
        self.switch = request.data
        response.message = "switch: " + str(self.switch)
        response.success = True
        return response

    def callback(self, data):
        """adapted from yolov5/detect.py"""
        if self.compressed_input:
            self.image_store = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        else:
            self.image_store = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

        self.camera_header = data

    def callback_timer(self):
        if not self.switch:
            return

        im, im0 = self.preprocess(self.image_store)

        # Run inference
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou_thres,
            self.classes,
            self.agnostic_nms,
            max_det=self.max_det,
        )

        ### To-do move pred to CPU and fill BoundingBox messages

        # Process predictions
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = self.camera_header.header
        bounding_boxes.image_header = self.camera_header.header

        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if self.switch:

            if self.trigger:
                self.stop = False
                msg = Bool()
                msg.data = self.stop
                self.stop_pub.publish(msg)
                self.trigger = False

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    bounding_box = BoundingBox()
                    c = int(cls)
                    # Fill in bounding box message
                    bounding_box.Class = self.names[c]
                    bounding_box.probability = conf
                    bounding_box.xmin = int(xyxy[0])
                    bounding_box.ymin = int(xyxy[1])
                    bounding_box.xmax = int(xyxy[2])
                    bounding_box.ymax = int(xyxy[3])

                    bounding_boxes.bounding_boxes.append(bounding_box)

                    if bounding_box.Class == "Line" and bounding_box.probability > 0.50 and self.count == 0:
                        self.count = 1
                        self.stop = True
                        msg = Bool()
                        msg.data = self.stop
                        self.stop_pub.publish(msg)

                    # Annotate the image
                    if self.publish_image or self.view_image:  # Add bbox to image
                        # integer class
                        label = f"{self.names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    ### POPULATE THE DETECTION MESSAGE HERE

                # Stream results
                im0 = annotator.result()

            # Publish prediction
            self.pred_pub.publish(bounding_boxes)

            # Publish & visualize images
            if self.view_image:
                cv2.imshow(str(0), im0)
                cv2.waitKey(1)  # 1 millisecond
            if self.publish_image:
                self.image_pub.publish(
                    self.bridge.cv2_to_imgmsg(im0, "bgr8")
                )

        else:
            self.count = 0

    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0


def main(args=None):
    rclpy.init(args=args)
    detector = Yolov5Detector()
    rclpy.spin(detector)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

