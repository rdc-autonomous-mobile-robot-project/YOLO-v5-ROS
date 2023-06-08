import launch
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    yolov5_ros_share_dir = get_package_share_directory('yolov5_ros')

    weights = LaunchConfiguration('weights', default=f'{yolov5_ros_share_dir}/src/yolov5/config/yolov5s.pt')
    data = LaunchConfiguration('data', default='/home/fmasa/ros2_ws/src/YOLOv5-ROS/yolov5_ros/yolov5_ros/data/coco128.yaml')
    # data = LaunchConfiguration('data', default=f'{yolov5_ros_share_dir}/src/yolov5/data/coco128.yaml')
    confidence_threshold = LaunchConfiguration('confidence_threshold', default='0.50')
    iou_threshold = LaunchConfiguration('iou_threshold', default='0.50')
    maximum_detections = LaunchConfiguration('maximum_detections', default='1000')
    device = LaunchConfiguration('device', default='0')
    agnostic_nms = LaunchConfiguration('agnostic_nms', default='true')
    line_thickness = LaunchConfiguration('line_thickness', default='3')
    dnn = LaunchConfiguration('dnn', default='true')
    half = LaunchConfiguration('half', default='false')
    inference_size_h = LaunchConfiguration('inference_size_h', default='640')
    inference_size_w = LaunchConfiguration('inference_size_w', default='640')
    view_image = LaunchConfiguration('view_image', default='true')
    input_image_topic = LaunchConfiguration('input_image_topic', default='/camera_under/rgb/image_raw')
    output_topic = LaunchConfiguration('output_topic', default='/yolov5/detections')
    publish_image = LaunchConfiguration('publish_image', default='false')
    output_image_topic = LaunchConfiguration('output_image_topic', default='/yolov5/image_out')

    declare_weights = DeclareLaunchArgument(
        'weights',
        default_value=weights,
        description='Path to the YOLOv5 model weights file'
    )

    declare_data = DeclareLaunchArgument(
        'data',
        default_value=data,
        description='Path to the YOLOv5 data configuration file'
    )

    declare_confidence_threshold = DeclareLaunchArgument(
        'confidence_threshold',
        default_value=confidence_threshold,
        description='Confidence threshold for object detection'
    )

    declare_iou_threshold = DeclareLaunchArgument(
        'iou_threshold',
        default_value=iou_threshold,
        description='IoU threshold for object detection'
    )

    declare_maximum_detections = DeclareLaunchArgument(
        'maximum_detections',
        default_value=maximum_detections,
        description='Maximum number of detections'
    )

    declare_device = DeclareLaunchArgument(
        'device',
        default_value=device,
        description='CUDA device index to use for inference'
    )

    declare_agnostic_nms = DeclareLaunchArgument(
        'agnostic_nms',
        default_value=agnostic_nms,
        description='Whether to use agnostic NMS'
    )

    declare_line_thickness = DeclareLaunchArgument(
        'line_thickness',
        default_value=line_thickness,
        description='Line thickness for bounding box visualization'
    )

    declare_dnn = DeclareLaunchArgument(
        'dnn',
        default_value=dnn,
        description='Whether to use DNN for image preprocessing'
    )

    declare_half = DeclareLaunchArgument(
        'half',
        default_value=half,
        description='Whether to use half-precision (FP16) inference'
    )

    declare_inference_size_h = DeclareLaunchArgument(
        'inference_size_h',
        default_value=inference_size_h,
        description='Inference size (height)'
    )

    declare_inference_size_w = DeclareLaunchArgument(
        'inference_size_w',
        default_value=inference_size_w,
        description='Inference size (width)'
    )

    declare_view_image = DeclareLaunchArgument(
        'view_image',
        default_value=view_image,
        description='Whether to visualize using OpenCV window'
    )

    declare_input_image_topic = DeclareLaunchArgument(
        'input_image_topic',
        default_value=input_image_topic,
        description='Input image topic'
    )

    declare_output_topic = DeclareLaunchArgument(
        'output_topic',
        default_value=output_topic,
        description='Output topic for detections'
    )

    declare_publish_image = DeclareLaunchArgument(
        'publish_image',
        default_value=publish_image,
        description='Whether to publish annotated image'
    )

    declare_output_image_topic = DeclareLaunchArgument(
        'output_image_topic',
        default_value=output_image_topic,
        description='Output topic for annotated image'
    )

    detect_node = launch_ros.actions.Node(
        package='yolov5_ros', executable='yolov5_ros_detect',
        name='detect',
        output='screen',
        parameters=[
            {'weights': weights},
            {'data': data},
            {'confidence_threshold': confidence_threshold},
            {'iou_threshold': iou_threshold},
            {'maximum_detections': maximum_detections},
            {'device': device},
            {'agnostic_nms': agnostic_nms},
            {'line_thickness': line_thickness},
            {'dnn': dnn},
            {'half': half},
            {'inference_size_h': inference_size_h},
            {'inference_size_w': inference_size_w},
            {'input_image_topic': input_image_topic},
            {'output_topic': output_topic},
            {'view_image': view_image},
            {'publish_image': publish_image},
            {'output_image_topic': output_image_topic}
        ]
    )

    return launch.LaunchDescription([
        declare_weights,
        declare_data,
        declare_confidence_threshold,
        declare_iou_threshold,
        declare_maximum_detections,
        declare_device,
        declare_agnostic_nms,
        declare_line_thickness,
        declare_dnn,
        declare_half,
        declare_inference_size_h,
        declare_inference_size_w,
        declare_view_image,
        declare_input_image_topic,
        declare_output_topic,
        declare_publish_image,
        declare_output_image_topic,
        detect_node
    ])

