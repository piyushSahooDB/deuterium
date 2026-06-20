#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose, Point, Quaternion
from cv_bridge import CvBridge
import cv2

from zed_msgs.msg import ObjectsStamped
from auv_msgs.msg import Detection, DetectionArray

# ── YOLO CLASS MAPPING ────────────────────────────────────────────────
YOLO_CLASS_MAP = {
    0: 'preq_gate',
    1: 'preq_pole'
}

# ── BOX COLORS PER CLASS (BGR) ──────────────────────────────────────────
CLASS_COLORS = {
    'preq_gate': (255, 0, 0),
    'preq_pole': (0, 255, 0),
}


class UnifiedDetectionNode(Node):
    def __init__(self):
        super().__init__('unified_detection_node')

        self.cv2_bridge  = CvBridge()
        self.imu_pose    = None
        self.zed_objects = []

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers
        self.detect_img_pub  = self.create_publisher(Image,            '/zed2i_front/detection_image', qos)
        self.detection_pub   = self.create_publisher(DetectionArray,   '/zed2i_front/detection_msg',   qos)
        self.detection3d_pub = self.create_publisher(Detection3DArray, '/detections_3d',               qos)

        # IMU
        self.imu_sub = self.create_subscription(
            Imu, '/zed2i_front/zed_node/imu/data', self.imu_callback, qos)

        # ZED SDK objects
        self.objects_sub = self.create_subscription(
            ObjectsStamped, '/zed2i_front/zed_node/obj_det/objects',
            self.objects_callback, qos)

        # RGB image (for drawing/publishing annotated frame)
        self.image_sub = self.create_subscription(
            Image, '/zed2i_front/zed_node/rgb/color/rect/image',
            self.image_callback, qos)

        self.get_logger().info('UnifiedDetectionNode started (YOLO only).')

    def imu_callback(self, msg):
        self.imu_pose = msg.orientation

    def objects_callback(self, msg):
        # NOTE: Safe under single-threaded rclpy.spin(). If you switch to
        # MultiThreadedExecutor, wrap this and the list() copy below with a threading.Lock.
        self.zed_objects = msg.objects

    def build_custom_det(self, header, bbox, class_id, confidence, pos, tracking_id):
        x1, y1, x2, y2 = bbox
        det = Detection()
        det.header       = header
        det.tracking_id  = tracking_id
        det.class_id     = class_id
        det.confidence   = confidence

        det.bbox_x      = float((x1 + x2) / 2)
        det.bbox_y      = float((y1 + y2) / 2)
        det.bbox_width  = float(x2 - x1)
        det.bbox_height = float(y2 - y1)

        det.position.x = pos[0]
        det.position.y = pos[1]
        det.position.z = pos[2]

        if self.imu_pose is not None:
            det.orientation = self.imu_pose
        return det

    def build_det3d(self, header, bbox, class_id, confidence, pos):
        x1, y1, x2, y2 = bbox
        det3d  = Detection3D()
        det3d.header = header

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = class_id
        hyp.hypothesis.score    = confidence

        pose = Pose()
        pose.position    = Point(x=pos[0], y=pos[1], z=pos[2])
        pose.orientation = Quaternion(w=1.0)
        hyp.pose.pose    = pose

        det3d.results.append(hyp)
        det3d.bbox.center  = pose
        det3d.bbox.size.x  = float(x2 - x1)
        det3d.bbox.size.y  = float(y2 - y1)
        det3d.bbox.size.z  = 0.1
        return det3d

    def image_callback(self, img_msg):
        frame = self.cv2_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8').copy()

        det_array  = DetectionArray()
        det_array.header  = img_msg.header
        det3d_array = Detection3DArray()
        det3d_array.header = img_msg.header

        current_zed_objects = list(self.zed_objects)

        # ── FIRST PASS: keep only the highest-confidence detection per class ──
        # best_per_class[cls_name] = (conf, bbox, pos, zed_tracking_id)
        best_per_class = {}
        for obj in current_zed_objects:
            if not obj.bounding_box_2d.corners:
                continue

            corners  = obj.bounding_box_2d.corners
            x_coords = [c.kp[0] for c in corners]
            y_coords = [c.kp[1] for c in corners]
            x1, x2   = int(min(x_coords)), int(max(x_coords))
            y1, y2   = int(min(y_coords)), int(max(y_coords))
            bbox     = (x1, y1, x2, y2)

            cls_id   = obj.label_id
            cls_name = YOLO_CLASS_MAP.get(cls_id)
            if cls_name is None:
                self.get_logger().warn(
                    f'Unknown label_id {cls_id} from ZED SDK — skipping object',
                    throttle_duration_sec=5.0)
                continue

            # NOTE: obj.confidence from ZED SDK is the threshold-filtered value (0–100),
            # not necessarily a per-object YOLO score. Verify with:
            #   ros2 topic echo /zed2i_front/zed_node/obj_det/objects --field objects[0].confidence
            # If it always returns a fixed value, it is the configured threshold, not a score.
            conf = float(obj.confidence) / 100.0
            pos  = (abs(float(obj.position[0])),
                    float(obj.position[1]),
                    float(obj.position[2]))
            zed_tracking_id = int(obj.id) if hasattr(obj, 'id') else 0

            if cls_name not in best_per_class or conf > best_per_class[cls_name][0]:
                best_per_class[cls_name] = (conf, bbox, pos, zed_tracking_id)

        # ── SECOND PASS: draw, build messages, publish ────────────────
        for cls_name, (conf, bbox, pos, zed_tracking_id) in best_per_class.items():
            x1, y1, x2, y2 = bbox
            distance = pos[0]  # forward axis in RIGHT_HANDED_Z_UP_X_FORWARD

            det_array.detections.append(
                self.build_custom_det(img_msg.header, bbox, cls_name,
                                      conf, pos, zed_tracking_id))
            det3d_array.detections.append(
                self.build_det3d(img_msg.header, bbox, cls_name, conf, pos))

            color = CLASS_COLORS.get(cls_name, (255, 0, 0))
            label = f'{cls_name} {conf:.2f} | {distance:.2f}m [id={zed_tracking_id}]'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            self.get_logger().info(
                f'[YOLO] {cls_name} id={zed_tracking_id} at {distance:.2f}m (conf: {conf:.2f})',
                throttle_duration_sec=2.0)

        if not det_array.detections:
            self.get_logger().info('No detections published this frame')

        # ── PUBLISH ───────────────────────────────────────────────────
        try:
            annotated_msg        = self.cv2_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            annotated_msg.header = img_msg.header
            self.detect_img_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish image: {e}')

        self.detection_pub.publish(det_array)
        self.detection3d_pub.publish(det3d_array)


def main(args=None):
    rclpy.init(args=args)
    node = UnifiedDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()