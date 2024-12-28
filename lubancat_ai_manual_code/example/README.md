# 目录说明

| 目录                | 对应教程章节                       |
| ------------------ | --------------------------------- |
| picodet            | 飞桨（PaddlePaddle） FastDeploy    | 
| ppocrv3            | PP-ORCv3                        |
| ppocrv4            | PP-ORCv4                        | 
| ppseg              | PP-LitetSeg                     |
| ppyoloe            | PP-YOLOE                        |
| RT-DETR            | RT-DETR                         |
| yolox              | YoloX                           |
| yolov5             | YOLOv5(目标检测)                 |
| yolov5_seg         | YOLOv5(实例分割)                 |
| yolov5face         | YOLOv5Face(人脸检测)             |
| yolov8             | YOLOv8                 |
| yolov8-obb         | YOLOv8旋转目标检测                |
| yolov10            | YOLOv10                     |
| yolo11             | YOLOv11                        |
| garbage_detection  | 垃圾检测和识别               |


# 文件说明

scaling_frequency.sh 是系统CPU，DDR，NPU频率修改脚本，例如：

```sh
# USAGE: ./fixed_frequency.sh -c {chip_name} [-h]"
# "  -c:  chip_name, such as rv1126 / rk3588"
# "  -h:  Help"
sudo bash scaling_frequency.sh -c rk3568
```

# 问题反馈

如果有任何问题请联系淘宝野火官方旗舰店技术支持反馈。