// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "yolov8_pose.h"
#include "file_utils.h"
#include "image_utils.h"

#include <opencv2/opencv.hpp>

static const unsigned char box_colors[19][3] = {
    {54, 67, 244},
    {99, 30, 233},
    {176, 39, 156},
    {183, 58, 103},
    {181, 81, 63},
    {243, 150, 33},
    {244, 169, 3},
    {212, 188, 0},
    {136, 150, 0},
    {80, 175, 76},
    {74, 195, 139},
    {57, 220, 205},
    {59, 235, 255},
    {7, 193, 255},
    {0, 152, 255},
    {34, 87, 255},
    {72, 85, 121},
    {158, 158, 158},
    {139, 125, 96}
};

static void draw_pose(cv::Mat& image, const object_detect_result *det_result){

    std::vector<cv::Scalar> pose_palette = {
        {255, 128,   0}, {255, 153,  51}, {255, 178, 102}, {230, 230,   0}, {255, 153, 255},
        {153, 204, 255}, {255, 102, 255}, {255, 51,  255}, {102, 178, 255}, {51,  153, 255},
        {255, 153, 153}, {255, 102, 102}, {255, 51,   51}, {153, 255, 153}, {102, 255, 102},
        {51,  255,  51}, {0,   255,   0}, {0,   0,   255}, {255, 0,     0}, {255, 255, 255}
    };

    std::vector<cv::Point> skeleton = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
        {5,   6}, {5,   7}, {6,   8}, {7,   9}, {8,  10}, {1,  2}, {0,  1}, 
        {0,   2}, {1,   3}, {2,   4}, {3,   5}, {4,   6}
    };

    std::vector<cv::Scalar> limb_color = {
        pose_palette[9],  pose_palette[9],  pose_palette[9],  pose_palette[9],  pose_palette[7],
        pose_palette[7],  pose_palette[7],  pose_palette[0],  pose_palette[0],  pose_palette[0],
        pose_palette[0],  pose_palette[0],  pose_palette[16], pose_palette[16], pose_palette[16],
        pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16]
    };

    std::vector<cv::Scalar> kpt_color = {
        pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16],
        pose_palette[0],  pose_palette[0],  pose_palette[0],  pose_palette[0],  pose_palette[0],
        pose_palette[0],  pose_palette[9],  pose_palette[9],  pose_palette[9],  pose_palette[9],
        pose_palette[9],  pose_palette[9]
    };

    for(int i = 0; i < 17; ++i){
        if(det_result->keypoints[i][2] < 0.5)
            continue;
        if(det_result->keypoints[i][0] != 0 && det_result->keypoints[i][1] != 0)
            cv::circle(image, cv::Point(det_result->keypoints[i][0], det_result->keypoints[i][1]), 2, kpt_color[i], -1, cv::LINE_AA);
    }

    for(int i = 0; i < skeleton.size(); ++i){
        auto& index = skeleton[i];
        int x = index.x;
        int y = index.y;

        if(det_result->keypoints[x][2] < 0.5 || det_result->keypoints[y][2]< 0.5)
            continue;
        if(det_result->keypoints[x][0] == 0 || det_result->keypoints[x][1] == 0 || det_result->keypoints[y][0] == 0 || det_result->keypoints[y][1] == 0)
            continue;
        cv::line(image, cv::Point(det_result->keypoints[x][0], det_result->keypoints[x][1]),
            cv::Point(det_result->keypoints[y][0], det_result->keypoints[y][1]), limb_color[i], 1, cv::LINE_AA);
    }
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model path> <camera device id/video path>\n", argv[0]);
        printf("Usage: %s  yolov8_pose.rknn  0 \n", argv[0]);
        printf("Usage: %s  yolov8_pose.rknn /path/xxxx.mp4\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *device_name = argv[2];

    int ret;
    cv::Mat frame, image;
    struct timeval start_time, stop_time;
    rknn_app_context_t rknn_app_ctx;
    image_buffer_t src_image;

    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    memset(&src_image, 0, sizeof(image_buffer_t));

    cv::VideoCapture cap;
    if (isdigit(device_name[0])) {
        // 摄像头
        int camera_id = atoi(argv[2]);
        cap.open(camera_id);

        if (!cap.isOpened()) {
            printf("Error: Could not open camera.\n");
            return -1;
        }
        // cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    } else {
        // 视频文件或者其他
        cap.open(argv[2]);
        if (!cap.isOpened()) {  
            printf("Error: Could not open video file.\n");
            return -1;
        }
    }

    init_post_process();
    ret = init_yolov8_pose_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov10_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

	while(true) {
        if (!cap.read(frame)) {  
            printf("cap read frame fail!\n");
            break;
        }

        cv::cvtColor(frame, image, cv::COLOR_BGR2RGB);
        // image.convertTo(image, CV_8UC3);
        src_image.width  = image.cols;
        src_image.height = image.rows;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.virt_addr = (unsigned char*)image.data;

        // rknn inference and postprocess
        object_detect_result_list od_results;
        ret = inference_yolov8_pose_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0)
        {
            printf("init_yolov10_model fail! ret=%d\n", ret);
            goto out;
        }

        int color_index = 0;
        char text[256];
        for (int i = 0; i < od_results.count; i++)
        {
            const unsigned char* color = box_colors[color_index % 19];
            cv::Scalar cc(color[0], color[1], color[2]);
            color_index++;

            object_detect_result *det_result = &(od_results.results[i]);
            printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                det_result->box.left, det_result->box.top,
                det_result->box.right, det_result->box.bottom,
                det_result->prop);
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);

            cv::rectangle(frame, cv::Rect(cv::Point(det_result->box.left, det_result->box.top), 
                            cv::Point(det_result->box.right, det_result->box.bottom)), cc, 2);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = det_result->box.left;
            int y = det_result->box.top - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > frame.cols)
                x = frame.cols - label_size.width;

            cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),cc,-1);
            cv::putText(frame, text, cv::Point(x, y + label_size.height),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));

            draw_pose(frame, det_result);

        }

        // 显示结果
        cv::imshow("yolov8_pose_videocapture_demo", frame);

        char c = cv::waitKey(1);
		if (c == 27) { // ESC
			break;
		}
    }

out:
    deinit_post_process();
    ret = release_yolov8_pose_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov10_model fail! ret=%d\n", ret);
    }
    return 0;
}
