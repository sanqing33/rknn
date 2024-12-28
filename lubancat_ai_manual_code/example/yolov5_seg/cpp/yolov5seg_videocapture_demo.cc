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
#include <ctype.h>

#include "yolov5_seg.h"
#include <opencv2/opencv.hpp>

/*-------------------------------------------
                  Functions
-------------------------------------------*/

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model path> <camera device id/video path>\n", argv[0]);
        printf("Usage: %s  yolov5s_seg.rknn  0 \n", argv[0]);
        printf("Usage: %s  yolov5s_seg.rknn /path/xxxx.mp4\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *device_name = argv[2];

    unsigned char class_colors[][3] = {
        {255, 56, 56},   // 'FF3838'
        {255, 157, 151}, // 'FF9D97'
        {255, 112, 31},  // 'FF701F'
        {255, 178, 29},  // 'FFB21D'
        {207, 210, 49},  // 'CFD231'
        {72, 249, 10},   // '48F90A'
        {146, 204, 23},  // '92CC17'
        {61, 219, 134},  // '3DDB86'
        {26, 147, 52},   // '1A9334'
        {0, 212, 187},   // '00D4BB'
        {44, 153, 168},  // '2C99A8'
        {0, 194, 255},   // '00C2FF'
        {52, 69, 147},   // '344593'
        {100, 115, 255}, // '6473FF'
        {0, 24, 236},    // '0018EC'
        {132, 56, 255},  // '8438FF'
        {82, 0, 133},    // '520085'
        {203, 56, 255},  // 'CB38FF'
        {255, 149, 200}, // 'FF95C8'
        {255, 55, 199}   // 'FF37C7'
    };

    int ret;
    cv::Mat image, frame;
    struct timeval start_time, stop_time;
    rknn_app_context_t rknn_app_ctx;
    image_buffer_t src_image;
    object_detect_result_list od_results;

    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    memset(&src_image, 0, sizeof(image_buffer_t));

    cv::VideoCapture cap;
    if (strlen(device_name)==1 && isdigit(device_name[0])) {
        // 打开摄像头
        int camera_id = atoi(argv[2]);
        cap.open(camera_id);
        if (!cap.isOpened()) {
            printf("Error: Could not open camera.\n");
            return -1;
        }
        // cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);//宽度
        // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);//高度
    } else {
        // 打开视频文件
        cap.open(argv[2]);
        if (!cap.isOpened()) {  
            printf("Error: Could not open video file.\n");
            return -1;
        }
    }

    // 初始化
    init_post_process();
#ifndef ENABLE_ZERO_COPY
    ret = init_yolov5_seg_model(model_path, &rknn_app_ctx);
#else
    ret = init_yolov5seg_zero_copy_model(model_path, &rknn_app_ctx);
#endif
    if (ret != 0)
    {
        printf("init yolov5seg_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

	while(true) {
        gettimeofday(&start_time, NULL);

		// cap >> frame;
        if (!cap.read(frame)) {  
            printf("cap read frame fail!\n");
            break;  
        }  

        cv::cvtColor(frame, image, cv::COLOR_BGR2RGB);
        src_image.width  = image.cols;
        src_image.height = image.rows;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.virt_addr = (unsigned char*)image.data;

#ifndef ENABLE_ZERO_COPY
        ret = inference_yolov5_seg_model(&rknn_app_ctx, &src_image, &od_results);
#else
        ret = inference_yolov5seg_zero_copy_model(&rknn_app_ctx, &src_image, &od_results);
#endif
        if (ret != 0)
        {
            printf("inference yolov5seg_model fail! ret=%d\n", ret);
            goto out;
        }

        // draw mask
        if (od_results.count >= 1)
        {
            int width = frame.cols;
            int height = frame.rows;
            char *img_data = (char *)frame.data;
            int cls_id = od_results.results[0].cls_id;
            uint8_t *seg_mask = od_results.results_seg[0].seg_mask;

            float alpha = 0.5f; // opacity
            for (int j = 0; j < height; j++)
            {
                for (int k = 0; k < width; k++)
                {
                    int pixel_offset = 3 * (j * width + k);
                    if (seg_mask[j * width + k] != 0)
                    {
                        img_data[pixel_offset + 2] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % 20][0] * (1 - alpha) + img_data[pixel_offset + 2] * alpha, 0, 255); // r
                        img_data[pixel_offset + 1] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % 20][1] * (1 - alpha) + img_data[pixel_offset + 1] * alpha, 0, 255); // g
                        img_data[pixel_offset + 0] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % 20][2] * (1 - alpha) + img_data[pixel_offset + 0] * alpha, 0, 255); // b
                    }
                }
            }
            free(seg_mask);
        }

        // draw boxes
        char text[256];
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                det_result->box.left, det_result->box.top,
                det_result->box.right, det_result->box.bottom,
                det_result->prop);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 2);
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            putText(frame, text, cv::Point(x1, y1 - 6), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0,0,255), 1, cv::LINE_AA);
        }

		// 计算FPS
        gettimeofday(&stop_time, NULL);
        float t = (__get_us(stop_time) - __get_us(start_time))/1000;
        printf("Infer time(ms): %f ms\n", t);
		putText(frame, cv::format("FPS: %.2f", 1.0 / (t / 1000)), cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
		cv::imshow("YOLOv5_Seg C++ Demo", frame);

		char c = cv::waitKey(1);
		if (c == 27) { // ESC
			break;
		}

    }

out:
    deinit_post_process();

#ifndef ENABLE_ZERO_COPY
    ret = release_yolov5_seg_model(&rknn_app_ctx);
#else
    ret = release_yolov5seg_zero_copy_model(&rknn_app_ctx);
#endif
    if (ret != 0)
    {
        printf("release_yolov5_seg_model fail! ret=%d\n", ret);
    }

    return 0;
}
