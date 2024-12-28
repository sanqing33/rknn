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

#include "yolov5face.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include "easy_timer.h"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];

    int ret;
    TIMER time;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_yolov5face_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov5face_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    time.tik();
    ret = read_image(image_path, &src_image);
    time.tok();
    time.print_time("read_image");
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        goto out;
    }

    yolov5face_result_list od_results;
    time.tik();
    ret = inference_yolov5face_model(&rknn_app_ctx, &src_image, &od_results);
    time.tok();
    time.print_time("inference and postprocess yolov5face model");
    if (ret != 0)
    {
        printf("inference_yolov5face_model fail! ret=%d\n", ret);
        goto out;
    }

    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        printf("face @ (%d %d %d %d) %.3f\n",det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

        sprintf(text, "%.1f%%", det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
        for(int j = 0; j < 5; j++) {
            draw_circle(&src_image, det_result->ponit[j].x, det_result->ponit[j].y, 1, COLOR_ORANGE, 2);
        }
    }

    write_image("result.png", &src_image);

out:
    ret = release_yolov5face_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov5face_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }

    return 0;
}
