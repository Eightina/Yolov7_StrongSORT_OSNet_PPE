# rewrite two functions from yolov7.utils.general
# mainly because these two function in yolov7.utils.general only deals with [:, :4] vals in the input tensor
# and the format of pose model output is different from normal ([x, y, conf, ...] repeating)

# input coords format (inferencing yolov7.utils.plots.output_to_keypoint()):
# [batch_id, class_id, x, y, w, h, conf, x0, y0, conf0, x1, y1, conf1, ......, x16, y16, conf16]
# that is 7 + 17 * 3 = 57, 0~6 is bbox, 7~58 are keypoints

# takes tensor shape = (n, 51)
def kpts_scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xycxyc...) from img1_shape to img0_shape, ignore conf vals
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    coords[:, [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]] -= pad[0]  # x padding
    coords[:, [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49]] -= pad[1]  # y padding
    coords[:, [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 46, 48, 49]] /= gain    
    kpts_clip_coords(coords, img0_shape)    
    return coords


def kpts_clip_coords(coords, img_shape):
    # Clip all xy coords to image shape (height, width)
    coords[:, [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]].clamp_(0, img_shape[1])  # xi
    coords[:, [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49]].clamp_(0, img_shape[0])  # yi
    # boxes[:, 0].clamp_(0, img_shape[1])  # x1
    # boxes[:, 1].clamp_(0, img_shape[0])  # y1
    # boxes[:, 2].clamp_(0, img_shape[1])  # x2
    # boxes[:, 3].clamp_(0, img_shape[0])  # y2