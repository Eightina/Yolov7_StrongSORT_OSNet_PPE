import torch
from torch import Tensor

class Person:
    def __init__(self, id, bbox, skeleton_coords):
        self.id = id
        self.bbox = bbox #xyxy
        self.skeleton_coords = skeleton_coords
        self.head_bbox = None
        self.body_bbox = None
        self.with_helmet = False
        self.with_vest = False
        self.LIMIT = 0.2
        self.generate_head_bbox()
        self.generate_body_bbox()
        self.helmet_bbox = None
        self.vest_bbox = None
        self.helmet_iou = 0
        self.vest_iou = 0
        
        
    # try vectorizing here later
    def generate_head_bbox(self):
        # if self.skeleton_coords[11] > 0.25 and self.skeleton_coords[14] > 0.25:
        #     xle = self.skeleton_coords[9]
        #     yle = self.skeleton_coords[10]
        #     xre = self.skeleton_coords[12]
        #     yre = self.skeleton_coords[13]
        #     x1 =   
        if self.skeleton_coords[2] > 0 and self.skeleton_coords[17] > 0 and self.skeleton_coords[20] > 0:
            x0 = self.skeleton_coords[15] + (self.skeleton_coords[15] - self.skeleton_coords[18]) / 4.0
            ymid = (self.skeleton_coords[16] + self.skeleton_coords[19]) / 2.0
            y0 =  self.skeleton_coords[1] + 1.5 * (self.skeleton_coords[1] - ymid)
            x1 = self.skeleton_coords[18] - (self.skeleton_coords[18] - self.skeleton_coords[15]) / 4.0           
            y1 = self.skeleton_coords[1]
            self.head_bbox = Tensor([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]).to(device=torch.device("cuda"))

        
        
    def generate_body_bbox(self):
        if self.skeleton_coords[17] > 0 and self.skeleton_coords[38] > 0 \
            and self.skeleton_coords[20] > 0 and self.skeleton_coords[35] > 0:
                x0 = min(self.skeleton_coords[[15, 18, 33, 36]])
                y0 = min(self.skeleton_coords[[16, 19, 34, 37]])
                x1 = max(self.skeleton_coords[[15, 18, 33, 36]])
                y1 = max(self.skeleton_coords[[16, 19, 34, 37]])
                self.body_bbox = Tensor([x0, y0, x1, y1]).to(device=torch.device("cuda"))

        #     self.body_bbox = Tensor([self.skeleton_coords[15],\
        #                             self.skeleton_coords[16],\
        #                             self.skeleton_coords[36],\
        #                             self.skeleton_coords[37],\
        #                             ]).to(device=torch.device("cuda"))
        
        # elif self.skeleton_coords[20] > 0 and self.skeleton_coords[35] > 0:
        #     self.body_bbox = Tensor([self.skeleton_coords[18],\
        #                             self.skeleton_coords[19],\
        #                             self.skeleton_coords[33],\
        #                             self.skeleton_coords[34],\
        #                             ]).to(device=torch.device("cuda"))
    
    # names: ['helmet', 'vest', 'worker']
    def ppe_paring(self, ppe_clss, ppe_bbox):
        if self.head_bbox != None and ppe_clss == 0 and self.with_helmet == False:
            self.helmet_iou = self.compute_iou(self.head_bbox, ppe_bbox)
            if self.helmet_iou >= self.LIMIT:
                self.helmet_bbox = ppe_bbox
                self.with_helmet = True
        elif self.body_bbox != None and ppe_clss == 1 and self.with_vest == False:
            self.vest_iou = self.compute_iou(self.body_bbox, ppe_bbox) 
            if self.vest_iou >= self.LIMIT:
                self.vest_bbox = ppe_bbox
                self.with_vest = True
    
    def compute_iou(self, rec1, rec2):
        """
        computing IoU
        :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
        :param rec2: (y0, x0, y1, x1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        
        S_rec1 = abs((rec1[2] - rec1[0]) * (rec1[3] - rec1[1]))
        S_rec2 = abs((rec2[2] - rec2[0]) * (rec2[3] - rec2[1]))
        
        # computing the sum_area
        sum_area = S_rec1 + S_rec2
        
        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
        
        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect))*1.0
            