
import logging
import numpy as np
import numpy.ma as ma
from typing import List, Optional, Tuple, Dict
import cv2
import torch
import matplotlib.pyplot as plt

from densepose.vis.densepose_results import DensePoseResultsVisualizer
from densepose.vis.base import Boxes, Image


class DensePoseResultsUVIntersectionVisualizer(DensePoseResultsVisualizer):
    """
    Visualize the intersection of U and V contour line
    """
    def __init__(self, **kwargs):
        self.plot_args = kwargs

    def create_visualization_context(self, image_bgr: Image):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasAgg as FigureCanvas

        context = {}
        context["image_bgr"] = image_bgr
        dpi = 80
        height_inches = float(image_bgr.shape[0]) / dpi
        width_inches = float(image_bgr.shape[1]) / dpi
        fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
        plt.axes([0, 0, 1, 1])
        plt.axis("off")
        context["fig"] = fig
        canvas = FigureCanvas(fig)
        context["canvas"] = canvas
        extent = (0, image_bgr.shape[1], image_bgr.shape[0], 0)
        plt.imshow(image_bgr[:, :, ::-1], extent=extent)
        return context

    def context_to_image_bgr(self, context):
        fig = context["fig"]
        w, h = map(int, fig.get_size_inches() * fig.get_dpi())
        canvas = context["canvas"]
        canvas.draw()
        image_1d = np.fromstring(canvas.tostring_rgb(), dtype="uint8")
        image_rgb = image_1d.reshape(h, w, 3)
        image_bgr = image_rgb[:, :, ::-1].copy()
        return image_bgr

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh: Boxes) -> None:


        u = self._extract_u_from_iuvarr(iuv_arr).astype(float) / 255.0
        v = self._extract_v_from_iuvarr(iuv_arr).astype(float) / 255.0
        I = iuv_arr[0, :, :]

        num_pts_w, num_pts_h = (3, 2)  # Number of points on Width and Height
        pos_arr = self.__gen_percentage_pos(num_pts_w, num_pts_h)
        inter_dict = self.__find_intersec(u, v, I, bbox_xywh, pos_arr)
        part_indices = list(range(1, 25))
        [part_indices.remove(p) for p in [3,4,5,6]]  # Remove hands and feets
        for part_i in part_indices:
            # Connect points in same part
            # Indices of position which should be connected.
            pos_conn_idx = self.__connections_inner(num_pts_w, num_pts_h)
            for cid in pos_conn_idx:
                self.__line(inter_dict[(part_i, cid[0])], inter_dict[(part_i, cid[1])])
            # Mark a number for each point
            for n in range(num_pts_w * num_pts_h):
                self.__number(n, inter_dict[(part_i, n)])


    def __gen_percentage_pos(self, num_w, num_h):
        """
        Generate percentage positions like: ((0.3, 0.3),(0.3, 0.5)...)
        to be found on UV texture.
        """
        x = np.linspace(0., 1., num_w + 2)[1: -1]  # Drop 0.0 and 1.0
        y = np.linspace(0., 1., num_h + 2)[1: -1]
        xv, yv = np.meshgrid(x, y)  # (H, W), (H, W)
        xy = np.stack((xv, yv), axis=0)  # shape: (2, H, W)
        xy = np.transpose(xy, (1,2,0))  # (H, W, 2)
        xy = np.reshape(xy, (-1, 2))  # (H*W, 2)
        return xy

    def __find_intersec(self, u: np.ndarray, v: np.ndarray, I: np.ndarray,
                        bbox_xywh: Boxes, pos_arr: np.ndarray):
        """
        Find intersections on each part
        Output: Dict of (part, position): (x,y,v)
        """
        inter_dict: Dict[Tuple[int, int], Tuple[int, int, int]] = {}  # (Part, idx): (x,y,vis)

        for pos_i, pos in enumerate(pos_arr):
            # pos.shape: (2), order:xy. For torso, v appears to be horizontal, u vertical; Opposite for head
            dist_arr = np.abs(v - pos[0]) + np.abs(u - pos[1])
            for part_i in range(1, 25):
                # 24 part + 1 background
                # Mask all parts except part i. "1" means invalid.
                part_mask = (I != part_i)
                dist_part_spec = ma.array(dist_arr, mask=part_mask)
                argmin = np.argmin(dist_part_spec)
                argmin = np.unravel_index(argmin, dist_arr.shape)  # H,W
                min_val = dist_part_spec[argmin]
                # if min_val > 0.1:
                if argmin == (0, 0):
                    vis = 0  # Not visible
                else:
                    vis = 1
                # x,y,v
                inter_dict[(part_i, pos_i)] = tuple(map(int, (argmin[1]+bbox_xywh[0], argmin[0]+bbox_xywh[1], vis)))
        return inter_dict

    def __line(self, xyv1, xyv2, **kwargs):
        # Plot two pts, connect if both visible
        if 'color' in kwargs:
            color = kwargs['color']
        else:
            color = 'red'
        if xyv1[2] != 1 and xyv2[2] != 1:  # Both invisible
            return
        elif xyv1[2] == 1 and xyv2[2] == 1:  # Both visible
            plt.plot([xyv1[0], xyv2[0]], [xyv1[1], xyv2[1]],
                     color=color, marker='o', linestyle='dashed')
        elif xyv1[2] == 1:  # p1 visible
            plt.plot(xyv1[0], xyv1[1], color=color, marker='o', linestyle='dashed')
        elif xyv2[2] == 0:
            plt.plot(xyv2[0], xyv2[1], color=color, marker='o', linestyle='dashed')

    def __number(self, num, xyv):
        # draw a number on point
        if xyv[2] == 0:  # invisible
            return
        plt.text(xyv[0], xyv[1], str(num), color='green', size='medium', bbox=dict(facecolor='black', alpha=0.7))

    def __connections_inner(self, num_w, num_h):
        # Point connections inside a part like: [(0,1), (2,3), (0,2), (1,3)]
        w, h = (num_w, num_h)
        horizontal = [(n, n+1) for n in range(w*h) if n % w != w-1]
        vertical = [(n, n+w) for n in range(w*h) if n < w*(h-1)]
        return horizontal + vertical

    def __connections_outer(self, num_w, num_h):
        # Connections between different parts
        # {(Part1, Part2): ((TB1, FR1), (TB2, FR2))}. TB: Top/Bottom. FR: Forward/Reverse

        conn_inner = {
            (1, 8):(('B', 'F'), ('T', 'R'))
        }

    @staticmethod
    def _extract_i_from_iuvarr(iuv_arr):
        return iuv_arr[0, :, :]

    @staticmethod
    def _extract_u_from_iuvarr(iuv_arr):
        return iuv_arr[1, :, :]

    @staticmethod
    def _extract_v_from_iuvarr(iuv_arr):
        return iuv_arr[2, :, :]