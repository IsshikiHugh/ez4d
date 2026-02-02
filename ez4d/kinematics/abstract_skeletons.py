"""
"Skeletons" here means virtual "bones" between joints.
It defines the topology of the skeleton and the visualization (color) symbols.
"""

class Skeleton():
    # Any topology related.s
    bones = []
    bone_colors = []
    # Tree topology related.
    root_idx = None
    chains = []
    parent = []


class Skeleton_COCO17(Skeleton):
    bones = [
            [ 0,  1], [ 0,  2], [ 1,  2],  # face triangle (nose & eyes)
            [ 1,  3], [ 3,  5],            # left head (ears to shoulders)
            [ 2,  4], [ 4,  6],            # right head (ears to shoulders)
            [ 5,  6],                      # neck (horizontal bone)
            [ 5,  7], [ 7,  9],            # left arm
            [ 6,  8], [ 8, 10],            # right arm
            [11, 12],                      # hip (horizontal bone)
            [ 5, 11], [11, 13], [13, 15],  # left leg
            [ 6, 12], [12, 14], [14, 16],  # right leg
        ]
    bone_colors = [
            [  0,   0, 127], [  0,   0, 127], [  0,   0, 127],   # blue
            [  0,   0, 127], [  0,   0, 127],                    # blue
            [  0,   0, 127], [  0,   0, 127],                    # blue
            [  0,   0, 127],                                     # blue
            [  0, 127, 127], [  0, 127, 127],                    # cyan
            [127,   0, 127], [127,   0, 127],                    # magenta
            [  0,   0, 127],                                     # blue
            [127,   0,   0], [127,   0,   0], [127,   0,   0],   # red
            [  0, 127,   0], [  0, 127,   0], [  0, 127,   0],   # green
        ]

    root_idx = None  # invalid for none-tree skeletons
    root_idx = None  # invalid for nontree skeletons
    parent   = None  # invalid for nontree skeletons


class Skeleton_SMPL24(Skeleton):
    """ Aligned with SMPL's official "parents". """
    root_idx = 0
    chains = [
            [ 0,  1,  4,  7, 10    ],  # left leg
            [ 0,  2,  5,  8, 11    ],  # right leg
            [ 0,  3,  6,  9, 12, 15],  # spine & head
            [ 9, 13, 16, 18, 20, 22],  # left arm
            [ 9, 14, 17, 19, 21, 23],  # right arm
        ]
    bones = [
            [ 0,  1], [ 1,  4], [ 4,  7], [ 7, 10],            # left leg
            [ 0,  2], [ 2,  5], [ 5,  8], [ 8, 11],            # right leg
            [ 0,  3], [ 3,  6], [ 6,  9], [ 9, 12], [12, 15],  # spine & head
            [ 9, 13], [13, 16], [16, 18], [18, 20], [20, 22],  # left arm
            [ 9, 14], [14, 17], [17, 19], [19, 21], [21, 23],  # right arm
        ]
    bone_colors = [
            [127,   0,   0], [148,  21,  21], [169,  41,  41], [191,  63,  63],                   # red
            [  0, 127,   0], [ 21, 148,  21], [ 41, 169,  41], [ 63, 191,  63],                   # green
            [  0,   0, 127], [ 15,  15, 143], [ 31,  31, 159], [ 47,  47, 175], [ 63,  63, 191],  # blue
            [  0, 127, 127], [ 15, 143, 143], [ 31, 159, 159], [ 47, 175, 175], [ 63, 191, 191],  # cyan
            [127,   0, 127], [143,  15, 143], [159,  31, 159], [175,  47, 175], [191,  63, 191],  # magenta
        ]
    parent = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


class Skeleton_SMPL22(Skeleton):
    """ Aligned with SMPL's official "parents". """
    root_idx = 0
    chains = [
            [ 0,  1,  4,  7, 10    ],  # left leg
            [ 0,  2,  5,  8, 11    ],  # right leg
            [ 0,  3,  6,  9, 12, 15],  # spine & head
            [ 9, 13, 16, 18, 20    ],  # left arm
            [ 9, 14, 17, 19, 21    ],  # right arm
        ]
    bones = [
            [ 0,  1], [ 1,  4], [ 4,  7], [ 7, 10],            # left leg
            [ 0,  2], [ 2,  5], [ 5,  8], [ 8, 11],            # right leg
            [ 0,  3], [ 3,  6], [ 6,  9], [ 9, 12], [12, 15],  # spine & head
            [ 9, 13], [13, 16], [16, 18], [18, 20],            # left arm
            [ 9, 14], [14, 17], [17, 19], [19, 21],            # right arm
        ]
    bone_colors = [
            [127,   0,   0], [148,  21,  21], [169,  41,  41], [191,  63,  63],                   # red
            [  0, 127,   0], [ 21, 148,  21], [ 41, 169,  41], [ 63, 191,  63],                   # green
            [  0,   0, 127], [ 15,  15, 143], [ 31,  31, 159], [ 47,  47, 175], [ 63,  63, 191],  # blue
            [  0, 127, 127], [ 15, 143, 143], [ 31, 159, 159], [ 47, 175, 175],                   # cyan
            [127,   0, 127], [143,  15, 143], [159,  31, 159], [175,  47, 175],                   # magenta
        ]
    parent = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


class Skeleton_SKEL24(Skeleton):
    # NOTE: Please only use this for visualization purposes.
    # FIXME: it's not consistent with the SKEL's official "parents".
    root_idx = 0
    chains = [
            [ 0,  6,  7,  8,  9, 10],  # left leg
            [ 0,  1,  2,  3,  4,  5],  # right leg
            [ 0, 11, 12, 13],          # spine & head
            [12, 19, 20, 21, 22, 23],  # left arm
            [12, 14, 15, 16, 17, 18],  # right arm
        ]
    bones = [
            [ 0,  6], [ 6,  7], [ 7,  8], [ 8,  9], [ 9, 10],  # left leg
            [ 0,  1], [ 1,  2], [ 2,  3], [ 3,  4], [ 4,  5],  # right leg
            [ 0, 11], [11, 12], [12, 13],                      # spine & head
            [12, 19], [19, 20], [20, 21], [21, 22], [22, 23],  # left arm
            [12, 14], [14, 15], [15, 16], [16, 17], [17, 18],  # right arm
        ]
    bone_colors = [
            [127,   0,   0], [148,  21,  21], [169,  41,  41], [191,  63,  63], [191,  63,  63],  # red
            [  0, 127,   0], [ 21, 148,  21], [ 41, 169,  41], [ 63, 191,  63], [ 63, 191,  63],  # green
            [  0,   0, 127], [ 31,  31, 159], [ 63,  63, 191],                                    # blue
            [  0, 127, 127], [ 15, 143, 143], [ 31, 159, 159], [ 47, 175, 175], [ 63, 191, 191],  # cyan
            [127,   0, 127], [143,  15, 143], [159,  31, 159], [175,  47, 175], [191,  63, 191],  # magenta
        ]
    parent = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 12, 19, 20, 21, 22, 12, 14, 15, 16, 17]


class Skeleton_OpenPose25(Skeleton):
    """ https://www.researchgate.net/figure/Twenty-five-keypoints-of-the-OpenPose-software-model_fig1_374116819 """
    root_idx = 8
    chain = [
            [ 8, 12, 13, 14, 19, 20],  # left leg
            [14, 21],                  # left heel
            [ 8,  9, 10, 11, 22, 23],  # right leg
            [11, 24],                  # right heel
            [ 8,  1,  0],              # spine & head
            [ 0, 16, 18],              # left face
            [ 0, 15, 17],              # right face
            [ 1,  5,  6,  7],          # left arm
            [ 1,  2,  3,  4],          # right arm
        ]
    bones = [
            [ 8, 12], [12, 13], [13, 14],  # left leg
            [14, 19], [19, 20], [14, 21],  # left foot
            [ 8,  9], [ 9, 10], [10, 11],  # right leg
            [11, 22], [22, 23], [11, 24],  # right foot
            [ 8,  1], [ 1,  0],            # spine & head
            [ 0, 16], [16, 18],            # left face
            [ 0, 15], [15, 17],            # right face
            [ 1,  5], [ 5,  6], [ 6,  7],  # left arm
            [ 1,  2], [ 2,  3], [ 3,  4],  # right arm
        ]
    bone_colors = [
            [ 95,   0, 255], [ 79,   0, 255], [ 83,   0, 255],  # dark blue
            [ 31,   0, 255], [ 15,   0, 255], [  0,   0, 255],  # dark blue
            [127, 205, 255], [127, 205, 255], [ 95, 205, 255],  # light blue
            [ 63, 205, 255], [ 31, 205, 255], [  0, 205, 255],  # light blue
            [255,   0,   0], [255,   0,   0],                   # red
            [191,  63,  63], [191,  63, 191],                   # magenta
            [255,   0, 127], [255,   0, 255],                   # purple
            [127, 255,   0], [ 63, 255,   0], [  0, 255,   0],  # green
            [255, 127,   0], [255, 191,   0], [255, 255,   0],  # yellow

        ]
    parent = [1, 8, 1, 2, 3, 1, 5, 6, -1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16, 14, 19, 14, 11, 22, 11]
