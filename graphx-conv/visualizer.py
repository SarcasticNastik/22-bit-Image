import numpy as np
import cv2
import sys
import os

changed = True
showsz = 800
mouse = [0.5, 0.5]
zoom = 1.0


def onMouse(*args):
    global mouse, changed
    mouse[0] = args[2] / float(showsz)
    mouse[1] = args[1] / float(showsz)
    changed = True


cv2.namedWindow('show3d')
cv2.moveWindow('show3d', 0, 0)
cv2.setMouseCallback('show3d', onMouse)


def showPoints(xyz, c0=None, c1=None, c2=None, wait_time=0, show_rotation=False,
               magnify_blue=0, freeze_rotation=False, background=(0, 0, 0),
               normalize_color=True):
    """
    c{0,1,2}: Color values for each point
    """
    global showsz, mouse, zoom, changed

    if len(xyz.shape) != 2 or xyz.shape[1] != 3:
        raise Exception('showpoints expects (n,3) shape for xyz')
    if c0 is not None and c0.shape != xyz.shape[:1]:
        raise Exception('showpoints expects (n,) shape for c0')
    if c1 is not None and c1.shape != xyz.shape[:1]:
        raise Exception('showpoints expects (n,) shape for c1')
    if c2 is not None and c2.shape != xyz.shape[:1]:
        raise Exception('showpoints expects (n,) shape for c2')

    xyz -= np.mean(xyz, axis=0)
    radius = np.sqrt((xyz ** 2).sum(axis=-1)).max()
    xyz /= radius * 2.2 / showsz

    if c0 is None:
        c0 = np.zeros((len(xyz),), dtype='float32') + 255
    if c1 is None:
        c1 = c0
    if c2 is None:
        c2 = c0
    if normalize_color:
        c0 /= (c0.max() + 1e-14) / 255.0
        c1 /= (c1.max() + 1e-14) / 255.0
        c2 /= (c2.max() + 1e-14) / 255.0

    # Size of the window
    show = np.zeros((showsz, showsz, 3), dtype='uint8')

    def render():
        rotation_matrix = np.eye(3)
        xangle = 0 if freeze_rotation else (mouse[1] - 0.5) * np.pi * 1.2

        print(
            f"Rotation matrix before x angle multiplication: {rotation_matrix}")
        # R after rotation in x-direction by xangle: R @ Rx'
        rotation_matrix = rotation_matrix @ (np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(xangle), -np.sin(xangle)],
            [0.0, np.sin(xangle), np.cos(xangle)],
        ]))

        # R after rotation in y-direction by xangle: R @ Ry'
        yangle = 0 if freeze_rotation else (mouse[0] - 0.5) * np.pi * 1.2
        rotation_matrix = rotation_matrix @ np.array([
            [np.cos(yangle), 0.0, -np.sin(yangle)],
            [0.0, 1.0, 0.0],
            [np.sin(yangle), 0.0, np.cos(yangle)],
        ])

        rotation_matrix *= zoom

        # matrix after rotation: xyz @ R
        nxyz = xyz.dot(rotation_matrix)
        # Sort the matrix by the maximum magnitude direction
        nz = nxyz[:, 2].argsort()
        # Convert the matrix by removing the last dimension (column here)
        nxyz = nxyz[nz]
        nxyz = (nxyz[:, :2] + [showsz / 2, showsz / 2]).astype('int32')
        # What does this do here?
        p = nxyz[:, 0] * showsz + nxyz[:, 1]
        show[:] = background
        m = (nxyz[:, 0] >= 0) * (nxyz[:, 0] < showsz) * (nxyz[:, 1] >= 0) * (
                nxyz[:, 1] < showsz)

        show.reshape((showsz * showsz, 3))[p[m], 1] = c0[nz][m]
        show.reshape((showsz * showsz, 3))[p[m], 2] = c1[nz][m]
        show.reshape((showsz * showsz, 3))[p[m], 0] = c2[nz][m]

        if magnify_blue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0],
                                       np.roll(show[:, :, 0], 1, axis=0))
            if magnify_blue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0],
                                       np.roll(show[:, :, 0], 1, axis=1))
            if magnify_blue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=1))
        if show_rotation:
            thickness = 0.5
            lineType = 2
            fontScale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(show, f"xangle {int(xangle / np.pi * 180)}",
                        (30, showsz - 30), font, fontScale, (255, 0, 0),
                        thickness, lineType)
            cv2.putText(show, f"xangle {int(xangle / np.pi * 180)}",
                        (30, showsz - 50), font, fontScale, (255, 0, 0),
                        thickness, lineType)
            cv2.putText(show, f"xangle {int(zoom * 100)}",
                        (30, showsz - 70), font, fontScale, (255, 0, 0),
                        thickness, lineType)

    while True:
        if changed:
            render()
            changed = False
        cv2.imshow('show3d', show)
        if wait_time == 0:
            cmd = cv2.waitKey(10) % 256
        else:
            cmd = cv2.waitKey(wait_time) % 256
        if cmd == ord('q'):
            break
        elif cmd == ord('Q'):
            sys.exit(0)
        if cmd == ord('n'):
            zoom *= 1.1
            changed = True
        elif cmd == ord('m'):
            zoom /= 1.1
            changed = True
        elif cmd == ord('r'):
            zoom = 1.0
            changed = True
        elif cmd == ord('s'):
            cv2.imwrite('show3d.png', show)
        if wait_time != 0:
            break
    return cmd


if __name__ == '__main__':
    # showPoints(np.random.rand(10000, 3))
    # green = np.linspace(0, 1, 10000)
    # red = np.linspace(1, 0, 10000) ** 0.5
    # blue = np.linspace(1, 0, 10000)
    # showPoints(np.random.rand(10000, 3), green, red, blue, magnify_blue=1)
    batch_num = 0
    file_path = input("> Please enter path to the respective .npy file: ")
    if os.path.isdir(file_path):
        batch = np.load(file_path)
    else:
        print("Wrong Path. Aborting")
        exit(0)
    example_num = int(input("> Enter the number of example [0-149]:"))
    showPoints(batch[example_num, :])
