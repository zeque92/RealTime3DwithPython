# -*- coding: utf-8 -*-
import pygame
import numpy as np
from os import chdir
from sys import exit


class BoxInABox:
    """
    Displays semi-transparent boxes inside one another.

    @author: kalle
    """

    def __init__(self, screen, target_fps, run_seconds):
        self.run_seconds = run_seconds
        self.screen = screen
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.screenCopy = self.screen.copy()
        self.alphaValue = 160.0
        self.font = pygame.font.SysFont('CourierNew', 14)               # initialize font and set font size
        self.backgroundColor = (0, 0, 0)
        self.midScreen = np.array([self.width / 2, self.height / 2], dtype=int)
        self.angleScale = (2.0 * np.pi) / 360.0  # to scale degrees.
        self.boxCount = 8
        self.boxColors = np.zeros((self.boxCount, 3, 3), dtype=int)     # for each box, three color sets (one for each visible surface)
        self.nodes = np.zeros((self.boxCount, 4, 3), dtype=float)       # nodes will have unrotated X,Y,Z coordinates. Just 4 needed per box as can be mirrored
        self.rotatedNodes = np.zeros((8, 3), dtype=float)               # rotatedNodes will have X,Y,Z coordinates after rotation
        self.transNodes = np.zeros((8, 2), dtype=int)                   # transNodes will have X,Y coordinates
        self.angles = np.zeros((self.boxCount, 3), dtype=float)         # angles for each box
        self.angleAdd = np.zeros((self.boxCount, 3), dtype=float)       # rotation for each box
        self.surfaces = np.zeros((6, 4), dtype=int)                     # surfaces are identical for each box - six surfaces of four nodes each.
        self.zoomZ = 0.0                                                # additional Z for inital zooming in of the boxes
        self.frameNodes = np.zeros((4, 3), dtype=float)                 # frameNodes will have unrotated X,Y,Z coordinates.
        self.frameAngles = np.zeros((3), dtype=float)                   # angles for frame
        self.frameAngleAdd = np.array([1.0, 2.5, 1.8])                  # frame rotation
        self.frameFrameCount = 180                                      # how many frames to bring the frame to front
        self.frameZoomCount = 220 + self.frameFrameCount                # how many frames to zoom the boxes
        self.frameRotateAllCount = 220 + self.frameZoomCount            # how many frames to rotate all together
        self.frameRotateDelayCount = 30                                 # how many frames to delay between starting each box rotating separately
        self.frameFadeCount = 120                                       # how many frames to use for fading colors
        self.frameCount = 0
        self.frameRect = pygame.Rect(0, 0, 0, 0)
        self.zPos = 3000.0
        self.zScale = 1000.0
        self.fullScreen = False
        self.clock = pygame.time.Clock()
        self.start_timer = pygame.time.get_ticks()
        self.target_fps = target_fps
        self.running = True
        self.stop = False

        self.screenCopy.set_colorkey(self.backgroundColor)
        self.screenCopy.set_alpha(self.alphaValue)
        self.screen.fill(self.backgroundColor)

        self.prepareBoxes()

    def run(self):
        """ Main loop. """

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        self.stop = True
                    if event.key == pygame.K_f:
                        self.toggleFullScreen()
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                                          self.__class__.__name__ + '.jpg')
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # left button: exit
                        self.running = False

            self.screen.lock()
            if self.frameCount < self.frameFrameCount:
                self.screen.fill(self.backgroundColor)
                self.flyFrame()
            else:
                # zoom position
                if self.frameCount < self.frameZoomCount:
                    self.screen.fill(self.backgroundColor, self.frameRect)  # clear inside of the frame only. Not needed when zooming is finished.
                    self.zoomZ = (1.0 - (self.frameCount - self.frameFrameCount) / (self.frameZoomCount - self.frameFrameCount)) * 12.0 * self.zPos
                else:
                    self.zoomZ = 0.0
                for i in range(self.boxCount):
                    self.addAngles(i)
                    self.rotateAndTransform(i)
                    self.drawBox(i)

            pygame.display.flip()
            # release any locks on screen
            while self.screen.get_locked():
                self.screen.unlock()

            self.clock.tick(self.target_fps)  # this keeps code running at max target_fps
            self.frameCount += 1

            if pygame.time.get_ticks() > self.start_timer + 1000 * self.run_seconds:
                self.running = False

        return self.stop

    def flyFrame(self):

        if self.frameCount < self.frameFrameCount:
            # set frame position
            if self.frameCount < self.frameFrameCount / 2:
                # move only from left to right
                frame_pos = np.array([(1.0 - self.frameCount / (self.frameFrameCount - 1)) * -7.0 * self.width, 0, 5.0 * self.zPos])
            else:
                # using sin from 0 to 90 degrees to curve closer
                sin_value = 1.0 - (np.sin(90.0 * (self.frameCount - (self.frameFrameCount - 1) / 2) / ((self.frameFrameCount - 1) / 2) * (2.0 * np.pi / 360)))
                cos_value = (np.cos(90.0 * (self.frameCount - (self.frameFrameCount - 1) / 2) / ((self.frameFrameCount - 1) / 2) * (2.0 * np.pi / 360)))
                frame_pos = np.array([sin_value * -3.5 * self.width, 0, cos_value * 5.0 * self.zPos])
            # add rotation angles
            self.frameAngles += self.frameAngleAdd
        else:
            frame_pos = np.array([0, 0, 0])

        # Set matrix for rotation and rotate and transform nodes.
        (sx, sy, sz) = np.sin(self.frameAngles * self.angleScale)
        (cx, cy, cz) = np.cos(self.frameAngles * self.angleScale)
        # build a matrix for X, Y, Z rotation (in that order, see Wikipedia: Euler angles).
        matrix = np.array([
            [cy * cz               , -cy * sz              , sy      ],
            [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
            [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]
            ])

        # rotate nodes
        self.rotatedNodes = np.dot(self.frameNodes, matrix)
        # use scaling to add remaining nodes
        self.rotatedNodes = np.vstack((self.rotatedNodes, self.rotatedNodes[0:4, 0:3] * 0.99))
        self.rotatedNodes = np.vstack((self.rotatedNodes, self.rotatedNodes[0:4, 0:3] * 0.98))
        # transform 3D to 2D by dividing X and Y with Z coordinate. Add frame Z pos to rotatedNodes
        self.transNodes = ((self.rotatedNodes[:, 0:2] + frame_pos[0:2])
                           / ((self.rotatedNodes[:, 2:3] + frame_pos[2] + self.zPos)) * self.zScale) + self.midScreen

        # draw outer frame
        node_list = self.cropEdges((
                self.transNodes[0, :],
                self.transNodes[1, :],
                self.transNodes[2, :],
                self.transNodes[3, :]
                ))
        self.drawPolygon(self.screen,  (200, 200, 200),  node_list)
        # draw inner frame
        node_list = self.cropEdges((
                self.transNodes[4, :],
                self.transNodes[5, :],
                self.transNodes[6, :],
                self.transNodes[7, :]
                ))
        self.drawPolygon(self.screen,  (160, 160, 160),  node_list)
        # draw black center and store it
        node_list = self.cropEdges((
                self.transNodes[ 8, :],
                self.transNodes[ 9, :],
                self.transNodes[10, :],
                self.transNodes[11, :]
                ))
        rect = self.drawPolygon(self.screen, (0, 0, 0), node_list)
        if self.frameCount + 1 >= self.frameFrameCount:
            # last draw: store this rect as active drawing area
            self.frameRect = rect

    def addAngles(self, boxNo):

        # rotate the box by changing its "angles". Until delay is over, use first box angleadds
        if self.frameCount > self.frameRotateAllCount + (7-boxNo) * self.frameRotateDelayCount:
            self.angles[boxNo, :] += self.angleAdd[boxNo, :]
        else:
            self.angles[boxNo, :] += self.angleAdd[0, :]

        for j in range(3):
            if self.angles[boxNo, j] > 360:
                self.angles[boxNo, j] -= 360
            if self.angles[boxNo, j] < 0:
                self.angles[boxNo, j] += 360

    def rotateAndTransform(self, boxNo):

        # Set matrix for rotation and rotate and transform nodes.
        (sx, sy, sz) = np.sin(self.angles[boxNo, :] * self.angleScale)
        (cx, cy, cz) = np.cos(self.angles[boxNo, :] * self.angleScale)
        # build a matrix for X, Y, Z rotation (in that order, see Wikipedia: Euler angles).
        matrix = np.array([
            [cy * cz               , -cy * sz              , sy      ],
            [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
            [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]
            ])

        # rotate nodes
        self.rotatedNodes = np.dot(self.nodes[boxNo, :, :], matrix)
        # use "mirroring" to add four remaining nodes
        self.rotatedNodes = np.vstack((self.rotatedNodes, -self.rotatedNodes))
        # transform 3D to 2D by dividing X and Y with Z coordinate
        self.transNodes = (self.rotatedNodes[:, 0:2] / ((self.rotatedNodes[:, 2:3] + self.zPos + self.zoomZ)) * self.zScale) + self.midScreen

    def drawBox(self, boxNo):

        # draw the first box directly on screen, then on a copy to perform blend
        if boxNo == 0:
            screen_use = self.screen
        else:
            screen_use = self.screenCopy

        # check if fading colors needed
        if self.frameCount <= self.frameFrameCount + self.frameFadeCount:
            fade = (self.frameCount - self.frameFrameCount) / self.frameFadeCount
        else:
            fade = 1.0

        rect_use = None
        screen_use.lock()
        # loop through each surface and draw the visible ones.
        for j in range(6):
            # calculate surface cross product vector
            vec_Viewer = self.rotatedNodes[self.surfaces[j, 1], :] + [0, 0, self.zPos + self.zoomZ]  # vec_Ciewer is from viewers (0, 0, 0) to node 1
            vec_A = self.rotatedNodes[self.surfaces[j, 2], :] - self.rotatedNodes[self.surfaces[j, 1], :]  # vec_A is from node 1 to node 2
            vec_B = self.rotatedNodes[self.surfaces[j, 0], :] - self.rotatedNodes[self.surfaces[j, 1], :]  # vec_B is from node 1 to node 0
            vec_Cross = ([
                vec_B[1] * vec_A[2] - vec_B[2] * vec_A[1],
                vec_B[2] * vec_A[0] - vec_B[0] * vec_A[2],
                vec_B[0] * vec_A[1] - vec_B[1] * vec_A[0]
                ])
            if np.dot(vec_Cross, vec_Viewer) < 0:
                # surface is visible to Viewer
                node_list = self.cropEdges((
                        self.transNodes[self.surfaces[j, 0], :],
                        self.transNodes[self.surfaces[j, 1], :],
                        self.transNodes[self.surfaces[j, 2], :],
                        self.transNodes[self.surfaces[j, 3], :]
                        ))

                # prepare color
                color = [int(x * fade) for x in self.boxColors[boxNo, (boxNo + j) % 3]]
                # draw polygon and store the area used.
                rect = self.drawPolygon(screen_use, color, node_list)
                if rect_use is None:
                    rect_use = rect
                else:
                    rect_use = rect_use.union(rect)

        # for other than the first box, blend them to the screen
        if boxNo > 0:
            # blit the semitransparent screen copy to main screen
            # release any locks on screen
            while self.screen.get_locked():
                self.screen.unlock()
            while self.screenCopy.get_locked():
                self.screenCopy.unlock()
            self.screen.blit(self.screenCopy, rect_use, rect_use)
            # clear the screen copy after use
            self.screenCopy.fill((0, 0, 0), rect_use)

    def cropEdges(self, node_list, cropX=True, cropY=True):

        # crop to screen size. "Auto crop" does not seem to work if points very far outside.
        # takes list of nodes (X,Y) in drawing order as input.
        # returns list of nodes (X,Y) cropped to screen edges.
        # crop both X, Y, if cropX and cropY = True; X: i=0, Y: i=1

        edges = np.zeros((4))
        edges[0] = self.frameRect[0]
        edges[1] = self.frameRect[1]
        edges[2] = self.frameRect[2] + edges[0]  # Rect has left. top, width, height --> change to left, top, right, bottom
        edges[3] = self.frameRect[3] + edges[1]
        if self.frameCount >= 180:
            i = 1
        if len(node_list) > 2:
            for i in range(2):
                if (i == 0 and cropX is True) or (i == 1 and cropY is True):
                    crop_nodes = []  # empty list
                    prev_node = node_list[-1]
                    for node in node_list:
                        diff_node = node - prev_node  # surface side vector
                        # start cropping from prev_node direction, as order must stay the same
                        if node[i] >= edges[i] and prev_node[i] < edges[i]:
                            # line crosses min, so add a "crop point". Start from previous node and add difference stopping to min
                            crop_nodes.append(prev_node + diff_node * ((edges[i] - prev_node[i]) / diff_node[i]))
                        if node[i] <= edges[i+2] and prev_node[i] > edges[i+2]:
                            # line crosses max, so add a "crop point". Start from previous node and add difference stopping to max
                            crop_nodes.append(prev_node + diff_node * ((edges[i+2] - prev_node[i]) / diff_node[i]))
                        # then crop current node
                        if node[i] < edges[i] and prev_node[i] >= edges[i]:
                            # line crosses min, so add a "crop point". Start from previous node and add difference stopping to min
                            crop_nodes.append(prev_node + diff_node * ((edges[i] - prev_node[i]) / diff_node[i]))
                        if node[i] > edges[i+2] and prev_node[i] <= edges[i+2]:
                            # line crosses max, so add a "crop point". Start from previous node and add difference stopping to max
                            crop_nodes.append(prev_node + diff_node * ((edges[i+2] - prev_node[i]) / diff_node[i]))
                        # always add current node, if it is on screen
                        if node[i] >= edges[i] and node[i] <= edges[i+2]:
                            crop_nodes.append(node)
                        prev_node = node
                    # for next i, copy results. Quit loop if no nodes to look at
                    node_list = crop_nodes
                    if len(node_list) < 3:
                        break
            # convert to integers
            node_list = [(int(x[0] + 0.5), int(x[1] + 0.5)) for x in node_list]
        return node_list

    def drawPolygon(self, screen, color, node_list):

        # draws a filled antialiased polygon, returns its Rect (area covered)
        if len(node_list) > 2:  # a polygon needs at least 3 nodes
            # pygame.draw.aalines(screen, color, True, node_list)
            return pygame.draw.polygon(screen, color, node_list, 0)
        else:
            return pygame.Rect(0, 0, 0, 0)

    def toggleFullScreen(self):

        pygame.display.toggle_fullscreen()
        if self.frameCount > self.frameFrameCount:
            self.frameRect = pygame.Rect(0, 0, self.width, self.height)  # reset drawing area
            self.flyFrame()  # redraw frame

    def prepareBoxes(self):

        # set up boxes
        self.boxCount = 8

        colorDiff = 15.0  # difference in color values (tint/tone) between box surfaces
        # colorset represents target colors. Note depending on transparency setting (self.alphaValue) not all colors can be reached with blending
        colorSet = np.array([
            [ 84, 152, 187],
            [187, 102,  84],
            [187, 152,  84],
            [187,  84, 152],
            [ 84, 102, 187],
            [ 84, 187, 102],
            [102,  84, 121],
            [187, 187, 187]
            ])

        # calculate blend colors so that resulting blend is close to target.
        # target may not be possible to get to, hence "clipping" colors to range [0,255]
        alpha_m = 1.0  # first box only
        alpha_m2 = 1.0 - alpha_m
        prev_color = np.array([0, 0, 0])
        for i in range(self.boxCount):
            colorLen = sum(colorSet[i, :]) / 3
            self.boxColors[i, 0, 0:3] = (colorSet[i] - alpha_m2 * prev_color) / alpha_m
            self.boxColors[i, 1, 0:3] = (colorSet[i] * ((colorLen + colorDiff) / colorLen) - alpha_m2 * prev_color) / alpha_m
            self.boxColors[i, 2, 0:3] = (colorSet[i] * ((colorLen - colorDiff) / colorLen) - alpha_m2 * prev_color) / alpha_m
            prev_color = colorSet[i]
            # crude check that colors remain valid. Also, if gone outside of valid range, the blend will not reach target color - adjust prev_color accordingly.
            for j in range(3):
                for k in range(3):
                    if self.boxColors[i, j, k] < 0:
                        if j == 0:
                            prev_color[k] = colorSet[i][k] - self.boxColors[i, j, k] * alpha_m
                        self.boxColors[i, j, k] = 0
                    elif self.boxColors[i, j, k] > 255:
                        if j == 0:
                            prev_color[k] = colorSet[i][k] - (self.boxColors[i, j, k] - 255) * alpha_m
                        self.boxColors[i, j, k] = 255
            alpha_m = self.alphaValue / 255.0
            alpha_m2 = 1.0 - alpha_m

        # set box nodes.
        node_size = self.height * 1.4
        node_scale = node_size / np.sqrt(node_size ** 2 * 3)    # node_scale sets next box size such that it will remain inside the previous, bigger box even when rotating
        for i in range(self.boxCount):
            # a box has 8 nodes but since the sides are mirror images, 4 is sufficient for rotation. X increases to the right, Y down, and Z to distance
            self.nodes[i, 0, :] = np.array([ node_size,  node_size,  node_size]) * (node_scale ** i)
            self.nodes[i, 1, :] = np.array([ node_size,  node_size, -node_size]) * (node_scale ** i)
            self.nodes[i, 2, :] = np.array([-node_size,  node_size, -node_size]) * (node_scale ** i)
            self.nodes[i, 3, :] = np.array([-node_size,  node_size,  node_size]) * (node_scale ** i)

        # set up surfaces. Each surface is defined by four nodes. Nodes must be used in clockwise order.
        # surface order is used in coloring - three colors per box in surfaces (0,3), (1,4), and (2,5) so these should be on opposite sides
        self.surfaces = np.array([
            [0, 3, 2, 1],
            [0, 1, 7, 6],
            [1, 2, 4, 7],
            [4, 5, 6, 7],
            [2, 3, 5, 4],
            [3, 0, 6, 5]
            ])

        # set rotation for box angles.
        self.angleAdd = np.array([
            [1.0, 2.5, 1.7],
            [1.7, 1.8, 0.9],
            [0.7, 1.5, 2.7],
            [2.0, 0.5, 0.7],
            [1.3, 1.5, 0.7],
            [0.5, 1.1, 2.3],
            [2.4, 1.5, 0.8],
            [1.1, 1.5, 1.3],
            ])

        # set frameNodes.
        if self.width > self.height * 5 / 4:
            frame_size = self.height * self.zPos / self.zScale / 1.8
        else:
            frame_size = (self.width / (5 / 4)) * self.zPos / self.zScale / 1.8

        self.frameNodes = np.array([
            [ frame_size, frame_size / (5 / 4), 0],
            [ frame_size,-frame_size / (5 / 4), 0],
            [-frame_size,-frame_size / (5 / 4), 0],
            [-frame_size, frame_size / (5 / 4), 0]
            ])

        # set frame angles so that will stop at (0, 0, 0).
        self.frameAngles = -self.frameFrameCount * self.frameAngleAdd
        self.frameAngles = (np.abs(self.frameAngles) % 360) * np.sign(self.frameAngles) - ((np.sign(self.frameAngles) - 1) * 180)

        self.frameRect = pygame.Rect(0, 0, self.width, self.height)


if __name__ == '__main__':
    """
    Prepare screen, etc.
    """

    # set data directory
    chdir("C:/Users/Kalle/OneDrive/Asiakirjat/Python")

    # set screen size
    # first check available full screen modes
    pygame.display.init()
    # pick disp0lay mode from list or set a specific resolution
    disp_modes = pygame.display.list_modes(0, pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    # disp_size = disp_modes[9] # selecting display size from available list. Assuming the 9th element is nice...
    # disp_size = (800, 600)  # to force display size
    disp_size = (1920, 1080)  # to force display size
    pygame.font.init()

    # initialize mixer
    pygame.mixer.init()
    music_file = "sinking2.ogg"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/release.php?id=45536
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play()

    screen = pygame.display.set_mode(disp_size)
    pygame.display.set_caption('Box in a Box in a Box')
    BoxInABox(screen, 60, 120).run()

    # exit; close display, stop music
    pygame.quit()
    exit()
