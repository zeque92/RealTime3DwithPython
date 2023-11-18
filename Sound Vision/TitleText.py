# -*- coding: utf-8 -*-
import pygame
import numpy as np
from os import chdir
from sys import exit


class TitleText:
    """
    Displays a morphing / blurring red-yellow image.

    @author: kalle
    """

    def __init__(self, screen, target_fps):
        self.screen = screen
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.screenCopy = self.screen.copy()
        self.screenCopy2 = self.screen.copy()
        self.screenStripes = self.screen.copy()
        self.screenCount = 126                                          # nr of separately stored images. Images are stored so the "oldest" one can be subtracted from final image.
                                                                        # obviously, it would be possible to store e.g. only the "transNodes" and redraw these images as needed - slower but memory-friendlier.
        self.screenCounter = self.screenCount - 1
        self.screenList = []
        self.rectList = []
        self.font = pygame.font.SysFont('CourierNew', 14)               # initialize font and set font size
        self.clock = pygame.time.Clock()
        self.title = [900, 110, 'PRESENTS', 900, -110, 'REFLECT']       # list of title texts, in reverse order: text, Y-position, word total width
        self.title2 = [700, 110, 'VISION', 700, -110, 'SOUND']          # list of title texts, in reverse order: text, Y-position, word total width
        self.phase = 0                                                  # used to steer code between phases
        self.currTitle = []
        self.currLetter = ''
        self.LetterSpacing = 0
        self.letterFrames = 0
        self.letterFrameCount = 90
        self.letterZBeg = 120
        self.letterZPos = 1000
        self.letterAngleAdds = np.zeros((3, 3), dtype=float)
        self.letterAngleNr = 0
        self.currLetterSize = np.array([0, 0])
        self.currPos = np.array([0, 0])
        self.backgroundColor = (0, 0, 0)
        self.letterColor = (30, 0, 200)
        self.stripeColor = (128, 128, 128)
        self.titleColor = (20, 0, 200)
        self.fadeFrames = 0
        self.fadeFrameCount = 120
        self.objColor = (2, 0, 0)                                       # using red colour to set the shadecolor from the palette - the R component must be integer < 256 / screenCount
        self.nrImagesPerFrame = 3                                       # how many images to draw per frame. Larger number will make the effect smoother.
        self.palette = np.zeros((256, 3), dtype=np.uint8)
        self.midScreen = np.array([self.width / 2, self.height / 2], dtype=int)
        self.angleScale = (2.0 * np.pi) / 360.0
        self.stripeNodes = np.zeros((4, 2), dtype=float)                # nodes for drawing the stripes.
        self.nodes = np.zeros((0, 2), dtype=float)                      # nodes will have unrotated X,Y coordinates - Z is always zero.
        self.titleNodes = np.zeros((0, 2), dtype=float)                 # titleNodes will store the titletext in its entirety.
        self.titleNodes2 = np.zeros((0, 2), dtype=float)                # titleNodes2 will store the titletext 2 in its entirety.
        self.rotatedNodes = np.zeros((0, 3), dtype=float)               # rotatedNodes will have X,Y,Z coordinates after rotation
        self.transNodes = np.zeros((0, 2), dtype=int)                   # transNodes will have X,Y coordinates "flattened" to 2D from rotatedNodes
        self.morphNodes = np.zeros((0, 2), dtype=int)                   # list of node pairs from title nodes to title2 nodes for morphing
        self.angles = np.zeros((3), dtype=float)                        # angles
        self.angleAdd = np.zeros((3), dtype=float)                      # rotation
        self.rotationMatrix = np.zeros((3, 3), dtype=float)
        self.letters = {}                                               # a dictionary containing letter coordinates
        self.surfaces = []                                              # surfaces connect nodes. The first item is 1 for color, 0 for background.
        self.titleSurfaces = []                                         # stores the whole of titletext surfaces.
        self.titleSurfaces2 = []                                        # stores the whole of titletext 2 surfaces.
        self.zPos = 1000.0
        self.zScale = self.height
        self.barYPos = (self.title[1] * 2.1) * self.zScale / self.letterZPos   # Y position of the vertical bar, relative to screen center
        self.barSize = int(12 * self.zScale / self.letterZPos)
        self.barFrameCount = 120
        self.barFrames = 0
        self.morphFrameCount = 240 * 3 / self.nrImagesPerFrame
        self.morphFrames = 0
        self.zoomFrameCount = 360 * 3 / self.nrImagesPerFrame
        self.zoomFrames = 0
        self.frameCount = 0
        self.target_fps = target_fps
        self.running = True
        self.stop = False
        self.rectUsed = [0, 0, 0, 0]
        self.infoDisplay = False
        self.millisecs = 0
        self.timer_avg_frames = 180
        self.timer_names = []
        self.timers = np.zeros((1, 1), dtype=int)
        self.timer_frame = 0
        self.start_timer = 0
        self.maxcol1 = np.zeros((3), dtype=int)
        self.maxcol2 = np.zeros((3), dtype=int)

        self.timer_names.append("draw image")
        self.timer_names.append("blend images")
        self.timer_names.append("colorize")
        self.timer_names.append("plot info")
        self.timer_names.append("display flip")
        self.timer_names.append("wait")
        self.timers = np.zeros((len(self.timer_names), self.timer_avg_frames), dtype=int)

        self.prepareLetters()
        self.screen.fill(self.backgroundColor)

    def run(self):
        """ Main loop. """

        # initialize timers
        self.start_timer = pygame.time.get_ticks()
        self.millisecs = self.start_timer
        self.newLetter()
        self.screenCopy.set_colorkey(self.backgroundColor)
        self.screenCopy2.fill(self.backgroundColor)

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
                    if event.key == pygame.K_i:
                        self.toggleInfoDisplay()
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                                          self.__class__.__name__ + '.jpg')
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # left button: exit
                        self.running = False

            if self.phase == 0:
                # phase 0: fly letters from viewer to form title text
                if self.letterFrames > self.letterFrameCount:
                    self.newLetter()
                if self.letterFrames <= self.letterFrameCount:
                    self.screen.blit(self.screenCopy2, (0, 0))  # initailize screen with "background" ie. already finished letters.
                    self.slideBars(1, self.screenCopy2)
                    self.flyLetter()

            if self.phase == 1:
                # phase 1: build a whole title text (one pass only) and copy it to screenList screens for the blending phase
                self.screen.set_alpha(None)
                self.screenCopy.set_alpha(None)
                self.screen.set_colorkey(None)
                self.screenCopy.set_colorkey(None)
                self.buildTitle()

            if self.phase == 2:
                # phase 2: fade out the stripes
                self.fadeTitle()

            if self.phase == 3:
                # phase 3: zoom out and in the title text
                self.zoomTitle(1)

            if self.phase == 4:
                # phase 4: build the second title text
                self.title = self.title2
                self.titleNodes2 = np.zeros((0, 2), dtype=float)
                self.titleSurfaces2 = []
                while self.phase == 4:
                    self.newLetter()
                self.zPos = self.letterZPos
                self.morphSetup()
                self.barFrames = self.barFrameCount - 1

            if self.phase == 5:
                # phase 5: metamorph the title text
                self.morphTitle()

            if self.phase == 6:
                # phase 6: build a whole title text (one pass only) and copy it to screenList screens for the blending phase
                self.screen.set_alpha(None)
                self.screenCopy.set_alpha(None)
                self.screen.set_colorkey(None)
                self.screenCopy.set_colorkey(None)
                self.buildTitle()

            if self.phase == 7:
                # phase 7: mirror the title text
                self.zoomTitle(2)

            if self.phase == 8:
                # phase 8: fade to black
                self.fadeTitleOut()
                self.slideBars(-1, self.screen)
                if self.barFrames < 0:
                    self.phase += 1

            if self.phase >= 9:
                # exit
                self.running = False

            if self.infoDisplay:
                self.plotInfo()
            self.nextTimeFrame()
            self.measureTime("plot info")
            pygame.display.flip()
            # release any locks on screen
            while self.screen.get_locked():
                self.screen.unlock()

            self.measureTime("display flip")
            self.clock.tick(self.target_fps)  # this keeps code running at max target_fps
            self.measureTime("wait")
            self.frameCount += 1

        return self.stop

    def slideBars(self, bfAdd, screen):

        # slide the blue bars in place or out in the background.
        if self.phase == 8:
            color1 = self.letterColor
        if self.barFrames >= 0 and self.barFrames < self.barFrameCount:
            self.barFrames += bfAdd
            color1 = self.letterColor
            color2 = ([round(0.5 * x, 0) for x in self.letterColor])
            color3 = ([round(0.3 * x, 0) for x in self.letterColor])
            y0 = int((self.barFrames - 1) / self.barFrameCount * (self.midScreen[1] - self.barYPos))
            y1 = int((self.barFrames    ) / self.barFrameCount * (self.midScreen[1] - self.barYPos))
            y2 = y1 + self.barSize
            y3 = int(y1 + self.barSize / 4)
            y4 = int(y2 - self.barSize / 4)
            y5 = int(y2 + self.barSize / 4)
            if y0 >= 0:
                pygame.draw.rect(screen, color3, (0, y0, self.width, y1 - y0))
                pygame.draw.rect(screen, color3, (0, self.height - y1, self.width, y1 - y0))
            if y1 >= 0:
                pygame.draw.rect(screen, color2, (0, y1, self.width, y2 - y1))
                pygame.draw.rect(screen, color2, (0, self.height - y2, self.width, y2 - y1))
                pygame.draw.rect(screen, color1, (0, y3, self.width, y4 - y3))
                pygame.draw.rect(screen, color1, (0, self.height - y4, self.width, y4 - y3))
                pygame.draw.rect(screen, self.backgroundColor, (0, y4, self.width, y5 - y4))
                pygame.draw.rect(screen, self.backgroundColor, (0, self.height - y5, self.width, y5 - y4))

    def newLetter(self):

        if len(self.currTitle) == 0:
            # new "word" needed
            if len(self.title) == 0:
                # all done.
                self.phase += 1
            else:
                # get next title word.
                wd = self.title.pop()
                self.currPos[1] = self.title.pop()
                wd_width = self.title.pop()
                letterWidthSum = 0
                # build a list of letters and calculate their total width.
                self.currTitle = []
                max_x_wd = 0
                max_y_wd = 0
                for lt in wd:
                    self.currTitle.append(lt)
                    max_x = 0
                    max_y = 0
                    for e in self.letters[lt]:
                        max_x = max(max_x, np.max(e[1][:, 0]))
                        max_y = max(max_y, np.max(e[1][:, 1]))
                    letterWidthSum += max_x
                    max_x_wd = max(max_x_wd, max_x)
                    max_y_wd = max(max_y_wd, max_y)

                self.letterSpacing = int((wd_width - letterWidthSum) / (len(wd) - 1))
                self.currPos[0] = int(-wd_width / 2 - self.letterSpacing)
                self.currLetterSize = np.array([0, 0], dtype=float)
                self.stripeNodes = np.array([
                    [       0,       0],
                    [max_x_wd,       0],
                    [max_x_wd,max_y_wd],
                    [       0,max_y_wd]
                    ], dtype=float)
                self.currTitle.reverse()  # reverse order for "popping".

        if not len(self.currTitle) == 0:
            # next letter
            lt = self.currTitle.pop()
            prev_letterSize = self.currLetterSize
            self.currLetter = lt
            self.surfaces = []
            # loop through letter elements
            for e in self.letters[lt]:
                # build nodes and surfaces
                if len(self.surfaces) == 0:
                    # first surface
                    node_list = []
                    node_list.append(e[0])  # first item: 0 for holes, 1 for filled.
                    node_list.extend(i for i in range(np.shape(e[1])[0]))  # nodes are just drawn in sequence.
                    self.surfaces.append(node_list)
                    self.nodes = e[1]
                else:
                    # additional surfaces
                    node_list = []
                    node_list.append(e[0])  # first item: 0 for holes, 1 for filled
                    node_list.extend(i + np.shape(self.nodes)[0] for i in range(np.shape(e[1])[0]))  # nodes are just drawn in sequence.
                    self.surfaces.append(node_list)
                    self.nodes = np.vstack((self.nodes, e[1]))
            self.currLetterSize = np.array([np.max(self.nodes[:, 0]), np.max(self.nodes[:, 1])], dtype=float)
            self.currPos[0] += int(prev_letterSize[0] + self.currLetterSize[0]) / 2 + self.letterSpacing

            # store these to titletext 2 data, adding currPos
            node_count = np.shape(self.titleNodes2)[0]
            self.titleNodes2 = np.vstack((self.titleNodes2, self.nodes - np.array([int(self.currLetterSize[0] / 2),
                                                                                   int(self.currLetterSize[1] / 2)]) + self.currPos))
            for surface in self.surfaces:
                surf = surface.copy()  # don't mess with the original
                surf[1:] = [s + node_count for s in surf[1:]]  # fix node numbering by adding previous node_count
                self.titleSurfaces2.append(surf)

            self.zPos = self.letterZBeg
            self.letterFrames = 0
            # preset angles so that at the end letter will be at zero angles.
            self.angleAdd = self.letterAngleAdds[self.letterAngleNr]
            self.letterAngleNr += 1
            if self.letterAngleNr >= np.shape(self.letterAngleAdds)[0]:
                self.letterAngleNr = 0
            self.angles = (-self.angleAdd * (self.letterFrameCount + 1)) % 360

    def flyLetter(self):

        # show a striped letter, flying from viewer to title text
        self.addAngles()
        self.zPos += (self.letterZPos - self.letterZBeg) / self.letterFrameCount
        self.rotateAndTransformPlanar()
        self.drawLetter()
        self.rotateAndTransformStripes()
        self.drawStripes()
        self.copyLetter()
        self.measureTime("draw image")
        self.letterFrames += 1

    def addAngles(self):

        # rotate by changing "angles".
        self.angles += self.angleAdd

        for j in range(3):
            if self.angles[j] > 360:
                self.angles[j] -= 360
            elif self.angles[j] < 0:
                self.angles[j] += 360

        # PLANAR: Assumes Z coordinate for all nodes is zero. Hence no need for row 3 of the matrix and a simpler dot product.
        # Set matrix for rotation and rotate and transform nodes.
        (sx, sy, sz) = np.sin(self.angles * self.angleScale)
        (cx, cy, cz) = np.cos(self.angles * self.angleScale)
        # build a matrix for X, Y, Z rotation (in that order, see Wikipedia: Euler angles).
        self.rotationMatrix = np.array([
            [cy * cz               , -cy * sz              , sy      ],
            [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx] #,
            # [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]
            ])

    def rotateAndTransformPlanar(self):

        # rotate nodes - Planar. Use currLetterSize to center and letter position for position
        self.rotatedNodes = np.matmul(self.nodes - self.currLetterSize / 2, self.rotationMatrix) \
            + [(self.currPos[0] * (self.letterFrames / self.letterFrameCount)), (self.currPos[1] * (self.letterFrames / self.letterFrameCount)), 0]
        # transform 3D to 2D by dividing X and Y with Z coordinate
        self.transNodes = (self.rotatedNodes[:, 0:2] / ((self.rotatedNodes[:, 2:3]) + self.zPos) * self.zScale) + self.midScreen

    def rotateAndTransformStripes(self):

        # rotate nodes - Planar
        self.rotatedNodes = np.matmul(self.stripeNodes - self.currLetterSize / 2, self.rotationMatrix) \
            + [(self.currPos[0] * (self.letterFrames / self.letterFrameCount)), (self.currPos[1] * (self.letterFrames / self.letterFrameCount)), 0]
        # nodes above only have the corners of a plane. Add 3 points to each side
        s_nodes = np.zeros((16, 3), dtype=float)
        for i in range(4):
            m1 = i / 4.0
            m2 = 1.0 - m1
            s_nodes[i    ] = m2 * self.rotatedNodes[0] + m1 * self.rotatedNodes[1]
            s_nodes[i + 4] = m2 * self.rotatedNodes[1] + m1 * self.rotatedNodes[2]
            s_nodes[i + 8] = m2 * self.rotatedNodes[2] + m1 * self.rotatedNodes[3]
            s_nodes[i +12] = m2 * self.rotatedNodes[3] + m1 * self.rotatedNodes[0]

        # transform 3D to 2D by dividing X and Y with Z coordinate
        self.transNodes = (s_nodes[:, 0:2] / ((s_nodes[:, 2:3]) + self.zPos) * self.zScale) + self.midScreen

    def drawLetter(self):

        # draw the rotated letter
        i = 0
        screen = self.screenCopy
        for surface in self.surfaces:
            if surface[0] == 1:
                color = self.letterColor
            else:
                color = self.backgroundColor
            node_list = ([self.transNodes[node][:2] for node in surface[1:]])
            rect = self.drawPolygon(screen, color, node_list)
            if i == 0:
                # first and largest shape: store its Rect
                self.rectUsed = rect
                i = 1

    def drawStripes(self):

        # draw stripes
        screen = self.screenStripes
        screen.fill((255, 255, 255), self.rectUsed)  # clear to white
        color = self.stripeColor
        node_list = ([self.transNodes[node][:2] for node in (0, 1, 15)])
        self.drawPolygon(screen, color, node_list)
        node_list = ([self.transNodes[node][:2] for node in (2, 3, 13, 14)])
        self.drawPolygon(screen, color, node_list)
        node_list = ([self.transNodes[node][:2] for node in (4, 5, 11, 12)])
        self.drawPolygon(screen, color, node_list)
        node_list = ([self.transNodes[node][:2] for node in (6, 7, 9, 10)])
        self.drawPolygon(screen, color, node_list)

    def copyLetter(self):

        # add stripes to the letter and copy it to the screen
        self.screenCopy.blit(self.screenStripes, self.rectUsed, self.rectUsed, pygame.BLEND_MULT)
        self.screen.blit(self.screenCopy, self.rectUsed, self.rectUsed)
        # if letter is ready, copy it to the "backgroud screen"
        if self.letterFrames == self.letterFrameCount:
            self.screenCopy2.blit(self.screenCopy, self.rectUsed, self.rectUsed)
        # clear the letter for letters
        self.screenCopy.fill(self.backgroundColor, self.rectUsed)

    def buildTitle(self):

        # make a whole title text on screenCopy2, and prepare screenList by drawing it to each as well.
        # also draw it on screenCopy using a "full" objColor (as if it was a blend of all screenList screens)
        # move data from title 2 to title
        self.titleNodes = self.titleNodes2.copy()
        self.titleSurfaces = self.titleSurfaces2.copy()
        # transNodes need no rotation and their Z is zero - simple scaling by Z position
        self.transNodes = (self.titleNodes[:, 0:2] / self.zPos * self.zScale) + self.midScreen

        screen2 = self.screenCopy2
        screen2.fill(self.backgroundColor)
        self.rectUsed = None
        self.draw(screen2, self.titleColor, (1.0, 1.0))

        screen = self.screenCopy
        screen.fill(self.backgroundColor)
        color = ([int(x * (self.screenCount - 1)) for x in self.objColor])
        self.draw(screen, color, (1.0, 1.0))

        screenObj = self.screenList[0]
        screenObj.fill(self.backgroundColor)
        self.draw(screenObj, self.objColor, (1.0, 1.0))

        # copy image to all screens in screenList
        self.rectList[0] = self.rectUsed.copy()
        for i in range(1, len(self.screenList)):
            self.screenList[i] = self.screenList[0].copy()
            self.rectList[i] = self.rectUsed.copy()
        self.phase += 1

    def zoomTitle(self, mode):

        # zoom the title text away and back in (mode = 1) OR mirror it and mirror it back (mode <> 1). Zooms for 1/4 of frame count, then stays put for 1/4
        for i in range(self.nrImagesPerFrame):
            if mode == 1:
                # zoom
                if self.zoomFrames * self.nrImagesPerFrame + i <= self.zoomFrameCount * self.nrImagesPerFrame / 2:
                    zoomX = max(0.0, 1.0 - ((self.zoomFrames * self.nrImagesPerFrame + i) / (self.zoomFrameCount * self.nrImagesPerFrame / 4)))
                else:
                    zoomX = min(1.0, (self.zoomFrames * self.nrImagesPerFrame + i - self.zoomFrameCount * self.nrImagesPerFrame / 2)
                                / (self.zoomFrameCount * self.nrImagesPerFrame / 4))
                zoomY = zoomX
            else:
                # mirror / flip horizontally
                if self.zoomFrames * self.nrImagesPerFrame + i <= self.zoomFrameCount * self.nrImagesPerFrame / 2:
                    zoomX = max(-1.0, 1.0 - ((self.zoomFrames * self.nrImagesPerFrame + i) / (self.zoomFrameCount * self.nrImagesPerFrame / 8)))
                else:
                    zoomX = min(1.0, (self.zoomFrames * self.nrImagesPerFrame + i - self.zoomFrameCount * self.nrImagesPerFrame / 1.6)
                                / (self.zoomFrameCount * self.nrImagesPerFrame / 8))
                zoomY = 1.0

            active_screen = self.screenList[self.screenCounter]
            active_screen.fill(self.backgroundColor, self.rectList[self.screenCounter])  # clear
            self.rectUsed = None
            if zoomY > 0.0:
                self.draw(active_screen, self.objColor, (zoomX, zoomY))
            self.rectList[self.screenCounter] = self.rectUsed
            self.measureTime("draw image")
            # blend latest image to screen copy, adding it
            self.blend(active_screen, self.rectList[self.screenCounter], pygame.BLEND_ADD)
            # blend oldest image to screen copy, subtracting it
            oldest_screen = self.screenList[self.screenCounter - 1]
            self.blend(oldest_screen, self.rectList[self.screenCounter - 1], pygame.BLEND_SUB)
            self.measureTime("blend images")
            self.screenCounter -= 1
            if self.screenCounter < 0:
                self.screenCounter = self.screenCount - 1

        # copy finished data to visible screen and update the colors
        self.colorize()
        self.measureTime("colorize")

        self.zoomFrames += 1
        if self.zoomFrames > self.zoomFrameCount:
            self.zoomFrames = 0
            self.phase += 1

    def morphSetup(self):

        # set up morphNodes. Assumes first title is in titleNodes and second title in titleNodes2
        # also assumes each title surface can be morphed to a title2 surface and must have enough nodes to do that.
        # hole surfaces will be morphed to the line formed by the first and last node of the target surface..
        # this is quite specific to the title texts, will not work generally.
        self.titleNodes3 = self.titleNodes.copy()
        self.titleSurfaces3 = self.titleSurfaces.copy()
        self.morphNodes = np.zeros(np.shape(self.transNodes), dtype=int)
        d = 0
        for s in range(len(self.titleSurfaces)):
            src = self.titleSurfaces[s]
            if src[0] == 1 and d < len(self.titleSurfaces2):
                # surface drawn, morph it
                dst = self.titleSurfaces2[d]
                src_nodes = len(src)
                dst_nodes = len(dst)
                d_node = 1
                for s_node in range(1, src_nodes):
                    if src_nodes - s_node == dst_nodes - d_node or d_node == 1:
                        # forced to use current d_node
                        self.morphNodes[src[s_node]] = np.array(([src[s_node], dst[d_node]]))
                        d_node += 1
                    else:
                        # check if previous d_node was closer and if so, use it instead
                        if self.nodeDist(self.titleNodes[src[s_node]], self.titleNodes2[dst[d_node - 1]]) \
                                < self.nodeDist(self.titleNodes[src[s_node]], self.titleNodes2[dst[d_node]]):
                            # use previous d_node, do not move to next node
                            self.morphNodes[src[s_node]] = np.array(([src[s_node], dst[d_node - 1]]))
                        else:
                            # use current d_node
                            self.morphNodes[src[s_node]] = np.array(([src[s_node], dst[d_node]]))
                            if d_node < dst_nodes - 1:
                                d_node += 1
                d += 1  # next destination surface
            else:
                # morph to (closer of) first and last node of last destination only. Will be a dot or a line..
                dst = self.titleSurfaces2[d - 1]
                for s_node in range(1, len(src)):
                    if self.nodeDist(self.titleNodes[src[s_node]], self.titleNodes2[dst[1]]) \
                            < self.nodeDist(self.titleNodes[src[s_node]], self.titleNodes2[dst[-1]]):
                        self.morphNodes[src[s_node]] = np.array(([src[s_node], dst[1]]))
                    else:
                        self.morphNodes[src[s_node]] = np.array(([src[s_node], dst[-1]]))

    def nodeDist(self, node1, node2):

        return np.sqrt((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2)

    def morphTitle(self):

        # morph the titleText from one text to another.
        for i in range(self.nrImagesPerFrame):
            # multiplier m goes from 0 to 1 and morphs nodes from titletext to titletext2
            if self.morphFrames < self.morphFrameCount / 2:
                m = (float(self.morphFrames) + i / self.nrImagesPerFrame) / (self.morphFrameCount / 2)
            else:
                m = 1.0
            self.transNodes = (((1.0 - m) * self.titleNodes[self.morphNodes[:, 0]] + m * self.titleNodes2[self.morphNodes[:, 1]])
                               / self.zPos * self.zScale) + self.midScreen

            active_screen = self.screenList[self.screenCounter]
            active_screen.fill(self.backgroundColor, self.rectList[self.screenCounter])  # clear
            self.rectUsed = None
            self.draw(active_screen, self.objColor, (1.0, 1.0))
            self.rectList[self.screenCounter] = self.rectUsed
            self.measureTime("draw image")
            # blend latest image to screen copy, adding it
            self.blend(active_screen, self.rectList[self.screenCounter], pygame.BLEND_ADD)
            # blend oldest image to screen copy, subtracting it
            oldest_screen = self.screenList[self.screenCounter - 1]
            self.blend(oldest_screen, self.rectList[self.screenCounter - 1], pygame.BLEND_SUB)
            self.measureTime("blend images")
            self.screenCounter -= 1
            if self.screenCounter < 0:
                self.screenCounter = self.screenCount - 1

        # copy finished data to visible screen and update the colors
        self.colorize()
        self.measureTime("colorize")

        self.morphFrames += 1
        if self.morphFrames >= self.morphFrameCount:
            self.phase += 1

    def draw(self, screen, colorUsed, zoomFactor):

        for surface in self.titleSurfaces:
            if surface[0] == 1:
                color = colorUsed
            else:
                color = self.backgroundColor
            node_list = []
            for node in surface[1:]:
                node_list.append([(self.transNodes[node][0] - self.midScreen[0]) * zoomFactor[0] + self.midScreen[0],
                                  (self.transNodes[node][1] - self.midScreen[1]) * zoomFactor[1] + self.midScreen[1]])
            rect = self.drawPolygon(screen, color, node_list)
            if self.rectUsed is None:
                self.rectUsed = rect
            else:
                self.rectUsed = self.rectUsed.union(rect)

    def fadeTitle(self):

        # fade the striped title text to one color. Use squared alpha as it is additive.
        self.screenCopy2.set_alpha(int(255 * np.power(self.fadeFrames / self.fadeFrameCount, 2)))
        self.screen.blit(self.screenCopy2, self.rectUsed, self.rectUsed)
        self.fadeFrames += 1
        if self.fadeFrames > self.fadeFrameCount:
            self.phase += 1

    def fadeTitleOut(self):

        # fade the title text out by picking its color from the palette.
        color = self.palette[int(255 * max(0, self.barFrames) / self.barFrameCount), :]
        self.draw(self.screen, color, (1.0, 1.0))

    def blend(self, screen, rect, mode):

        # first test if rect is something to process
        if rect is not None:
            # blit a surface to the "complete image surface, additive (the newest) or subtract (the oldest) mode
            while self.screenCopy.get_locked():
                self.screenCopy.unlock()
            while screen.get_locked():
                screen.unlock()
            self.screenCopy.blit(screen, rect, rect, mode)

    def colorize(self):

        # first find the minimum area to process by looking at all image rects
        rect = pygame.Rect(self.midScreen[0], self.midScreen[1], 1, 1)
        for image_rect in self.rectList:
            if image_rect is not None:
                rect = rect.union(image_rect)
        x0 = rect[0]
        x1 = rect[0] + rect[2]
        y0 = rect[1]
        y1 = rect[1] + rect[3]
        self.rectUsed = rect

        # blit the screen copy to the visible screen. Screen copy has only red shades of the image.
        while self.screenCopy.get_locked():
            self.screenCopy.unlock()
        while self.screen.get_locked():
            self.screen.unlock()
        self.screen.blit(self.screenCopy, rect, rect)

        # use surfarray.pixels3d to process image as a numpy array
        rgb_array = pygame.surfarray.pixels3d(self.screen)
        # pick a new color from the 256 color palette for each pixel, based on its shade (0 to 255) of RED color
        self.maxcol1 = np.array([np.max(rgb_array[x0:x1, y0:y1, 0]), np.max(rgb_array[x0:x1, y0:y1, 1]), np.max(rgb_array[x0:x1, y0:y1, 2])])
        rgb_array[x0:x1, y0:y1, 0:3] = self.palette[rgb_array[x0:x1, y0:y1, 0]]
        self.maxcol2 = np.array([np.max(rgb_array[x0:x1, y0:y1, 0]), np.max(rgb_array[x0:x1, y0:y1, 1]), np.max(rgb_array[x0:x1, y0:y1, 2])])

    def drawPolygon(self, screen, color, node_list):

        # draws a filled antialiased polygon, returns its Rect (area covered)
        if len(node_list) > 2:  # a polygon needs at least 3 nodes
            # pygame.draw.aalines(screen, color, True, node_list)
            return pygame.draw.polygon(screen, color, node_list, 0)
        else:
            return pygame.Rect(0, 0, 0, 0)

    def plotInfo(self):
        """
        Add info on screen.
        This can obviously be skipped, just for information.
        """

        # release any locks on screen
        while self.screen.get_locked():
            self.screen.unlock()

        self.plotInfoBlit(10, 0 * 15 + 10, "frame  " + ("     " + str(self.frameCount))[-6:] + "  phase  " + ("  " + str(self.phase))[-3:])
        self.plotInfoBlit(10, 1 * 15 + 10, "angles " + ("     " + str(int(self.angles[0])))[-6:] + ("     " + str(int(self.angles[1])))[-6:]
                          + ("     " + str(int(self.angles[2])))[-6:])
        self.plotInfoBlit(10, 2 * 15 + 10, "rect   " + ("     " + str(int(self.rectUsed[0])))[-6:] + ("     " + str(int(self.rectUsed[1])))[-6:]
                          + ("     " + str(int(self.rectUsed[2])))[-6:] + ("     " + str(int(self.rectUsed[3])))[-6:])

        self.plotInfoBlit(10, 3 * 15 + 10, "cols 1 " + ("     " + str(int(self.maxcol1[0])))[-6:] + ("     " + str(int(self.maxcol1[1])))[-6:]
                          + ("     " + str(int(self.maxcol1[2])))[-6:])
        self.plotInfoBlit(10, 4 * 15 + 10, "cols 2 " + ("     " + str(int(self.maxcol2[0])))[-6:] + ("     " + str(int(self.maxcol2[1])))[-6:]
                          + ("     " + str(int(self.maxcol2[2])))[-6:])

        # add Frames Per Second - avg frame rate using the last ten frames
        self.plotInfoBlit(10, 5 * 15 + 10, "fps    " + ("     " + str(round(self.clock.get_fps(), 1)))[-6:])

        # add measured times as percentage of total
        tot_time = np.sum(self.timers)
        if tot_time > 0:
            for i in range(len(self.timer_names)):
                info_msg = (self.timer_names[i] + ' '*16)[:16] + (' '*10 + str(round(np.sum(self.timers[i, :]) * 100 / tot_time, 1)))[-7:]
                self.plotInfoBlit(10, (i + 6) * 15 + 10, info_msg)

    def plotInfoBlit(self, x, y, msg):

        f_screen = self.font.render(msg, False, [255, 255, 255])
        f_screen.set_colorkey(None)
        self.screen.blit(f_screen, (x, y))

    def toggleInfoDisplay(self):

        # switch between a windowed display and full screen
        if self.infoDisplay == True:
            self.infoDisplay = False
        else:
            self.infoDisplay = True

    def toggleFullScreen(self):

        # toggle between fulls creen and windowed mode
        pygame.display.toggle_fullscreen()

    def measureTime(self, timer_name):
        # add time elapsed from previous call to selected timer
        i = self.timer_names.index(timer_name)
        new_time = pygame.time.get_ticks()
        self.timers[i, self.timer_frame] += (new_time - self.millisecs)
        self.millisecs = new_time

    def nextTimeFrame(self):
        # move to next timer and clear data
        self.timer_frame += 1
        if self.timer_frame >= self.timer_avg_frames:
            self.timer_frame = 0
        self.timers[:, self.timer_frame] = 0

    def prepareLetters(self):

        self.letterAngleAdds = np.array((
            [ 1.7, 5.2,-3.0],
            [ 6.1,-2.3,-4.1],
            [-4.3, 3.2,-2.0]
            ), dtype=float)

        # generate palette. This 256 color palette will be used for object colors even if display is full color.
        for i in range(0, 85):
            self.palette[i, :] = np.array([int(i * 200 / 85), 0, 0], dtype=np.uint8)  # from black to red
        for i in range(85, 170):
            self.palette[i, :] = np.array([200, 0, int((i-85) * 220 / 85)], dtype=np.uint8)  # from red to purple
        for i in range(170, 256):
            self.palette[i, :] = np.array([int(max(200.0 - (i - 170.0) / (255.0 - 170.0) * 170.0, 0)), 0, 200], dtype=np.uint8)  # from purple to blue

        # create screen copies and init Rect list
        for i in range(self.screenCount):
            self.screenList.append(self.screen.copy())
            self.rectList.append(pygame.Rect(0, 0, 0, 0))

        # define letters as polygon coordinates.
        # NOTE later assumed that all letters have a minimum (X,Y) of (0,0).

        R = np.array([
         [  0,   0],
         [ 48,   0],
         [ 78,   8],
         [102,  26],
         [109,  47],
         [109,  77],
         [102,  96],
         [ 82, 117],
         [110, 200],
         [ 75, 200],
         [ 52, 120],
         [ 33, 120],
         [ 33, 200],
         [  0, 200]
        ], dtype=float)
        Rh = np.array([
         [ 33,  30],
         [ 55,  30],
         [ 70,  38],
         [ 77,  51],
         [ 77,  70],
         [ 70,  83],
         [ 55,  89],
         [ 33,  89]
        ], dtype=float)

        E = np.array([
         [  0,   0],
         [100,   0],
         [100,  33],
         [ 33,  33],
         [ 33,  83],
         [ 75,  83],
         [ 75, 116],
         [ 33, 116],
         [ 33, 167],
         [100, 167],
         [100, 200],
         [  0, 200]
        ], dtype=float)

        F = np.array([
         [  0,   0],
         [100,   0],
         [100,  33],
         [ 33,  33],
         [ 33,  83],
         [ 75,  83],
         [ 75, 116],
         [ 33, 116],
         [ 33, 200],
         [  0, 200]
        ], dtype=float)

        # some additional nodes for L to enable transforming to U
        L = np.array([
         [  0,   0],
         [ 33,   0],
         [ 33, 167],
         [ 50, 167],
         [100, 167],
         [100, 170],
         [100, 175],
         [100, 180],
         [100, 190],
         [100, 200],
         [ 33, 200],
         [  0, 200]
        ], dtype=float)

        C = np.array([
         [ 33,   0],
         [100,   0],
         [100,  33],
         [ 50,  33],
         [ 33,  50],
         [ 33, 151],
         [ 50, 167],
         [100, 167],
         [100, 200],
         [ 33, 200],
         [  0, 167],
         [  0,  33]
        ], dtype=float)

        # some additional nodes for T to enable transforming to N
        T = np.array([
         [  0,   0],
         [100,   0],
         [100,  33],
         [ 67,  33],
         [ 67, 100],
         [ 67, 200],
         [ 33, 200],
         [ 33, 100],
         [ 33,  33],
         [  0,  33]
        ], dtype=float)

        P = np.array([
         [  0,   0],
         [ 48,   0],
         [ 78,   8],
         [102,  26],
         [109,  47],
         [109,  77],
         [102,  96],
         [ 82, 117],
         [ 52, 120],
         [ 33, 120],
         [ 33, 200],
         [  0, 200]
        ], dtype=float)
        Ph = Rh

        S = np.array([
         [  0,   0],
         [100,   0],
         [100,  33],
         [ 33,  33],
         [ 33,  83],
         [100,  83],
         [100, 200],
         [  0, 200],
         [  0, 167],
         [ 67, 167],
         [ 67, 116],
         [  0, 116]
        ], dtype=float)

        O = np.array([
         [ 33,   0],
         [ 87,   0],
         [120,  33],
         [120, 167],
         [ 87, 200],
         [ 33, 200],
         [  0, 167],
         [  0,  33]
        ], dtype=float)
        Oh = np.array([
         [ 50,  33],
         [ 70,  33],
         [ 87,  50],
         [ 87, 150],
         [ 70, 167],
         [ 50, 167],
         [ 33, 150],
         [ 33,  50]
        ], dtype=float)

        Or = np.array([
         [ 33,   0],
         [ 87,   0],
         [120,  33],
         [120, 167],
         [ 87, 200],
         [ 70, 167],
         [ 87, 150],
         [ 87,  50],
         [ 70,  33],
         [ 50,  33]
        ], dtype=float)
        Ol = np.array([
         [ 33,   0],
         [ 50,  33],
         [ 33,  50],
         [ 33, 150],
         [ 50, 167],
         [ 70, 167],
         [ 87, 200],
         [ 33, 200],
         [  0, 167],
         [  0,  33]
        ], dtype=float)

        U = np.array([
         [  0,   0],
         [ 33,   0],
         [ 33, 150],
         [ 50, 167],
         [ 70, 167],
         [ 87, 150],
         [ 87,   0],
         [120,   0],
         [120, 167],
         [ 87, 200],
         [ 33, 200],
         [  0, 167]
        ], dtype=float)

        N = np.array([
         [  0,   0],
         [ 33,   0],
         [ 67,  95],
         [ 67,   0],
         [100,   0],
         [100, 200],
         [ 67, 200],
         [ 33, 105],
         [ 33, 200],
         [  0, 200]
        ], dtype=float)

        D = np.array([
         [  0,   0],
         [ 67,   0],
         [100,  33],
         [100, 167],
         [ 67, 200],
         [  0, 200]
        ], dtype=float)
        Dh = np.array([
         [ 33,  33],
         [ 50,  33],
         [ 67,  50],
         [ 67, 150],
         [ 50, 167],
         [ 33, 167]
        ], dtype=float)

        Dr = np.array([
         [ 67,   0],
         [100,  33],
         [100, 167],
         [ 67, 200],
         [ 50, 167],
         [ 67, 150],
         [ 67,  50],
         [ 50,  33]
        ], dtype=float)
        Dl = np.array([
         [  0,   0],
         [ 67,   0],
         [ 50,  33],
         [ 33,  33],
         [ 33, 167],
         [ 50, 167],
         [ 67, 200],
         [  0, 200]
        ], dtype=float)

        V = np.array([
         [  0,   0],
         [ 33,   0],
         [ 50,  95],
         [ 67,   0],
         [100,   0],
         [ 67, 200],
         [ 33, 200]
        ], dtype=float)

        I = np.array([
         [  0,   0],
         [ 33,   0],
         [ 33, 200],
         [  0, 200]
        ], dtype=float)

        # build a dictionary for easy fetching of a letter's data.
        # use a list to allow multiple components (when needing a hole in a letter, for instance). First element is 1 for filled and 0 for hole.
        # append in drawing order ie. the first item added will be processed first.
        a = []
        a.append([1, R])
        a.append([0, Rh])
        self.letters['R'] = a
        a = []
        a.append([1, E])
        self.letters['E'] = a
        a = []
        a.append([1, F])
        self.letters['F'] = a
        a = []
        a.append([1, L])
        self.letters['L'] = a
        a = []
        a.append([1, C])
        self.letters['C'] = a
        a = []
        a.append([1, T])
        self.letters['T'] = a
        a = []
        a.append([1, P])
        a.append([0, Ph])
        self.letters['P'] = a
        a = []
        a.append([1, S])
        self.letters['S'] = a
        a = []
        a.append([1, N])
        self.letters['N'] = a
        a = []
        a.append([1, Ol])
        a.append([1, Or])
        self.letters['O'] = a
        a = []
        a.append([1, U])
        self.letters['U'] = a
        a = []
        a.append([1, Dl])
        a.append([1, Dr])
        self.letters['D'] = a
        a = []
        a.append([1, V])
        self.letters['V'] = a
        a = []
        a.append([1, I])
        self.letters['I'] = a


if __name__ == '__main__':
    """
    Prepare screen, etc.
    """

    # set data directory
    chdir("C:/Users/Kalle/OneDrive/Asiakirjat/Python")

    # set screen size
    # first check available full screen modes
    pygame.display.init()
    # pick display mode from list or set a specific resolution
    # disp_modes = pygame.display.list_modes()
    # disp_size = disp_modes[9] # selecting display size from available list. Assuming the 9th element is nice...
    # disp_size = (800, 600)  # to force display size
    # disp_size = (1280, 800)  # to force display size
    disp_size = (1920, 1080)  # to force display size
    pygame.font.init()

    # initialize mixer
    pygame.mixer.init()
    music_file = "firepower.ogg"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/release.php?id=45536
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play()

    pygame.display.set_caption('Title Text')
    screen = pygame.display.set_mode(disp_size)
    TitleText(screen, 60).run()

    # exit; close everything
    pygame.quit()
    exit()
