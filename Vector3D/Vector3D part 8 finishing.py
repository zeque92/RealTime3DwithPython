# -*- coding: utf-8 -*-
import pygame
import numpy as np
import math
import os
from operator import itemgetter
from scipy.interpolate import interp1d
import copy
import xml.etree.ElementTree as et
from collections import Counter

key_to_function = {
    pygame.K_ESCAPE: (lambda x: x.terminate()),         # ESC key to quit
    pygame.K_SPACE:  (lambda x: x.pause()),             # SPACE to pause
    pygame.K_f:      (lambda x: x.toggleFullScreen()),  # f to switch between windowed and full screen display
    pygame.K_i:      (lambda x: x.toggleInfoDisplay())  # i to toggle info display on/off
    }

class VectorViewer:
    """
    Displays 3D vector objects on a Pygame screen.

    @author: kalle
    """

    def __init__(self, width, height, music_file, font_size):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width,height))
        self.screen_blends = self.screen.copy()
        self.screen_shadows = self.screen.copy()
        self.font = pygame.font.SysFont('CourierNew', font_size)  # initialize font and set font size
        self.music_file = music_file
        self.use_gfxdraw = False                        # pygame.gfxdraw is "experimental" and may be discontinued.
        self.fullScreen = False
        pygame.display.set_caption('VectorViewer')
        self.backgroundColor = (0,0,0)
        self.fadeSpeed = 5.0                            # controls start and end fade speed (bigger = faster)
        self.fade = 1.0
        self.shadowColor = (140,140,140)                # strength of shadow: 255 = black shadow, 0 = no shadow
        self.VectorAnglesList = []
        self.VectorMovementList = []
        self.viewerMovement = None
        self.VectorObjs = []
        self.VectorObjPrios = []
        self.VectorPos = VectorPosition()
        self.midScreen = np.array([width / 2, height / 2], dtype=float)
        self.zScale = width * 0.7                       # Scaling for z coordinates
        self.objMinZ = 100.0                            # minimum z coordinate for object visibility
        self.groundZ = 64000.0                          # for a ground object, maximum distance in Z
        self.groundZShade = 15.0                        # ground color strength (in percentage) at groundZ
        self.groundShadeNr = 16                         # number of ground elements with different shading
        self.groundBlendPrio = None                     # the prio after which ground belnding will be applied
        self.groundObject = None
        self.lightObject = None
        self.lightNode = np.array([-500.0, 600.0, -1300.0])     # presets only used if no lightObject
        self.lightPosition = np.array([-500.0, 600.0, -1300.0])     # presets only used if no lightObject
        self.target_fps = 60                            # affects movement speeds, sets maximum refresh rate
        self.running = True
        self.paused = True
        self.clock = pygame.time.Clock()
        # the following for checking performance only
        self.infoDisplay = False
        self.millisecs = 0
        self.timer_avg_frames = 180
        self.timer_names = []
        self.timers = np.zeros((1,1), dtype=int)
        self.timer_frame = 0
        self.start_timer = 0

    def setVectorPos(self, VectorPosObj):
        self.VectorPos = VectorPosObj

    def addVectorObj(self, VectorObj):
        self.VectorObjs.append(VectorObj)

    def addVectorAnglesList(self, VectorAngles):
        self.VectorAnglesList.append(VectorAngles)

    def addVectorMovementList(self, VectorMovement):
        self.VectorMovementList.append(VectorMovement)

    def run(self):
        """ Main loop. """

        for VectorObj in self.VectorObjs:
            VectorObj.initObject() # initialize objects

        if self.paused == True:
            pygame.time.wait(5000)
            VectorMovement.prevTime = pygame.time.get_ticks()
            self.paused = False

        # start music player
        pygame.mixer.music.load(self.music_file)
        pygame.mixer.music.play()

        # initialize timers
        self.start_timer = pygame.time.get_ticks()
        self.millisecs = self.start_timer

        # "all white" for surfaces used for blending
        self.screen_blends.fill((255,255,255))
        self.screen_shadows.fill((255,255,255))

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in key_to_function:
                        key_to_function[event.key](self)

            if self.paused == True:
                pygame.time.wait(100)

            else:
                # main components executed here
                self.movement()
                self.rotate()
                self.calculate()
                self.display()
                if self.infoDisplay == True:
                    self.plotInfo()

                # release any locks on screen
                while self.screen.get_locked():
                    self.screen.unlock()

                self.nextTimeFrame()

                # switch between currently showed and the next screen (prepared in "buffer")
                pygame.display.flip()
                self.measureTime("display flip")
                self.clock.tick(self.target_fps) # this keeps code running at max target_fps
                self.measureTime("wait")

        # exit; close display, stop music
        pygame.display.quit()
        pygame.mixer.quit()

    def movement(self):
        """
        Apply movement. Movements may change object positions and rotate angles.
        """

        for VectorMovement in self.VectorMovementList:
            # move forward in precalculated movement loop based on elapsed time
            # position is a float and the actual movement is interpolated between the two closest observations
            VectorMovement.moveLoop()
            loop_pos = int(VectorMovement.loopPos)
            mult = VectorMovement.loopPos - loop_pos

            # apply movement to angles
            if VectorMovement.angles is not None:
                VectorMovement.setRotateAngles()  # additional constant rotation
                VectorMovement.angles.setAngles(
                        (1.0 - mult) * VectorMovement.moveSeries[3:6, loop_pos]
                        + mult * VectorMovement.moveSeries[3:6, loop_pos + 1]
                        + VectorMovement.rotateAngles
                        )

            if VectorMovement == self.viewerMovement:
                # check for fade at start and end
                if loop_pos < 255 / self.fadeSpeed:
                    self.fade = (loop_pos * self.fadeSpeed) / 255
                elif loop_pos > VectorMovement.loopEnd - 255 / self.fadeSpeed:
                    self.fade = ((VectorMovement.loopEnd - loop_pos) * self.fadeSpeed) / 255
                else:
                    self.fade = 1.0
                # set viewer position, if viewer movement. Note Y coordinate must be reversed
                self.VectorPos.position = np.array([
                        (1.0 - mult) * VectorMovement.moveSeries[0, loop_pos] + mult * VectorMovement.moveSeries[0, loop_pos + 1],
                        -((1.0 - mult) * VectorMovement.moveSeries[1, loop_pos] + mult * VectorMovement.moveSeries[1, loop_pos + 1]),
                        (1.0 - mult) * VectorMovement.moveSeries[2, loop_pos] + mult * VectorMovement.moveSeries[2, loop_pos + 1],
                        1
                        ])
            else:
                # copy new position from movement to each object position (for other than viewer movement)
                for (node_num, VectorObj) in self.VectorPos.objects:
                    if VectorObj.movement == VectorMovement:
                        self.VectorPos.nodes[node_num, 0:3] = (1.0 - mult) * VectorMovement.moveSeries[0:3, loop_pos] + mult * VectorMovement.moveSeries[0:3, loop_pos + 1]

        self.measureTime("movements")

    def rotate(self):
        """
        Rotate all objects. First calculate rotation matrix.
        Then apply the relevant rotation matrix with object position to each VectorObject.
        """

        # calculate rotation matrices for all angle sets
        for VectorAngles in self.VectorAnglesList:
            VectorAngles.setRotationMatrix()
        self.measureTime("rotation matrix")

        # rotate object positions, copy those to objects.
        self.VectorPos.rotate()
        for (node_num, VectorObj) in self.VectorPos.objects:
            VectorObj.setPosition(self.VectorPos.rotatedNodes[node_num, :])
            VectorObj.setNodePosition(self.VectorPos.nodes[node_num, :])
            # also copy light  source object position to light position, if such object is defined.
            if self.lightObject == VectorObj:
                self.lightPosition = self.VectorPos.rotatedNodes[node_num, :]
                self.lightNode = self.VectorPos.nodes[node_num, 0:3]
        self.measureTime("positions")

        # rotate and flatten (transform) objects
        for VectorObj in self.VectorObjs:
            VectorObj.updateVisiblePos(self.objMinZ) # test for object position Z
            if VectorObj.visible == 1:
                VectorObj.rotate(self.viewerMovement.angles) # rotates objects in 3D
                VectorObj.updateVisibleNodes(self.objMinZ) # test for object minimum Z
                self.measureTime("rotate")
                if VectorObj.visible == 1:
                    VectorObj.transform(self.midScreen, self.zScale, self.objMinZ) # flattens to 2D
                    VectorObj.updateVisibleTrans(self.midScreen) # test for outside of screen
                    self.measureTime("transform 2D")

    def calculate(self):
        """
        Calculate shades and visibility.
        """

        if self.lightObject is not None:
            light_position = self.lightPosition
        else:
            light_position = self.lightPosition + self.VectorPos.position[0:3]

        for VectorObj in (vobj for vobj in self.VectorObjs if vobj.visible == 1 and vobj != self.groundObject):
            # calculate angles to Viewer (determines if surface is visible) and LightSource (for shading)
            VectorObj.updateSurfaceZPos()
            VectorObj.updateSurfaceCrossProductVector()
            self.measureTime("crossprodvec")
            VectorObj.updateSurfaceAngleToViewer()
            VectorObj.updateSurfaceAngleToLightSource(light_position)
            VectorObj.updateSurfaceColorShade()
            self.measureTime("object angles")
            if VectorObj.shadow == 1:
                # calculate shadow data. Note requires a viewer movement.
                VectorObj.updateShadow(self.viewerMovement.angles, self.lightNode, light_position, VectorObj.nodePosition[0:3], self.objMinZ, self.zScale, self.midScreen)
                self.measureTime("shadow calc")

    def display(self):
        """
        Draw the VectorObjs on the screen.
        """

        # clear screen. If ground object is used, do clearing there, not here.
        if self.groundObject is None:
            self.screen.fill(self.backgroundColor)
            self.measureTime("screen clear")

        # first sort VectorObjs so that the most distant is first, but prio classes separately.
        # prio classes allow e.g. to draw all roads first and only then other objects
        self.VectorObjs.sort(key=lambda VectorObject: (VectorObject.prio, VectorObject.position[2]), reverse=True)

        # draw prio by prio
        for prio_nr in self.VectorObjPrios:

            # draw shadows always first for all visible objects
            self.screen_shadows.lock()
            shadow_rect = None
            for VectorObj in (vobj for vobj in self.VectorObjs if vobj.visible == 1 and vobj.prio == prio_nr and vobj.shadow == 1):
                node_list = self.cropEdges(VectorObj.shadowTransNodes)
                rect = self.drawPolygon(self.screen_shadows, self.shadowColor, node_list, 0)
                if shadow_rect is None:
                    shadow_rect = rect
                else:
                    shadow_rect = shadow_rect.union(rect)
            while self.screen_shadows.get_locked():
                self.screen_shadows.unlock()
            if shadow_rect is not None:
                # blit the "multiplied" screen copy to add shadows on main screen
                # release any locks on screen
                while self.screen.get_locked():
                    self.screen.unlock()
                self.screen.blit(self.screen_shadows, shadow_rect, shadow_rect, pygame.BLEND_MULT)
                # clear the surface copy after use back to "all white"
                self.screen_shadows.fill((255,255,255), shadow_rect)
                self.measureTime("draw shadows")

            # lock screen for drawing operations
            self.screen.lock()

            # draw the actual objects
            for VectorObj in (vobj for vobj in self.VectorObjs if vobj.visible == 1 and vobj.prio == prio_nr):

                if VectorObj == self.groundObject:
                    # special object to add ground data
                    transNodes = VectorObj.groundData(self.midScreen, self.zScale, self.objMinZ, self.groundZ, self.groundShadeNr)
                    # draw ground. transNodes is of shape(groundShadeNr + 2, 4) and each row has two (left & right) X,Y coordinates
                    surface = VectorObj.surfaces[0]  # just one ground surface
                    # the first component is not part of the ground but used to clear the rest of the screen
                    node_list = self.cropEdges([transNodes[0,0:2], transNodes[0,2:4], transNodes[1,2:4], transNodes[1,0:2]], False, True) # X already cropped
                    use_color = ([round(self.fade * x, 0) for x in self.backgroundColor])
                    self.drawPolygon(self.screen, use_color, node_list, 0)
                    if self.groundBlendPrio is not None:
                        # fill the whole ground area with one color
                        node_list = self.cropEdges([transNodes[1,0:2], transNodes[1,2:4], transNodes[self.groundShadeNr + 1,2:4], transNodes[self.groundShadeNr + 1,0:2]], False, True) # X already cropped
                        use_color = ([round(self.fade * x, 0) for x in surface.color])
                        self.drawPolygon(self.screen, use_color, node_list, 0)
                        self.measureTime("ground + clear")
                        # draw the blends needed to "copy surface" for later blitting.
                        self.screen_blends.lock()
                        blend_rect = None
                        for i in range(self.groundShadeNr - 1):
                            node_list = self.cropEdges([transNodes[i+1,0:2], transNodes[i+1,2:4], transNodes[i+2,2:4], transNodes[i+2,0:2]], False, True) # X already cropped
                            # set color for blending as all R,G,B being between 0 (all blended away) and 255 (no effect)
                            use_color = int((2.55 * self.groundZShade * (self.groundShadeNr - 1 - i) + 255.0 * i) / (self.groundShadeNr - 1) + 0.5)
                            rect = self.drawPolygon(self.screen_blends, (use_color,use_color,use_color), node_list, 0)
                            if blend_rect is None:
                                blend_rect = rect
                            else:
                                blend_rect = blend_rect.union(rect)
                        while self.screen_blends.get_locked():
                            self.screen_blends.unlock()
                        self.measureTime("draw blends")
                    else:
                        # draw ground directly.
                        for i in range(self.groundShadeNr):
                            node_list = self.cropEdges([transNodes[i+1,0:2], transNodes[i+1,2:4], transNodes[i+2,2:4], transNodes[i+2,0:2]], False, True) # X already cropped
                            # set color for blending as a percentage of surface color
                            color_shade = self.fade * ((self.groundZShade * (self.groundShadeNr - 1 - i) + 100.0 * i) / (self.groundShadeNr - 1)) / 100.0
                            use_color = ([round(color_shade * x, 0) for x in surface.color])
                            self.drawPolygon(self.screen, use_color, node_list, surface.edgeWidth)
                        self.measureTime("ground + clear")

                else:
                    if VectorObj.isFlat == 1:
                        # flat objects have a single surface and a prebuilt list of transNodes
                        surface = VectorObj.surfaces[0]
                        node_list = self.cropEdges(VectorObj.transNodes)
                        self.drawPolygon(self.screen, surface.colorRGB(self.fade), node_list, surface.edgeWidth)
                    else:
                        # first sort object surfaces so that the most distant is first. For concave objects there should be no overlap, though.
                        VectorObj.sortSurfacesByZPos()
                        # then draw surface by surface. This is the most common case, the above are for special objects.
                        for surface in (surf for surf in VectorObj.surfaces if surf.visible == 1):
                            # build a list of transNodes for this surface
                            node_list = ([VectorObj.transNodes[node][:2] for node in surface.nodes])
                            self.drawPolygon(self.screen, surface.colorRGB(self.fade), node_list, surface.edgeWidth)
                    self.measureTime("draw surfaces")

            if prio_nr == self.groundBlendPrio and blend_rect is not None:
                # after groundBlendPrio, blit the "multiplied" screen copy to use a sliding color scale
                # release any locks on screen
                while self.screen.get_locked():
                    self.screen.unlock()
                self.screen.blit(self.screen_blends, blend_rect, blend_rect, pygame.BLEND_MULT)
                self.screen.lock()
                # clear the surface copy after use back to "all white"
                self.screen_blends.fill((255,255,255), blend_rect)
                self.measureTime("draw blends")

        # unlock screen
        self.screen.unlock()

    def plotInfo(self):
        """
        Add info on what is taking the time within the program.
        This can obviously be skipped, just for information.
        """

        # release any locks on screen
        while self.screen.get_locked():
            self.screen.unlock()

        # print viewer angles
        info_msg = (self.viewerMovement.angles.angName + " ang" + ' '*10)[:12]
        info_msg += (' '*10 + str(int(self.viewerMovement.angles.angles[0])))[-6:]
        info_msg += (' '*10 + str(int(self.viewerMovement.angles.angles[1])))[-6:]
        info_msg += (' '*10 + str(int(self.viewerMovement.angles.angles[2])))[-6:]
        f_screen = self.font.render(info_msg, False, [255,255,255])
        self.screen.blit(f_screen, (10,10))

        # print viewer position
        info_msg = ("viewer pos" + ' '*10)[:12]
        info_msg += (' '*10 + str(int(self.VectorPos.position[0])))[-6:]
        info_msg += (' '*10 + str(int(self.VectorPos.position[1])))[-6:]
        info_msg += (' '*10 + str(int(self.VectorPos.position[2])))[-6:]
        f_screen = self.font.render(info_msg, False, [255,255,255])
        self.screen.blit(f_screen, (10,30))

        # print movement loopPos
        info_msg = ("move pos" + ' '*10)[:12]
        for movement in self.VectorMovementList:
            info_msg += (' '*10 + str(int(movement.loopPos)))[-6:]
        f_screen = self.font.render(info_msg, False, [255,255,255])
        self.screen.blit(f_screen, (10,50))

        # add Frames Per Second
        fps = self.clock.get_fps() # avg frame rate using the last ten frames
        info_msg = ("fps" + ' '*16)[:16] + (' '*10 + str(round(fps, 1)))[-7:]
        self.screen.blit(self.font.render(info_msg, False, [255,255,255]), (10,90))

        # add measured times as percentage of total
        tot_time = np.sum(self.timers)
        if tot_time > 0:
            for i in range(len(self.timer_names)):
                info_msg = (self.timer_names[i] + ' '*16)[:16] + (' '*10 + str(round(np.sum(self.timers[i,:]) * 100 / tot_time, 1)))[-7:]
                self.screen.blit(self.font.render(info_msg, False, [255,255,255]), (10, 110 + i * 20))

        self.measureTime("plot info")

    def cropEdges(self, node_list, cropX = True, cropY = True):
        # crop to screen size. "Auto crop" does not seem to work if points very far outside.
        # takes list of nodes (X,Y) in drawing order as input.
        # returns list of nodes (X,Y) cropped to screen edges.
        # crop both X, Y, if cropX and cropY = True; X: i=0, Y: i=1
        if len(node_list) > 2:
            for i in range(2):
                if (i == 0 and cropX == True) or (i == 1 and cropY == True):
                    crop_nodes = [] # empty list
                    prev_node = node_list[-1]
                    for node in node_list:
                        diff_node = node - prev_node # surface side vector
                        # start cropping from prev_node direction, as order must stay the same
                        if node[i] >= 0 and prev_node[i] < 0:
                            # line crosses 0, so add a "crop point". Start from previous node and add difference stopping to 0
                            crop_nodes.append(prev_node + diff_node * ((0 - prev_node[i]) / diff_node[i]))
                        if node[i] <= self.midScreen[i] * 2 and prev_node[i] > self.midScreen[i] * 2:
                            # line crosses screen maximum, so add a "crop point". Start from previous node and add difference stopping to midScreen[i] * 2
                            crop_nodes.append(prev_node + diff_node * ((self.midScreen[i] * 2 - prev_node[i]) / diff_node[i]))
                        # then crop current node
                        if node[i] < 0 and prev_node[i] >= 0:
                            # line crosses 0, so add a "crop point". Start from previous node and add difference stopping to 0
                            crop_nodes.append(prev_node + diff_node * ((0 - prev_node[i]) / diff_node[i]))
                        if node[i] > self.midScreen[i] * 2 and prev_node[i] <= self.midScreen[i] * 2:
                            # line crosses screen maximum, so add a "crop point". Start from previous node and add difference stopping to midScreen[i] * 2
                            crop_nodes.append(prev_node + diff_node * ((self.midScreen[i] * 2 - prev_node[i]) / diff_node[i]))
                        # always add current node, if it is on screen
                        if node[i] >= 0 and node[i] <= self.midScreen[i] * 2:
                            crop_nodes.append(node)
                        prev_node = node
                    # for next i, copy results. Quit loop if no nodes to look at
                    node_list = crop_nodes
                    if len(node_list) < 3:
                        break
            # convert to integers
            node_list = [(int(x[0] + 0.5), int(x[1] + 0.5)) for x in node_list]
        return node_list

    def drawPolygon(self, screen, color, node_list, edgeWidth):

        # draws a filled antialiased polygon, returns its Rect (area covered)
        if len(node_list) > 2: # a polygon needs at least 3 nodes
            if self.use_gfxdraw  == True:
                pygame.gfxdraw.aapolygon(screen, node_list, color)
                return pygame.gfxdraw.filled_polygon(screen, node_list, color)
            else:
                pygame.draw.aalines(screen, color, True, node_list)
                return pygame.draw.polygon(screen, color, node_list, edgeWidth)
        else:
            return pygame.Rect(0,0,0,0)

    def terminate(self):

        self.running = False

    def pause(self):

        if self.paused == True:
            self.paused = False
        else:
            self.paused = True

    def toggleFullScreen(self):

        # switch between a windowed display and full screen
        if self.fullScreen == True:
            self.fullScreen = False
            self.screen = pygame.display.set_mode((self.width,self.height))
        else:
            self.fullScreen = True
            self.screen_blends = pygame.display.set_mode((self.width,self.height), pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)

        self.screen_blends = self.screen.copy()
        self.screen_shadows = self.screen.copy()
        self.screen_blends.fill((255,255,255))
        self.screen_shadows.fill((255,255,255))

    def toggleInfoDisplay(self):

        # switch between a windowed display and full screen
        if self.infoDisplay == True:
            self.infoDisplay = False
        else:
            self.infoDisplay = True

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

class VectorObject:

    """
    Position is the object's coordinates.
    Nodes are the predefined, static definition of object "corner points", around object position anchor point (0,0,0).
    RotatedNodes are the Nodes rotated by the given Angles and moved to Position.
    TransNodes are the RotatedNodes transformed from 3D to 2D (X.Y) screen coordinates.

    @author: kalle
    """
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0, 1.0])      # position
        self.nodePosition = np.array([0.0, 0.0, 0.0, 1.0])  # position before rotation
        self.angles = VectorAngles()
        self.movement = VectorMovement()
        self.nodes = np.zeros((0, 4))                       # nodes will have unrotated X,Y,Z coordinates plus a column of ones for position handling
        self.objRotatedNodes = np.zeros((0, 3))             # objRotatedNodes will have X,Y,Z coordinates after object rotation in place (object angles)
        self.rotatedNodes = np.zeros((0, 3))                # rotatedNodes will have X,Y,Z coordinates after rotation ("final 3D coordinates")
        self.transNodes = np.zeros((0, 2))                  # transNodes will have X,Y coordinates
        self.nodeNum = 0                                    # number of object nodes. nodes, rotatedNodes and transNodes may contain also shadow nodes
        self.shadow = 0
        self.shadowNodeList = []                            # list of shadow nodes, in drawing order
        self.shadowRotatedNodes = np.zeros((0, 3))          # rotatedNodes will have X,Y,Z coordinates
        self.shadowTransNodes = np.zeros((0, 2))            # transNodes will have X,Y coordinates
        self.surfaces = []
        self.visible = 1
        self.isFlat = 0
        self.prio = 0                                       # priority order when drawn. Highest prio will be drawn first
        self.objName = ""
        self.minShade = 0.2                                 # shade (% of color) to use when surface is parallel to light source

    def initObject(self):
        self.updateSurfaceZPos()
        self.updateSurfaceCrossProductVector()
        self.updateSurfaceCrossProductLen()

    def setPosition(self, position):
        # move object by giving it a rotated position.
        self.position = position

    def setNodePosition(self, nodePosition):
        # move object by giving it an unrotated position.
        self.nodePosition = nodePosition

    def setFlat(self):
        # set isFlat
        self.isFlat = 1

    def addNodes(self, node_array):
        # add nodes (all at once); add a column of ones for using position in transform
        self.nodes = np.hstack((node_array, np.ones((len(node_array), 1))))
        self.rotatedNodes = node_array # initialize rotatedNodes with nodes (no added ones required)

    def addSurfaces(self, idnum, color, edgeWidth, showBack, backColor, node_list):
        # add a Surface, defining its properties
        surface = VectorObjectSurface()
        surface.idnum = idnum
        surface.color = color
        surface.edgeWidth = edgeWidth
        surface.showBack = showBack
        surface.backColor = backColor
        surface.nodes = node_list
        self.surfaces.append(surface)

    def updateVisiblePos(self, objMinZ):
        # check if object is visible. If any of node Z coordinates are too close to viewer, set to 0, unless is flat
        if self.isFlat == 0 and self.position[2] < objMinZ:
            self.visible = 0
        else:
            self.visible = 1

    def updateVisibleNodes(self, objMinZ):
        # check if object is visible. If any of node Z coordinates are too close to viewer, set to 0, unless is flat
        if self.isFlat == 0:
            if min(self.rotatedNodes[:, 2]) < objMinZ:
                self.visible = 0
            else:
                self.visible = 1
        else:
            # for flat objects, check if the whole object is behind the viewing point (minZ)
            if max(self.rotatedNodes[:, 2]) < objMinZ:
                self.visible = 0
            else:
                self.visible = 1

    def updateVisibleTrans(self, midScreen):
        # check if object is visible. If not enough nodes or all X or Y coordinates are outside of screen, set to 0
        if \
            np.shape(self.transNodes)[0] < 3 \
            or max(self.transNodes[:, 0]) < 0 \
            or min(self.transNodes[:, 0]) > midScreen[0] * 2 \
            or max(self.transNodes[:, 1]) < 0 \
            or min(self.transNodes[:, 1]) > midScreen[1] * 2:
            self.visible = 0
        else:
            self.visible = 1

    def updateSurfaceZPos(self):
        # calculate average Z position for each surface using rotatedNodes
        for surface in self.surfaces:
            zpos = sum([self.rotatedNodes[node, 2] for node in surface.nodes]) / len(surface.nodes)
            surface.setZPos(zpos)
            surface.setVisible(1) # set all surfaces to "visible"

    def updateSurfaceCrossProductVector(self):
        # calculate cross product vector for each surface using rotatedNodes
        # always use vectors (1, 0) and (1, 2) (numbers representing nodes)
        # numpy "cross" was terribly slow, calculating directly as below takes about 10 % of the time.
        for surface in self.surfaces:
            vec_A = self.rotatedNodes[surface.nodes[2], 0:3] - self.rotatedNodes[surface.nodes[1], 0:3]
            vec_B = self.rotatedNodes[surface.nodes[0], 0:3] - self.rotatedNodes[surface.nodes[1], 0:3]
            vec_Cross =  ([
                vec_B[1] * vec_A[2] - vec_B[2] * vec_A[1],
                vec_B[2] * vec_A[0] - vec_B[0] * vec_A[2],
                vec_B[0] * vec_A[1] - vec_B[1] * vec_A[0]
                ])
            surface.setCrossProductVector(vec_Cross)

    def updateSurfaceCrossProductLen(self):
        # calculate cross product vector length for each surface.
        # this is constant and done only at init stage.
        for surface in self.surfaces:
            surface.setCrossProductLen()

    def updateSurfaceAngleToViewer(self):
        # calculate acute angle between surface plane and Viewer
        # surface plane cross product vector and Viewer vector both from node 1.
        for surface in (surf for surf in self.surfaces if surf.visible == 1):
            vec_Viewer = self.rotatedNodes[surface.nodes[1], 0:3]
            surface.setAngleToViewer(vec_Viewer)
            if surface.angleToViewer > 0 or surface.showBack == 1:
                surface.setVisible(1)
            else:
                surface.setVisible(0)

    def updateSurfaceAngleToLightSource(self, lightPosition):
        # calculate acute angle between surface plane and light source, similar to above for Viewer.
        # this is used to define shading and shadows; needed for visible surfaces using shading AND all surfaces, if shadow to be drawn.
        for surface in (surf for surf in self.surfaces if (surf.visible == 1 and self.minShade < 1.0) or self.shadow == 1):
            surface.setLightSourceVector(self.rotatedNodes[surface.nodes[1], 0:3] - lightPosition)
            surface.setAngleToLightSource()

    def updateSurfaceColorShade(self):
        # calculate shade for surface.
        for surface in (surf for surf in self.surfaces if surf.visible == 1):
            surface.setColorShade(self.minShade)
            if surface.showBack == 1: surface.setBackColorShade(self.minShade)

    def sortSurfacesByZPos(self):
        # sorts surfaces by Z position so that the most distant comes first in list
        self.surfaces.sort(key=lambda VectorObjectSurface: VectorObjectSurface.zpos, reverse=True)

    def updateShadow(self, viewerAngles, lightNode, light_position, obj_pos, objMinZ, zScale, midScreen):
        """
        Update shadowTransNodes (a list of nodes which define the shadow of the object).
        This routine assumes objects are "whole", that they do not have any holes - such objects should be built as several parts.
        A more resilient way would be to calculate and draw a shadow for each surface separately, but leading to extra calculations and draws.
        Update nodes by calculating and adding the required shadow nodes to the array.
        Note for objects standing on the ground, the nodes where Y=0 are already there and need no calculation / adding.
        """
        shadow_edges = [] # list of object edges that belong to surfaces facing the lightsource, i.e. producing a shadow.
        for surface in (surf for surf in self.surfaces if surf.angleToLightSource > 0 or surf.showBack == 1):
            # add all egdes using the smaller node nr first in each edge node pair.
            shadow_edges.extend([((min(surface.nodes[i], surface.nodes[i-1]), max(surface.nodes[i], surface.nodes[i-1]))) for i in range(len(surface.nodes))])
        # get all edges which are in the list only once - these should define the outer perimeter. Inner edges are used twice.
        use_edges = [list(c[0]) for c in (d for d in list(Counter(shadow_edges).items()) if d[1] == 1)]
        # these edges should form a continuous line (the perimeter of the shadowed object). Prepare a list of nodes required:
        node_list = []
        node_list.append(use_edges[0][0]) # first node from first edge
        node_list.append(use_edges[0][1]) # second node from first edge
        prev_edge = use_edges[0]
        for i in range(len(use_edges)):
            if node_list[-1] != node_list[0]:
                for edge in use_edges:
                    if edge != prev_edge: # do not check the edge itself
                        if edge[0] == node_list[-1]:
                            node_list.append(edge[1]) # this edge begins with previous node, add its end
                            prev_edge = edge
                            break
                        if edge[1] == node_list[-1]:
                            node_list.append(edge[0]) # this edge ends with previous node, add its beginning
                            prev_edge = edge
                            break
            else:
                break # full circle reached
        node_list = node_list[0:len(node_list) - 1] # full circle - drop the last node (is equal to first)

        # then project these nodes on the ground i.e. Y = 0. If necessary. Add the required shadowRotatedNodes.
        self.shadowRotatedNodes = np.zeros((0, 4))
        for node_num in range(len(node_list)):
            node = node_list[node_num]
            # check that node is not already on the ground level or too high compared to light source
            if obj_pos[1] + self.objRotatedNodes[node, 1] > 3 and obj_pos[1] + self.objRotatedNodes[node, 1] < (lightNode[1] - 3):
                # node must be projected. Add a shadow node and replace current node in node_list with it.
                node_list[node_num] = self.nodeNum + np.shape(self.shadowRotatedNodes)[0]
                diff_node = (obj_pos + self.objRotatedNodes[node,:]) - lightNode # vector from lightNode to this node. In this space, Y=0 is ground
                # the projection multiplier if based on above diff_node where Y=0 is ground, but it can be applied directly to rotatedNodes as well, skipping one rotation step.
                diff_rotated_node = self.rotatedNodes[node,:] - light_position
                self.shadowRotatedNodes = np.vstack((self.shadowRotatedNodes, np.hstack((light_position + diff_rotated_node * (lightNode[1] / -diff_node[1]), 1))))

        # flatten rotated shadow nodes and build a list of shadowTransNodes. shadowTransNodes has all shadow nodes.
        flat_nodes = np.zeros((0, 2))
        if node_list[-1] < self.nodeNum:
            prev_node = self.rotatedNodes[node_list[-1], 0:3] # previous node XYZ coordinates
        else:
            prev_node = self.shadowRotatedNodes[node_list[-1] - self.nodeNum, 0:3]
        for node_num in range(len(node_list)):
            if node_list[node_num] < self.nodeNum:
                node = self.rotatedNodes[node_list[node_num], 0:3] # current node XYZ coordinates
            else:
                node = self.shadowRotatedNodes[node_list[node_num] - self.nodeNum, 0:3]
            diff_node = node - prev_node # surface side vector
            # if both Z coordinates behind the viewer: do not draw at all, do not add a transNode
            if (node[2] < objMinZ and prev_node[2] >= objMinZ) or (node[2] >= objMinZ and prev_node[2] < objMinZ):
                # line crosses objMinZ, so add a "crop point". Start from previous node and add difference stopping to objMinZ. Flatten to 2D.
                f_node = prev_node + diff_node * ((objMinZ - prev_node[2]) / diff_node[2])
                flat_nodes = np.vstack((flat_nodes, (-f_node[0:2] * zScale) / f_node[2:3] + midScreen))
            if node[2] >= objMinZ:
                # add current node, if it is visible
                if node_list[node_num] < self.nodeNum:
                    flat_nodes = np.vstack((flat_nodes, self.transNodes[node_list[node_num], 0:2])) # object node on the ground, already calculated
                else:
                    flat_nodes = np.vstack((flat_nodes, (-node[0:2] * zScale) / node[2:3] + midScreen)) # shadow node, needs flattening 3D to 2D
            prev_node = node
        self.shadowTransNodes = flat_nodes


    def rotate(self, viewerAngles):
        """
        Apply a rotation defined by a given rotation matrix.
        For objects with their own angles / rotation matrix, apply those first and store results in objRotatedNodes.
        Could be done in one step but objRotatedNodes needed for shadow calculations.
        """

        if self.angles != viewerAngles:
            # rotate object with its own angles "in place" ie. with synthetic zero vector as position
            matrix = np.vstack((self.angles.rotationMatrix, np.zeros((1,3))))
            self.objRotatedNodes = np.dot(self.nodes, matrix)
        else:
            # no own angles, just copy nodes then
            self.objRotatedNodes = self.nodes[:,0:3]
        # then rotate with viewer angles. Add position to rotation matrix to enable both rotation and position change at once
        matrix = np.vstack((viewerAngles.rotationMatrix, self.position[0:3]))
        self.rotatedNodes = np.dot(np.hstack((self.objRotatedNodes, np.ones((np.shape(self.objRotatedNodes)[0], 1)))), matrix)

    def transform(self, midScreen, zScale, objMinZ):
        """
        Flatten from 3D to 2D and add screen center.
        First crop flat objects by Z coordinate so that they can be drawn even if some Z coordinates are behind the viewer.
        """
        if self.isFlat == 1:
            # for flat objects, build a list of transNodes for the surface by first cropping the necessary surface sides to minZ
            for surface in self.surfaces:
                surface.setVisible(1) # set all surfaces to "visible"
                flat_nodes = np.zeros((0, 3))
                for node_num in range(len(surface.nodes)):
                    node = self.rotatedNodes[surface.nodes[node_num], 0:3] # current node XYZ coordinates
                    prev_node = self.rotatedNodes[surface.nodes[node_num - 1], 0:3] # previous node XYZ coordinates
                    diff_node = node - prev_node # surface side vector
                    # if both Z coordinates behind the viewer: do not draw at all, do not add a transNode
                    if (node[2] < objMinZ and prev_node[2] >= objMinZ) or (node[2] >= objMinZ and prev_node[2] < objMinZ):
                        # line crosses objMinZ, so add a "crop point". Start from previous node and add difference stopping to objMinZ
                        flat_nodes = np.vstack((flat_nodes, prev_node + diff_node * ((objMinZ - prev_node[2]) / diff_node[2])))
                    if node[2] >= objMinZ:
                        # add current node, if it is visible
                        flat_nodes = np.vstack((flat_nodes, node))
                # apply perspective using Z coordinates and add midScreen to center on screen to get to transNodes
                self.transNodes = (-flat_nodes[:, 0:2] * zScale) / (flat_nodes[:, 2:3]) + midScreen
        else:
            # apply perspective using Z coordinates and add midScreen to center on screen to get to transNodes.
            # for normal objects, some of the transNodes will not be required, but possibly figuring out which are and processing them
            #   individually could take more time than this.
            self.transNodes = (self.rotatedNodes[:, 0:2] * zScale) / (-self.rotatedNodes[:, 2:3]) + midScreen

    def groundData(self, midScreen, zScale, objMinZ, groundZ, groundShadeNr):
        """
        Calculate ground data for on a ground object.
        Assumes the ground object "covers the ground" reasonably and isFlat = 1, has 4 nodes, and the perimeter is concave.
        Ground settings are defined in VectorViewer.
        Returns an array of shape(groundShadeNr + 2, 4) where each row has X,Y of left edge and X.Y of right edge starting from most distant.
        The first array is used to clear the screen above the ground (so in effect this covers the whole screen).
        """
        # find the most distant node
        maxZ = max(self.rotatedNodes[:, 2])
        for nodenum in range(len(self.nodes)):
            if self.rotatedNodes[nodenum, 2] == maxZ:
                node = self.rotatedNodes[nodenum, :]
                break
        prev_node = self.rotatedNodes[nodenum - 1, :]
        if nodenum == len(self.nodes) - 1:
            next_node = self.rotatedNodes[0, :]
        else:
            next_node = self.rotatedNodes[nodenum + 1, :]

        # get a straight line where Z (ie, distance from viewer) is constant. Start with the mid of farthest of the two lines.
        # then find the point with matching Z coordinate on the other line.
        # special cases: next_node or prev_node as far as node.
        if node[2] == prev_node[2]:
            mid1_node = node
            mid2_node = prev_node
        else:
            if node[2] == next_node[2]:
                mid1_node = node
                mid2_node = next_node
            else:
                if next_node[2] > prev_node[2]:
                    mid1_node = (next_node + node) / 2
                    mid2_node = node + (prev_node - node) * (mid1_node[2] - node[2]) / (prev_node[2] - node[2])
                else:
                    mid1_node = (prev_node + node) / 2
                    mid2_node = node + (next_node - node) * (mid1_node[2] - node[2]) / (next_node[2] - node[2])
        if mid1_node[1] < mid2_node[1]:
            # make sure mid1_node X < mid2_node X
            mid1_node, mid2_node = mid2_node, mid1_node
        # adjust Z
        mid1_node = mid1_node * groundZ / mid1_node[2]
        mid2_node = mid2_node * groundZ / mid2_node[2]
        # finalize a square around object position
        mid2_node_back = self.position[0:3] + (self.position[0:3] - mid1_node) # from front left (mid1) to back right (mid2_back)
        mid1_node_back = self.position[0:3] + (self.position[0:3] - mid2_node) # from front right (mid2) to back left (mid1_back)

        # then generate arrays with necessary node data and transNode data
        left_nodes = np.zeros((groundShadeNr + 1, 3), dtype=float)
        right_nodes = np.zeros((groundShadeNr + 1, 3), dtype=float)
        # multipliers will span ground component span between groundZ/2 (furthest) and objMinZ
        mult = (mid1_node[2] / 2 - objMinZ) / ((mid1_node[2] - mid1_node_back[2]) / 2)
        # the most distant component (at groundZ). Most distant component will be very large (half of total)
        left_nodes[0,:] = mid1_node
        right_nodes[0,:] = mid2_node

        # other components from groundZ/2 to objMinZ
        for i in range(groundShadeNr):
            mult_i =  mult * math.sqrt((i+1) / groundShadeNr)
            left_nodes[i+1,:] = (mid1_node * (1.0 - mult_i) + mid1_node_back * mult_i) / 2
            right_nodes[i+1,:] = (mid2_node * (1.0 - mult_i) + mid2_node_back * mult_i) / 2
        left_transNodes = (-left_nodes[:, 0:2] * zScale) / (left_nodes[:, 2:3]) + midScreen
        right_transNodes = (-right_nodes[:, 0:2] * zScale) / (right_nodes[:, 2:3]) + midScreen

        # crop these nodes to screen X edges
        diff_transNodes = right_transNodes - left_transNodes
        mult_nodes = right_transNodes[:, 0] / diff_transNodes[:, 0]
        left_transNodes = right_transNodes - np.multiply(np.transpose(np.vstack((mult_nodes, mult_nodes))), diff_transNodes)
        diff_transNodes = right_transNodes - left_transNodes
        mult_nodes = (midScreen[0] * 2) / diff_transNodes[:,0]
        right_transNodes = left_transNodes + np.multiply(np.transpose(np.vstack((mult_nodes, mult_nodes))), diff_transNodes)

        # the first component is "the top of the sky".
        if left_transNodes[0,1] < left_transNodes[1,1]:
            # "normal ground", add a node to the top of the screen
            if left_transNodes[0,1] < 0:
                # if ground already covers the whole screen, use the top node
                left_skynode = left_transNodes[0,:]
            else:
                left_skynode = np.array([0, 0])
        else:
            # inverted ground ie. going upside down, add a node to the bottom of the screen
            if left_transNodes[0,1] > midScreen[1] * 2:
                # if ground already covers the whole screen, use the top node
                left_skynode = left_transNodes[0,:]
            else:
                left_skynode = np.array([0, midScreen[1] * 2])
        if right_transNodes[0,1] < right_transNodes[1,1]:
            # "normal ground", add a node to the top of the screen
            if right_transNodes[0,1] < 0:
                # if ground already covers the whole screen, use the top node
                right_skynode = right_transNodes[0,:]
            else:
                right_skynode = np.array([midScreen[0] * 2, 0])
        else:
            # inverted ground ie. going upside down, add a node to the bottom of the screen
            if right_transNodes[0,1] > midScreen[1] * 2:
                # if ground already covers the whole screen, use the top node
                right_skynode = right_transNodes[0,:]
            else:
                right_skynode = midScreen * 2
        # add the first component and build an array of all the transnodes
        transNodes = np.vstack((np.hstack((left_skynode, right_skynode)), np.hstack((left_transNodes, right_transNodes))))

        return(transNodes)

class VectorObjectSurface:

    """
    Surfaces for a VectorObject.

    @author: kalle
    """
    def __init__(self):
        self.idnum = 0
        # properties set when defining the object
        self.nodes = []
        self.color = (0,0,0)
        self.edgeWidth = 0           # if 0, fills surface. Otherwise a wireframe (edges only), with edgeWidth thickness.
        # the following are calculated during program execution
        self.zpos = 0.0
        self.crossProductVector = np.zeros((0,3))
        self.crossProductLen = 0.0   # precalculated length of the cross product vector - this is constant
        self.lightSourceVector = np.zeros((0,3))
        self.angleToViewer = 0.0
        self.angleToLightSource = 0.0
        self.visible = 1
        self.colorShade = 1.0        # Shade of color; 0 = black, 1 = full color

    def setZPos(self, zpos):
        self.zpos = zpos

    def setVisible(self, visible):
        self.visible = visible

    def setCrossProductVector(self, crossProductVector):
        self.crossProductVector = crossProductVector

    def setLightSourceVector(self, lightSourceVector):
        self.lightSourceVector = lightSourceVector

    def setCrossProductLen(self):
        self.crossProductLen = self.vectorLen(self.crossProductVector)

    def setAngleToViewer(self, vec_Viewer):
        if self.crossProductLen > 0 and vec_Viewer.any() != 0:
            # instead of true angle calculation using asin and vector lengths, a simple np.vdot is sufficient to find the sign (which defines if surface is visible)
            # self.angleToViewer = math.asin(np.dot(self.crossProductVector, vec_Viewer) / (self.crossProductLen * np.linalg.norm(vec_Viewer)))
            self.angleToViewer = np.dot(self.crossProductVector, vec_Viewer)

    def setAngleToLightSource(self):
        if self.crossProductLen > 0 and self.lightSourceVector.any() != 0:
            # instead of true angle calculation using asin and vector lengths, a simple np.vdot is sufficient to find the sign (which defines if surface is shadowed)
            # self.angleToLightSource = math.asin(np.dot(self.crossProductVector, self.lightSourceVector) / (self.crossProductLen * np.linalg.norm(self.lightSourceVector))) / (np.pi / 2)
            self.angleToLightSource = np.dot(self.crossProductVector, self.lightSourceVector)

    def setColorShade(self, minShade):
        if self.angleToLightSource <= 0:
            self.colorShade = minShade
        else:
            self.colorShade = minShade + (1.0 - minShade) * math.asin(self.angleToLightSource / (self.crossProductLen * self.vectorLen(self.lightSourceVector))) / (np.pi / 2)

    def setBackColorShade(self, minShade):
        if self.angleToLightSource >= 0:
            self.backColorShade = minShade
        else:
            self.backColorShade = minShade + (1.0 - minShade) * -math.asin(self.angleToLightSource / (self.crossProductLen * self.vectorLen(self.lightSourceVector))) / (np.pi / 2)

    def colorRGB(self, fade):
        if self.angleToViewer > 0:
            use_color = ([round(fade * self.colorShade * x, 0) for x in self.color]) # apply shading
        else:
            use_color = ([round(fade * self.backColorShade * x, 0) for x in self.backColor]) # apply shading to backside color
        return use_color

    def vectorLen(self, vector):
        # equivalent to numpy.linalg.norm for a 3D real vector, but much faster. math.sqrt is faster than numpy.sqrt.
        return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])

class VectorAngles:

    """
    Angles for rotating vector objects. For efficiency, one set of angles can be used for many objects.
    Angles are defined for axes X (horizontal), Y (vertical), Z ("distance") in degrees (360).

    @author: kalle
    """
    def __init__(self):
        self.angles = np.array([0.0, 0.0, 0.0])
        self.angleScale = (2.0 * np.pi) / 360.0 # to scale degrees.
        self.angName = ""
        self.rotationMatrix = np.zeros((3,3))
        self.rotateAngles = np.array([0.0, 0.0, 0.0])
        self.rotate = np.array([0.0, 0.0, 0.0])

    def setAngles(self, angles):
        # Set rotation angles to fixed values.
        self.angles = angles

    def setRotateAngles(self):
        self.rotateAngles += self.rotate
        for i in range(3):
            if self.rotateAngles[i] >= 360: self.rotateAngles[i] -= 360
            if self.rotateAngles[i] < 0: self.rotateAngles[i] += 360

    def setRotationMatrix(self):
        # Set matrix for rotation using angles.

        (sx, sy, sz) = np.sin((self.angles + self.rotateAngles) * self.angleScale)
        (cx, cy, cz) = np.cos((self.angles + self.rotateAngles) * self.angleScale)

        # build a matrix for X, Y, Z rotation (in that order, see Wikipedia: Euler angles).
        self.rotationMatrix = np.array([[cy * cz               , -cy * sz              , sy      ],
                                        [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
                                        [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]])

class VectorMovement:

    """
    Movement handling. This class directs position and/or angles rotation.

    @author: kalle
    """
    def __init__(self):
        self.moveName = ""
        self.angles = VectorAngles()
        self.rotate = np.array([0.0, 0.0, 0.0])
        self.rotateAngles = np.array([0.0, 0.0, 0.0])
        self.timeSeries = np.zeros((7,0))
        self.moveSeries = np.zeros((6,0))
        self.objects = []                               # list of objects controlled by this movement (typically just one)
        self.loopStart = 0
        self.loopEnd = -1                               # will be set >= 0 by starting procedure
        self.loopPos = 0.0                              # float as position can be a fraction
        self.prevTime = pygame.time.get_ticks()

    def setAngles(self, angles):
        self.angles = angles

    def setRotateAngles(self):
        self.rotateAngles += self.rotate
        for i in range(3):
            if self.rotateAngles[i] >= 360: self.rotateAngles[i] -= 360
            if self.rotateAngles[i] < 0: self.rotateAngles[i] += 360

    def moveLoop(self):
        new_time = pygame.time.get_ticks()
        self.loopPos += (new_time - self.prevTime) / 20.0 # increase by 50 / second (1000 ms / 20)
        self.prevTime = new_time
        if self.loopPos >= self.loopEnd:
            self.loopPos = self.loopPos - int(self.loopPos) + self.loopStart # back to loop start, preserving any fractions

    def addTimeSeries(self, time_array):
        self.timeSeries = time_array

    def addMoveSeries(self, move_array):
        self.moveSeries = move_array

    def addObject(self, VectorObject):
        self.objects.append(VectorObject)

class VectorPosition:

    """
    A vector object defining the positions of other objects in its nodes (see VectorObject).

    @author: kalle
    """
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0, 1.0])
        self.angles = VectorAngles()
        self.nodes = np.zeros((0, 4))                   # nodes will have unrotated X,Y,Z coordinates plus a column of ones for position handling
        self.rotatedNodes = np.zeros((0, 3))            # rotatedNodes will have X,Y,Z coordinates
        self.objects = []                               # connects each node to a respective VectorObject
        self.objName = ""

    def addNodes(self, node_array):
        # add nodes (all at once); add a column of ones for using position in transform
        self.nodes = np.hstack((node_array, np.ones((len(node_array), 1))))
        self.rotatedNodes = node_array # initialize with nodes

    def addObjects(self, object_list):
        self.objects = object_list

    def rotate(self):
        # apply a rotation defined by a given rotation matrix.
        matrix = np.vstack((self.angles.rotationMatrix, np.zeros((1, 3))))
        # apply rotation and position matrix to nodes
        self.rotatedNodes = np.dot(self.nodes + self.position, matrix)

if __name__ == '__main__':
    """
    Prepare screen, read objects etc. from file.
    """

    # set screen size
    # first check available full screen modes
    pygame.display.init()
    # pick disp0lay mode from list or set a specific resolution
    # disp_modes = pygame.display.list_modes(0, pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    # disp_size = disp_modes[4] # selecting display size from available list. Assuming the 5th element is nice...
    disp_size = (1280, 800)
    if disp_size[1] >= 800:
        font_size = 18
    else:
        font_size = 14

    # initialize font & mixer
    pygame.font.init()
    pygame.mixer.init()

    music_file = "sinking2.ogg"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/release.php?id=45536
    vv = VectorViewer(disp_size[0], disp_size[1], music_file, font_size)

    # set up timers
    vv.timer_names.append("movements")
    vv.timer_names.append("positions")
    vv.timer_names.append("rotation matrix")
    vv.timer_names.append("rotate")
    vv.timer_names.append("transform 2D")
    vv.timer_names.append("crossprodvec")
    vv.timer_names.append("object angles")
    vv.timer_names.append("shadow calc")
    vv.timer_names.append("screen clear")
    vv.timer_names.append("ground + clear")
    vv.timer_names.append("draw blends")
    vv.timer_names.append("draw shadows")
    vv.timer_names.append("draw surfaces")
    vv.timer_names.append("plot info")
    vv.timer_names.append("display flip")
    vv.timer_names.append("wait")
    vv.timers = np.zeros((len(vv.timer_names), vv.timer_avg_frames), dtype=int)

    # read data file defining angles, movements and objects
    vecdata = et.parse("vectordata.xml")

    root = vecdata.getroot()

    for angles in root.iter('vectorangles'):
        ang = VectorAngles()
        ang.angName = angles.get('name')
        ang.angles[0] = float(angles.findtext("angleX", default="0"))
        ang.angles[1] = float(angles.findtext("angleY", default="0"))
        ang.angles[2] = float(angles.findtext("angleZ", default="0"))
        vv.addVectorAnglesList(ang)

    for movements in root.iter('movement'):
        mov = VectorMovement()
        mov.moveName = movements.get('name')

        viewer = None
        viewer = movements.get('viewer')

        angleName = movements.findtext("anglesref", None)
        for angles in vv.VectorAnglesList:
            if angles.angName == angleName:
                mov.angles = angles
                if viewer == "1":
                    angles.angleScale = -angles.angleScale # change viewer angles to negative by changing angleScale to negative
                break
        mov.rotate[0] = float(movements.findtext("rotateX", default="0"))
        mov.rotate[1] = float(movements.findtext("rotateY", default="0"))
        mov.rotate[2] = float(movements.findtext("rotateZ", default="0"))

        # build movement timeseries array; [0] is time in seconds (if target_fps = movSpeed), [1:4] are positions X,Y,Z, [4:7] are angles X,Y,Z
        movSpeed = float(movements.findtext("speed", default="1"))
        for timedata in movements.iter('timeseries'):
            mov_times = np.zeros((7, 0))
            mov_point = np.zeros((7, 1))
            for stepdata in timedata.iter('movementstep'):
                #steptime = float(stepdata.get("time"))
                mov_point[0, 0] = float(stepdata.get("time"))
                mov_point[1, 0] = float(stepdata.findtext("movepositionX", default=str(mov_point[1, 0])))
                mov_point[2, 0] = float(stepdata.findtext("movepositionY", default=str(mov_point[2, 0])))
                mov_point[3, 0] = float(stepdata.findtext("movepositionZ", default=str(mov_point[3, 0])))
                mov_point[4, 0] = float(stepdata.findtext("moveangleX", default="0"))
                mov_point[5, 0] = float(stepdata.findtext("moveangleY", default="0"))
                mov_point[6, 0] = float(stepdata.findtext("moveangleZ", default="0"))
                mov_times = np.hstack((mov_times, mov_point))
            # sort by time (first row), just in case ordering is wrong
            mov_times = mov_times[:,mov_times[0,:].argsort()]
            if viewer == "1":
                # for viewer, invert sign of Y coordinate
                mov_times[1,:] = -1 * mov_times[1,:]
            mov.addTimeSeries(mov_times)
            # out of the time series, build frame by frame movement using interpolation.
            fx = interp1d(mov_times[0,:], mov_times[1:7,:], kind='quadratic')
            # data needed for the whole range of time series. Use speed = "time" per second.
            num_times = int((mov_times[0,-1] - mov_times[0,0]) * vv.target_fps / movSpeed)
            time_frames = np.linspace(mov_times[0,0], mov_times[0,-1], num=num_times, endpoint=True)
            mov_moves = fx(time_frames)

            mov_angleforward = movements.findtext("angleforward", default="off")
            if mov_angleforward != "off":
                # angleforward means that object angles should be such that it is rotated according to movement vector, always facing towards movement
                # NOTE: Tested for Y only! Not well defined for XYZ anyway.
                mov_diffs = mov_moves[0:3,1:] - mov_moves[0:3,:-1]
                mov_diffs = np.hstack((mov_diffs, mov_diffs[:,-1:])) # copy second last diff to last
                if mov_angleforward.find("X") >= 0:
                    mov_moves[3,:] += np.arctan2(mov_diffs[1,:], mov_diffs[2,:]) * 180 / np.pi
                if mov_angleforward.find("Y") >= 0:
                    mov_moves[4,:] += np.arctan2(mov_diffs[0,:], -mov_diffs[2,:]) * 180 / np.pi
                if mov_angleforward.find("Z") >= 0:
                    mov_moves[5,:] += np.arctan2(mov_diffs[0,:], mov_diffs[1,:]) * 180 / np.pi

            mov.addMoveSeries(mov_moves)
            break # only one time series accepted

        mov.loopEnd = 0 # preset
        for loopdata in movements.iter('loop'):
            mov.loopStart = int(float(loopdata.findtext("loopstart", default="0")) * vv.target_fps / movSpeed)
            mov.loopEnd = int(float(loopdata.findtext("loopend", default="0")) * vv.target_fps / movSpeed)
            mov.loopPos = int(float(loopdata.findtext("looppos", default="0")) * vv.target_fps / movSpeed)
            break # only one loop accepted

        vv.addVectorMovementList(mov)
        # define this (last added) movement as the viewer movement if so specified.
        if viewer == "1":
            vv.viewerMovement = vv.VectorMovementList[-1]

    for vecobjs in root.iter('vectorobject'):
        vobj = VectorObject()
        vobj.objName = vecobjs.get('name')
        vobj.prio = int(vecobjs.findtext("prio", default="0"))
        # check if object is a light source. Will be set later.
        lightsource = None
        lightsource = vecobjs.get('lightsource')
        # check if object is the ground. Will be set later.
        ground = None
        ground = vecobjs.get('ground')
        if ground == "1":
            vv.groundBlendPrio = int(vecobjs.findtext("blendprio", default=None))

        if vecobjs.findtext('shadow', 'OFF').upper() == "ON":
            vobj.shadow = 1
        else:
            vobj.shadow = 0

        # check if object is a copy of another, previously defined object
        copyfrom = None
        copyfrom = vecobjs.get('copyfrom')
        is_copy = False
        if copyfrom is not None:
            for VectorObj in vv.VectorObjs:
                if VectorObj.objName == str(copyfrom):
                    # copy data from another object
                    vobj = copy.deepcopy(VectorObj) # copy properties
                    vobj.angles = VectorObj.angles  # copy reference to angles, does not seem to work with deepcopy()
                    vobj.movement = VectorObj.movement  # copy reference to movement, does not seem to work with deepcopy()
                    is_copy = True
                    break

        # set position and references to angles and movement
        for posdata in vecobjs.iter('position'):
            vobj.position[0] = float(posdata.findtext("positionX", default="0"))
            vobj.position[1] = float(posdata.findtext("positionY", default="0"))
            vobj.position[2] = -float(posdata.findtext("positionZ", default="0")) # inverted for easier coordinates definition
        angleName = vecobjs.findtext("anglesref", None)
        for angles in vv.VectorAnglesList:
            if angles.angName == angleName:
                vobj.angles = angles
                break
        vobj.movement = None
        movementName = vecobjs.findtext("movementref", None)
        for movement in vv.VectorMovementList:
            if movement.moveName == movementName:
                vobj.movement = movement
                break

        # get (or set) some default values. Not needed for copied objects.
        if is_copy == True:
            def_minshade = str(vobj.minShade)
        else:
            def_minshade = "0.3"
            def_color = (def_colorR, def_colorG, def_colorB) = (128, 128, 128)
            # if not a copied object, read default values for some surface properties
            for def_colordata in vecobjs.iter('defcolor'):
                def_colorR = def_colordata.findtext("defcolorR", default="128")
                def_colorG = def_colordata.findtext("defcolorG", default="128")
                def_colorB = def_colordata.findtext("defcolorB", default="128")
                def_color = (int(def_colorR), int(def_colorG), int(def_colorB))
                break
            def_backColor = (def_backColorR, def_backColorG, def_backColorB) = def_color
            for def_backcolordata in vecobjs.iter('defbackcolor'):
                def_backColorR = def_backcolordata.findtext("defbackcolorR", default=str(def_backColorR))
                def_backColorG = def_backcolordata.findtext("defbackcolorG", default=str(def_backColorG))
                def_backColorB = def_backcolordata.findtext("defbackcolorB", default=str(def_backColorB))
                def_backColor = (int(def_backColorR), int(def_backColorG), int(def_backColorB))
                break
            def_edgeWidth = vecobjs.findtext("defedgewidth", default="0")
            def_showBack = vecobjs.findtext("defshowback", default="0")
        vobj.minShade = float(vecobjs.findtext("minshade", default=def_minshade))

        if is_copy == False:
            # add nodes ie. "points" or "corners". No changes allowed for copied objects.
            for nodedata in vecobjs.iter('nodelist'):
                vobj.nodeNum = int(nodedata.get("numnodes"))
                vobj_nodes = np.zeros((vobj.nodeNum, 3))
                for node in nodedata.iter('node'):
                    node_num = int(node.get("ID"))
                    vobj_nodes[node_num, 0] = float(node.findtext("nodeX", default="0"))
                    vobj_nodes[node_num, 1] = float(node.findtext("nodeY", default="0"))
                    vobj_nodes[node_num, 2] = -float(node.findtext("nodeZ", default="0")) # inverted for easier coordinates definition
                vobj.addNodes(vobj_nodes)
                break # only one nodelist accepted

        # check for initangles ie. initial rotation. Default is none which requires no action.
        for angledata in vecobjs.iter('initangles'):
            angleXadd = float(angledata.findtext("angleXadd", default="0"))
            angleYadd = float(angledata.findtext("angleYadd", default="0"))
            angleZadd = float(angledata.findtext("angleZadd", default="0"))
            if angleXadd != 0 or angleYadd != 0 or angleZadd != 0:
                storeangles = copy.copy(vobj.angles.angles) # store temporarily
                storeposition = copy.copy(vobj.position) # store temporarily
                vobj.angles.angles = np.array([angleXadd, angleYadd, angleZadd]) # use the requested initial rotation
                vobj.position = np.array([0.0, 0.0, 0.0, 1.0]) # set position temporarily to zero for initial rotation
                vobj.angles.setRotationMatrix()
                vobj.rotate(vobj.angles)
                vobj.nodes = np.hstack((vobj.rotatedNodes, np.ones((np.shape(vobj.rotatedNodes)[0], 1)))) # overwrite original nodes with rotated (to init angles) nodes
                vobj.angles.angles = storeangles
                vobj.position = storeposition

        # add surfaces ie. 2D flat polygons.
        for surfacedata in vecobjs.iter('surfacelist'):
            for surf in surfacedata.iter('surface'):
                idnum = int(surf.get("ID"))

                if is_copy == True:
                    for surfobj in vobj.surfaces:
                        if surfobj.idnum == idnum:
                            break
                    def_color = surfobj.color
                    def_backColor = surfobj.backColor
                    def_edgeWidth = str(surfobj.edgeWidth)
                    def_showBack = str(surfobj.showBack)

                color = def_color
                for colordata in surf.iter('color'):
                    colorR = int(colordata.findtext("colorR", default=def_colorR))
                    colorG = int(colordata.findtext("colorG", default=def_colorG))
                    colorB = int(colordata.findtext("colorB", default=def_colorB))
                    color = (colorR, colorG, colorB)
                    break
                backColor = def_backColor
                for backColordata in surf.iter('backColor'):
                    backColorR = int(backColordata.findtext("backColorR", default=def_backColorR))
                    backColorG = int(backColordata.findtext("backColorG", default=def_backColorG))
                    backColorB = int(backColordata.findtext("backColorB", default=def_backColorB))
                    backColor = (backColorR, backColorG, backColorB)
                    break
                edgeWidth = int(surf.findtext("edgewidth", default=def_edgeWidth))
                showBack = int(surf.findtext("showback", default=def_showBack))
                if is_copy == False:
                    # create a list of nodes. No changes allowed for copied objects.
                    node_list = []
                    for nodelist in surf.iter('nodelist'):
                        for node in nodelist.iter('node'):
                            node_order = int(node.get("order"))
                            node_refID = int(node.get("refID"))
                            node_list.append((node_order, node_refID))
                    node_list.sort(key=itemgetter(0)) # sort nodes by node_order
                    node_list = list(zip(*node_list))[1] # pick just the node references
                    vobj.addSurfaces(idnum, color, edgeWidth, showBack, backColor, node_list)
                else:
                    vobj.modifySurfaces(idnum, color, edgeWidth, showBack, backColor)

        # check if is a flat object (one surface only)
        if len(vobj.surfaces) == 1:
            vobj.isFlat = 1
        # add object prio to prio list, if not there
        if not vobj.prio in vv.VectorObjPrios:
            vv.VectorObjPrios.append(vobj.prio)
        # add the object
        vv.addVectorObj(vobj)

        # add it to a movement's list of objects, if so specified
        if vv.VectorObjs[-1].movement is not None:
            vv.VectorObjs[-1].movement.addObject(vv.VectorObjs[-1])
        # define this (last added) object as light source or ground if so specified
        if lightsource is not None:
            vv.lightObject = vv.VectorObjs[-1]
        # define this (last added) object as ground if so specified
        if ground is not None:
            vv.groundObject = vv.VectorObjs[-1]

    # sort prio list
    vv.VectorObjPrios.sort(reverse=True)

    # define a vector position object holding all the positions of other objects in its nodes
    for vecobjs in root.iter('positionobject'):
        vobj = VectorPosition()
        vobj.objName = vecobjs.get('name')
        for posdata in vecobjs.iter('position'):
            vobj.position[0] = float(posdata.findtext("positionX", default="0"))
            vobj.position[1] = float(posdata.findtext("positionY", default="0"))
            vobj.position[2] = float(posdata.findtext("positionZ", default="1500"))
        angleName = vecobjs.findtext("anglesref", None)
        for angles in vv.VectorAnglesList:
            if angles.angName == angleName:
                vobj.angles = angles
                break
        # get nodes from existing objects
        vobj_nodes = np.zeros((0, 3))
        object_num = 0
        object_list = []
        for VectorObj in vv.VectorObjs:
            vobj_nodes = np.vstack((vobj_nodes, VectorObj.position[0:3]))
            object_list.append((object_num, VectorObj))  # reference to object position data row and respective object
            object_num += 1
        vobj.addNodes(vobj_nodes)
        vobj.addObjects(object_list)
        # set the object
        vv.setVectorPos(vobj)
        break

    # run the main program
    vv.run()
