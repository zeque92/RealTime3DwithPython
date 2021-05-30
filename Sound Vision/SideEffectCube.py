# -*- coding: utf-8 -*-
import pygame
import numpy as np
import math
from sys import exit


class SideEffectCube:
    """
    Displays a cube with animated effects on its surfaces.

    @author: kalle
    """

    def __init__(self, screen, target_fps, run_seconds):
        self.run_seconds = run_seconds
        self.screen = screen
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.fullScreen = False
        self.backgroundColor = (0, 0, 0)
        self.angleScale = (2.0 * np.pi) / 360.0         # to scale degrees.
        self.VectorObjs = []
        self.starAngle = 0.0
        self.starAngleAdd = 5.0
        self.starNodes = np.zeros((10, 2), dtype=float)  # star nodes have angle and distance from center data
        self.starField = np.zeros((50, 3), dtype=float)  # X and Y of each star + X speed
        self.starFieldColor = (170, 170, 222)
        self.zoomFrameCount = 180
        self.zoomDirection = 1
        self.zoomFrames = 0
        self.totalFrameCount = 60 * 45                  # 45 seconds
        self.totalFrames = 0
        self.midScreen = np.array([int(self.width / 2), int(self.height / 2)], dtype=float)
        self.zScale = self.height                       # Scaling for z coordinates
        self.startZPos = 10000.0
        self.endZPos = 500.0
        self.zPos = self.startZPos
        self.target_fps = target_fps                    # affects movement speeds
        self.running = True
        self.stop = False
        self.paused = False
        self.clock = pygame.time.Clock()
        self.start_timer = pygame.time.get_ticks()

        self.prepareCube()
        for VectorObj in self.VectorObjs:
            VectorObj.initObject()  # initialize objects
        self.screen.fill(self.backgroundColor)

    def run(self):
        """ Main loop. """

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.zoomDirection = -1
                        self.totalFrames = self.totalFrameCount - self.zoomFrames
                        self.stop = True
                    if event.key == pygame.K_f:
                        self.toggleFullScreen()
                    if event.key == pygame.K_SPACE:
                        self.pause()
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                                          self.__class__.__name__ + '.jpg')
                    # if event.key == pygame.K_i:
                    #     self.toggleInfoDisplay()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # left button: exit
                        self.zoomDirection = -1
                        self.totalFrames = self.totalFrameCount - self.zoomFrames

            if self.paused:
                pygame.time.wait(100)
            else:
                # main components executed here
                self.zoomCube()
                self.moveStarField()
                self.rotateStar()
                self.calcAndDisplay()

                # release any locks on screen
                while self.screen.get_locked():
                    self.screen.unlock()

                # switch between currently showed and the next screen (prepared in "buffer")
                pygame.display.flip()
                self.clock.tick(self.target_fps)  # this keeps code running at max target_fps

            if pygame.time.get_ticks() > self.start_timer + 1000 * self.run_seconds:
                self.zoomDirection = -1
                self.totalFrames = self.totalFrameCount - self.zoomFrames

        return self.stop

    def zoomCube(self):
        """
        Zoom the cube in and out by changing zPos.
        """

        self.zoomFrames += self.zoomDirection
        self.zPos = self.endZPos + self.startZPos * (self.zoomFrameCount - self.zoomFrames) / self.zoomFrameCount
        if self.zoomFrames == self.zoomFrameCount:
            self.zoomDirection = 0  # stop zoom when at the front
        self.totalFrames += 1
        if self.totalFrames == int(self.totalFrameCount / 2):
            self.VectorObjs[0].rotateSpeed *= 2  # double the rotation speed at half way
        if self.totalFrames == self.totalFrameCount - self.zoomFrameCount:
            self.zoomDirection = -1  # inititate zooming out
        if self.totalFrames >= self.totalFrameCount:
            self.running = False

    def calcAndDisplay(self):
        """
        Rotate and calculate objects and draw the cube on screen.
        """

        # lock screen for pixel operations
        self.screen.lock()

        # clear screen.
        self.screen.fill(self.backgroundColor)

        # unlock screen
        self.screen.unlock()

        # first rotate and transform both cubes.
        for VectorObj in self.VectorObjs:
            VectorObj.increaseAngles()
            VectorObj.setRotationMatrix()
            VectorObj.rotate()
            VectorObj.transform(self.zScale, self.zPos, self.midScreen)
            VectorObj.updateSurfaceCrossProductVector()
            VectorObj.updateSurfaceAngleToViewer(self.zPos)

        # draw cube 0 and, depending on surface, add projected additional details
        VectorObj = self.VectorObjs[0]
        # draw surface by surface, if surface is visible.
        for surface in (surf for surf in VectorObj.surfaces if surf.visible == 1):
            # build a list of transNodes for this surface
            node_list = ([VectorObj.transNodes[node][:2] for node in surface.nodes])
            pygame.draw.aalines(self.screen, surface.color, True, node_list)
            pygame.draw.polygon(self.screen, surface.color, node_list)
            if int(surface.idnum / 2) == 0:
                # "left and right" sides: add starfield
                self.addStarField(node_list)
            if int(surface.idnum / 2) == 1:
                # "top and bottom" surfaces: add star
                self.addStar(node_list)
            if int(surface.idnum / 2) == 2:
                # "front and back" surfaces: add cube
                self.addCube(node_list, surface.idnum)

    def addStarField(self, node_list):

        # add a star field to the surface just drawn. Each star has X and Y coordinates in range [0,1]; simply scale them within the surface.
        # use less stars when zoomed far away.

        XY = self.scaleToSurface(self.starField[0:int(np.shape(self.starField)[0] * (self.zoomFrames + 10) / (self.zoomFrameCount + 10)), 0:2], node_list)
        if self.zPos > self.endZPos + 1000.0:
            for i in range(int(np.shape(XY)[0] * ((self.startZPos + 1000.0 - self.zPos) / (self.startZPos - self.endZPos)))):
                self.screen.set_at((int(XY[i][0]), int(XY[i][1])), self.starFieldColor)
        else:
            c_size = 1.0 + (self.endZPos + 1000.0 - self.zPos) / 1000.0
            for i in range(np.shape(XY)[0]):
                pygame.draw.circle(self.screen, self.starFieldColor, (int(XY[i][0]), int(XY[i][1])), c_size)

    def moveStarField(self):

        # add "X-add" to X coordinate
        self.starField[:, 0] = self.starField[:, 0] + self.starField[:, 2]
        # check if still within cube boundaries
        for i in range(np.shape(self.starField)[0]):
            if self.starField[i, 0] > 0.99:
                self.starField[i, 0] -= 0.99

    def addStar(self, node_list):

        # add a star to the surface just drawn. node_list has surface node X,Y pair screen coordinates in clockwise order
        node_array = np.hstack((np.sin((self.starAngle + self.starNodes[:, 0:1]) * self.angleScale) * self.starNodes[:, 1:2] / 2 + 0.5,
                                np.cos((self.starAngle + self.starNodes[:, 0:1]) * self.angleScale) * self.starNodes[:, 1:2] / 2 + 0.5))
        XY = self.scaleToSurface(node_array, node_list)

        # draw the star, one triangle at a time
        node_list = (XY[0], XY[5], XY[9])
        pygame.draw.aalines(self.screen, self.starFieldColor, True, node_list)
        pygame.draw.polygon(self.screen, self.starFieldColor, node_list)
        for i in (range(4)):
            node_list = (XY[i + 1], XY[i + 6], XY[i + 5])
            pygame.draw.aalines(self.screen, self.starFieldColor, True, node_list)
            pygame.draw.polygon(self.screen, self.starFieldColor, node_list)

    def rotateStar(self):

        # rotate star by increasing the angle
        self.starAngle += self.starAngleAdd
        if self.starAngle >= 360:
            self.starAngle -= 360

    def addCube(self, node_list, idnum):

        # add object 1 i.e. the other cube by projecting it to the surface by simple scaling.

        CubeObj = self.VectorObjs[1]
        # recalculate transNodes. First they must be scaled to range [0,1]
        new_transNodes = (CubeObj.transNodes - self.midScreen) * (0.0025 * self.zPos / self.zScale) + np.array([0.5, 0.5])
        new_transNodes = self.scaleToSurface(new_transNodes, node_list)
        for surface in (surf for surf in CubeObj.surfaces if surf.visible == 1):
            # build a list of transNodes for this surface
            node_list = ([new_transNodes[node][:2] for node in surface.nodes])
            # change color depending on which master cube surface drawn
            color = surface.color
            if idnum % 2 == 0:
                color = (color[1], color[0], color[1])
            pygame.draw.aalines(self.screen, color, True, node_list)
            pygame.draw.polygon(self.screen, color, node_list)

    def scaleToSurface(self, XY_array, node_list):

        # node_list has surface four node X,Y pair screen coordinates in clockwise order
        # XY_array is a numpy array of shape (n,2) with values between 0 and 1, scaling within the surface X and Y.

        n_array = np.shape(XY_array)[0]
        # build a node array for "up" and "down" nodes "left" to "right" from node_list
        node_up =   np.array([node_list[0][0], node_list[0][1], node_list[1][0], node_list[1][1]])
        node_down = np.array([node_list[3][0], node_list[3][1], node_list[2][0], node_list[2][1]])
        # using Y coordinate, find the line (X1,Y1 - X2,Y2) wher the star is located
        Y_line = np.ones((n_array, 1)) * node_up - XY_array[:, 1:2] * node_up + XY_array[:, 1:2] * node_down

        # scale the results using star X coordinate
        m = XY_array[:, 0:1] * Y_line
        # combine to get end result
        XY = np.hstack((Y_line[:, 0:1] - m[:, 0:1] + m[:, 2:3],  Y_line[:, 1:2] - m[:, 1:2] + m[:, 3:4]))

        # perhaps a more readable scaling below, commented out of use
        # X1 = (1.0 - Y_array) * node_list[0][0] + Y_array * node_list[3][0]
        # Y1 = (1.0 - Y_array) * node_list[0][1] + Y_array * node_list[3][1]
        # X2 = (1.0 - Y_array) * node_list[1][0] + Y_array * node_list[2][0]
        # Y2 = (1.0 - Y_array) * node_list[1][1] + Y_array * node_list[2][1]
        # # using star X coordinate, set its screen X,Y on the line
        # X = (1.0 - X_array) * stars_X1 + X_array * stars_X2
        # Y = (1.0 - X_array) * stars_Y1 + X_array * stars_Y2

        return XY

    def toggleFullScreen(self):

        # toggle between fulls creen and windowed mode
        pygame.display.toggle_fullscreen()

    def pause(self):

        if self.paused:
            self.paused = False
        else:
            self.paused = True

    def addVectorObj(self, VectorObj):
        self.VectorObjs.append(VectorObj)

    def prepareCube(self):

        # set up two simple cubes. The second one will then be projected on the side of the other.

        vobj = VectorObject()
        vobj2 = VectorObject()
        # first add nodes, i.e. the corners of the cube, in (X, Y, Z) coordinates
        node_array = np.array([
                [ 100.0, 100.0, 100.0],
                [ 100.0, 100.0,-100.0],
                [ 100.0,-100.0,-100.0],
                [ 100.0,-100.0, 100.0],
                [-100.0, 100.0, 100.0],
                [-100.0, 100.0,-100.0],
                [-100.0,-100.0,-100.0],
                [-100.0,-100.0, 100.0]
                ])
        vobj.addNodes(node_array)
        vobj2.addNodes(node_array)
        # then define surfaces: ID, color, nodes
        node_list = [0, 3, 2, 1]  # node_list defines the four nodes forming a cube surface, in clockwise order
        vobj.addSurfaces( 0, ( 51, 51,137), node_list)
        vobj2.addSurfaces(0, ( 51,137, 51), node_list)
        node_list = [4, 5, 6, 7]
        vobj.addSurfaces( 1, ( 51, 51,137), node_list)
        vobj2.addSurfaces(1, ( 51,137, 51), node_list)
        node_list = [0, 1, 5, 4]
        vobj.addSurfaces( 2, ( 34, 34,102), node_list)
        vobj2.addSurfaces(2, ( 34,102, 34), node_list)
        node_list = [3, 7, 6, 2]
        vobj.addSurfaces( 3, ( 34, 34,102), node_list)
        vobj2.addSurfaces(3, ( 34,102, 34), node_list)
        node_list = [0, 4, 7, 3]
        vobj.addSurfaces( 4, ( 69, 69,170), node_list)
        vobj2.addSurfaces(4, ( 69,170, 69), node_list)
        node_list = [1, 2, 6, 5]
        vobj.addSurfaces( 5, ( 69, 69,170), node_list)
        vobj2.addSurfaces(5, ( 69,170, 69), node_list)

        vobj.rotateSpeed = np.array([2.1, -1.3, 1.55])
        vobj2.rotateSpeed = np.array([2.75, 1.4, -0.95])

        # add the object
        self.addVectorObj(vobj)
        self.addVectorObj(vobj2)

        # set up random star field with X,Y ranging from 0 to 1 and "X speed" from 0.01 to 0.04
        self.starField = np.random.rand(int(self.height / 8), 3) * np.array([1, 1, 0.03]) + np.array([0, 0, 0.01])

        # set up star nodes
        for i in range(5):
            self.starNodes[i, :] = np.array([i * 360 / 5, 0.9])
            self.starNodes[i + 5, :] = np.array([(i + 0.5) * 360 / 5, 0.4])


class VectorObject:

    """
    Position is the object's coordinates.
    Nodes are the predefined, static definition of object "corner points", around object position anchor point (0,0,0).
    RotatedNodes are the Nodes rotated by the given Angles and moved to Position.
    TransNodes are the RotatedNodes transformed from 3D to 2D (X.Y) screen coordinates.

    @author: kalle
    """
    def __init__(self):
        self.angles = np.array([0.0, 0.0, 0.0])
        self.angleScale = (2.0 * np.pi) / 360.0             # to scale degrees.
        self.rotationMatrix = np.zeros((3, 3))
        self.rotateSpeed = np.array([0.0, 0.0, 0.0])
        self.nodes = np.zeros((0, 3))                       # nodes will have unrotated X,Y,Z coordinates
        self.rotatedNodes = np.zeros((0, 3))                # rotatedNodes will have X,Y,Z coordinates after rotation ("final 3D coordinates")
        self.transNodes = np.zeros((0, 2))                  # transNodes will have X,Y coordinates
        self.surfaces = []

    def initObject(self):
        self.updateSurfaceCrossProductVector()
        self.updateSurfaceCrossProductLen()

    def addNodes(self, node_array):
        # add nodes (all at once)
        self.nodes = node_array
        self.rotatedNodes = node_array  # initialize rotatedNodes with nodes

    def addSurfaces(self, idnum, color, node_list):
        # add a Surface, defining its properties
        surface = VectorObjectSurface()
        surface.idnum = idnum
        surface.color = color
        surface.nodes = node_list
        self.surfaces.append(surface)

    def increaseAngles(self):
        self.angles += self.rotateSpeed
        for i in range(3):
            if self.angles[i] >= 360:
                self.angles[i] -= 360
            if self.angles[i] < 0:
                self.angles[i] += 360

    def setRotationMatrix(self):
        """ Set matrix for rotation using angles. """

        (sx, sy, sz) = np.sin((self.angles) * self.angleScale)
        (cx, cy, cz) = np.cos((self.angles) * self.angleScale)

        # build a matrix for X, Y, Z rotation (in that order, see Wikipedia: Euler angles) including position shift.
        # add a column of zeros for later position use
        self.rotationMatrix = np.array([[cy * cz               , -cy * sz              , sy      ],
                                        [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
                                        [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]])

    def updateSurfaceCrossProductVector(self):
        # calculate cross product vector for each surface using rotatedNodes
        # always use vectors (1, 0) and (1, 2) (numbers representing nodes)
        # numpy "cross" was terribly slow, calculating directly as below takes about 10 % of the time.
        for surface in self.surfaces:
            vec_A = self.rotatedNodes[surface.nodes[2], 0:3] - self.rotatedNodes[surface.nodes[1], 0:3]
            vec_B = self.rotatedNodes[surface.nodes[0], 0:3] - self.rotatedNodes[surface.nodes[1], 0:3]
            vec_Cross = ([
                vec_B[1] * vec_A[2] - vec_B[2] * vec_A[1],
                vec_B[2] * vec_A[0] - vec_B[0] * vec_A[2],
                vec_B[0] * vec_A[1] - vec_B[1] * vec_A[0]
                ])
            surface.crossProductVector = vec_Cross

    def updateSurfaceCrossProductLen(self):
        # calculate cross product vector length for each surface.
        # this is constant and done only at init stage.
        for surface in self.surfaces:
            surface.setCrossProductLen()

    def updateSurfaceAngleToViewer(self, zPos):
        # calculate acute angle between surface plane and Viewer
        # surface plane cross product vector and Viewer vector both from node 1.
        for surface in self.surfaces:
            vec_Viewer = self.rotatedNodes[surface.nodes[1], 0:3] + np.array([0, 0, zPos])
            surface.setAngleToViewer(vec_Viewer)
            if surface.angleToViewer > 0:
                surface.visible = 1
            else:
                surface.visible = 0

    def rotate(self):
        """
        Apply a rotation defined by a given rotation matrix. First add
        """
        self.rotatedNodes = np.dot(self.nodes, self.rotationMatrix)

    def transform(self, zScale, zPos, midScreen):
        """
        Add screen center.
        """
        # add zPos and tranform from 3D to 2D; add midScreen to center on screen.
        self.transNodes = (self.rotatedNodes[:, 0:2] * zScale) / (self.rotatedNodes[:, 2:3] + zPos) + midScreen


class VectorObjectSurface:

    """
    Surfaces for a VectorObject.

    @author: kalle
    """

    def __init__(self):
        self.idnum = 0
        # properties set when defining the object
        self.nodes = []
        self.color = (0, 0, 0)
        # the following are calculated during program execution
        self.crossProductVector = np.zeros((0, 3))
        self.crossProductLen = 0.0   # precalculated length of the cross product vector - this is constant
        self.angleToViewer = 0.0
        self.angleToLightSource = 0.0
        self.visible = 1

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

    def vectorLen(self, vector):
        # equivalent to numpy.linalg.norm for a 3D real vector, but much faster. math.sqrt is faster than numpy.sqrt.
        return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


if __name__ == '__main__':
    """
    Prepare screen, objects etc.
    """

    # set screen size
    # first check available full screen modes
    pygame.display.init()
    # disp_modes = pygame.display.list_modes(0, pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    # disp_size = disp_modes[4] # selecting display size from available list. Assuming the 5th element is nice...
    disp_size = (1920, 1080)
    # disp_size = (800, 600)

    screen = pygame.display.set_mode(disp_size)
    pygame.display.set_caption('Side Effect Cube')
    SideEffectCube(screen, 60, 120).run()

    # exit; close display, stop music
    pygame.quit()
    exit()
