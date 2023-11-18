# -*- coding: utf-8 -*-
import pygame
import numpy as np
import math
import os
from operator import itemgetter
import copy
import xml.etree.ElementTree as et

key_to_function = {
    pygame.K_ESCAPE: (lambda x: x.terminate()),         # ESC key to quit
    pygame.K_SPACE:  (lambda x: x.pause())              # SPACE to pause
    }

class VectorViewer:
    """
    Displays 3D vector objects on a Pygame screen.

    @author: kalle
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width,height))
        self.fullScreen = False
        pygame.display.set_caption('VectorViewer')
        self.backgroundColor = (0,0,0)
        self.VectorAnglesList = []
        self.VectorObjs = []
        self.VectorPos = VectorPosition()
        self.midScreen = np.array([width / 2, height / 2], dtype=float)
        self.zScale = width * 0.7                       # Scaling for z coordinates
        self.objMinZ = 100.0                            # minimum z coordinate for object visibility
        self.lightPosition = np.array([400.0, 800.0, -500.0])
        self.target_fps = 60                            # affects movement speeds
        self.running = True
        self.paused = False
        self.clock = pygame.time.Clock()

    def setVectorPos(self, VectorPosObj):
        self.VectorPos = VectorPosObj

    def addVectorObj(self, VectorObj):
        self.VectorObjs.append(VectorObj)

    def addVectorAnglesList(self, VectorAngles):
        self.VectorAnglesList.append(VectorAngles)

    def run(self):
        """ Main loop. """

        for VectorObj in self.VectorObjs:
            VectorObj.initObject() # initialize objects

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
                self.rotate()
                self.calculate()
                self.display()

                # release any locks on screen
                while self.screen.get_locked():
                    self.screen.unlock()

                # switch between currently showed and the next screen (prepared in "buffer")
                pygame.display.flip()
                self.clock.tick(self.target_fps) # this keeps code running at max target_fps

        # exit; close display, stop music
        pygame.display.quit()

    def rotate(self):
        """
        Rotate all objects. First calculate rotation matrix.
        Then apply the relevant rotation matrix with object position to each VectorObject.
        """

        # calculate rotation matrices for all angle sets
        for VectorAngles in self.VectorAnglesList:
            VectorAngles.setRotateAngles()
            VectorAngles.setRotationMatrix()

        # rotate object positions, copy those to objects.
        self.VectorPos.rotate()
        for (node_num, VectorObj) in self.VectorPos.objects:
            VectorObj.setPosition(self.VectorPos.rotatedNodes[node_num, :])

        # rotate and flatten (transform) objects
        for VectorObj in self.VectorObjs:
            VectorObj.updateVisiblePos(self.objMinZ) # test for object position Z
            if VectorObj.visible == 1:
                VectorObj.rotate() # rotates objects in 3D
                VectorObj.updateVisibleNodes(self.objMinZ) # test for object minimum Z
                if VectorObj.visible == 1:
                    VectorObj.transform(self.midScreen, self.zScale) # flattens to 2D, crops X,Y
                    VectorObj.updateVisibleTrans(self.midScreen) # test for outside of screen

    def calculate(self):
        """
        Calculate shades and visibility.
        """

        for VectorObj in (vobj for vobj in self.VectorObjs if vobj.visible == 1):
            # calculate angles to Viewer (determines if surface is visible) and LightSource (for shading)
            VectorObj.updateSurfaceZPos()
            VectorObj.updateSurfaceCrossProductVector()
            VectorObj.updateSurfaceAngleToViewer()
            VectorObj.updateSurfaceAngleToLightSource(self.lightPosition)
            VectorObj.updateSurfaceColorShade()

    def display(self):
        """
        Draw the VectorObjs on the screen.
        """

        # lock screen for pixel operations
        self.screen.lock()

        # clear screen.
        self.screen.fill(self.backgroundColor)

        # first sort VectorObjs so that the most distant is first.
        self.VectorObjs.sort(key=lambda VectorObject: VectorObject.position[2], reverse=True)

        # draw the actual objects
        for VectorObj in (vobj for vobj in self.VectorObjs if vobj.visible == 1):
            # first sort object surfaces so that the most distant is first. For concave objects there should be no overlap, though.
            VectorObj.sortSurfacesByZPos()
            # then draw surface by surface.
            for surface in (surf for surf in VectorObj.surfaces if surf.visible == 1):
                # build a list of transNodes for this surface
                node_list = ([VectorObj.transNodes[node][:2] for node in surface.nodes])
                if len(node_list) > 2:
                    pygame.draw.aalines(self.screen, surface.colorRGB(), True, node_list)
                    pygame.draw.polygon(self.screen, surface.colorRGB(), node_list, surface.edgeWidth)

        # unlock screen
        self.screen.unlock()

    def terminate(self):

        self.running = False

    def pause(self):

        if self.paused == True:
            self.paused = False
        else:
            self.paused = True

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
        self.angles = VectorAngles()
        self.nodes = np.zeros((0, 4))                       # nodes will have unrotated X,Y,Z coordinates plus a column of ones for position handling
        self.rotatedNodes = np.zeros((0, 3))                # rotatedNodes will have X,Y,Z coordinates after rotation ("final 3D coordinates")
        self.transNodes = np.zeros((0, 2))                  # transNodes will have X,Y coordinates
        self.surfaces = []
        self.visible = 1
        self.objName = ""
        self.minShade = 0.2                                 # shade (% of color) to use when surface is parallel to light source

    def initObject(self):
        self.updateSurfaceZPos()
        self.updateSurfaceCrossProductVector()
        self.updateSurfaceCrossProductLen()

    def setPosition(self, position):
        # move object by giving it a rotated position.
        self.position = position

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
        # check if object is visible. If any of node Z coordinates are too close to viewer, set to 0
        if self.position[2] < objMinZ:
            self.visible = 0
        else:
            self.visible = 1

    def updateVisibleNodes(self, objMinZ):
        # check if object is visible. If any of node Z coordinates are too close to viewer, set to 0
        if min(self.rotatedNodes[:, 2]) < objMinZ:
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
        for surface in (surf for surf in self.surfaces if surf.visible == 1 and self.minShade < 1.0):
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

    def rotate(self):
        """
        Apply a rotation defined by a given rotation matrix.
        """
        matrix = np.vstack((self.angles.rotationMatrix, self.position[0:3]))   # add position to rotation matrix to move object at the same time
        self.rotatedNodes = np.dot(self.nodes, matrix)

    def transform(self, midScreen, zScale):
        """
        Add screen center.
        """
        # add midScreen to center on screen to get to transNodes. Invert Z as coordinates partially inverted at setup!
        self.transNodes = (self.rotatedNodes[:, 0:2] * zScale) / (-self.rotatedNodes[:, 2:3]) + midScreen

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

    def colorRGB(self):
        use_color = ([round(self.colorShade * x, 0) for x in self.color]) # apply shading
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
        self.rotate = np.array([0.0, 1.0, 0.0])

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

        # build a matrix for X, Y, Z rotation (in that order, see Wikipedia: Euler angles) including position shift.
        # add a column of zeros for later position use
        self.rotationMatrix = np.array([[cy * cz               , -cy * sz              , sy      ],
                                        [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
                                        [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]])

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
        self.rotatedNodes = np.dot(self.nodes, matrix) + self.position[0:3]


if __name__ == '__main__':
    """
    Prepare screen, read objects etc. from file.
    """

    # set screen size
    # first check available full screen modes
    pygame.display.init()
    # disp_modes = pygame.display.list_modes(0, pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    # disp_size = disp_modes[4] # selecting display size from available list. Assuming the 5th element is nice...
    disp_size = (1280, 800)

    vv = VectorViewer(disp_size[0], disp_size[1])

    # read data file defining angles, movements and objects
    vecdata = et.parse("vectordata cityscape.xml")

    root = vecdata.getroot()

    for angles in root.iter('vectorangles'):
        ang = VectorAngles()
        ang.angName = angles.get('name')
        ang.angles[0] = float(angles.findtext("angleX", default="0"))
        ang.angles[1] = float(angles.findtext("angleY", default="0"))
        ang.angles[2] = float(angles.findtext("angleZ", default="0"))
        vv.addVectorAnglesList(ang)

    for vecobjs in root.iter('vectorobject'):
        vobj = VectorObject()
        vobj.objName = vecobjs.get('name')

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
                vobj.rotate()
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

        # add the object
        vv.addVectorObj(vobj)

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
