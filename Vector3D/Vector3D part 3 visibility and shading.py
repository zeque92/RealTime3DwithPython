# -*- coding: utf-8 -*-
import pygame
import numpy as np
import math

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
        self.VectorObjs = []
        self.midScreen = np.array([width / 2, height / 2], dtype=float)
        self.zScale = width * 0.7                       # Scaling for z coordinates
        self.lightPosition = np.array([400.0, 800.0, -500.0])
        self.target_fps = 60                            # affects movement speeds
        self.running = True
        self.paused = False
        self.clock = pygame.time.Clock()
            
    def addVectorObj(self, VectorObj):
        self.VectorObjs.append(VectorObj)
             
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
                
        # rotate and flatten (transform) objects
        for VectorObj in self.VectorObjs:
            VectorObj.increaseAngles()
            VectorObj.setRotationMatrix()
            VectorObj.rotate()
            VectorObj.transform(self.zScale, self.midScreen)
            
    def calculate(self):
        """ 
        Calculate shades and visibility. 
        """

        for VectorObj in self.VectorObjs:
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
                   
        # draw the actual objects
        for VectorObj in self.VectorObjs:
            # first sort object surfaces so that the most distant is first. For concave objects there should be no overlap, though.
            VectorObj.sortSurfacesByZPos()
            # then draw surface by surface, if surface is visible.
            for surface in (surf for surf in VectorObj.surfaces if surf.visible == 1):
                # build a list of transNodes for this surface
                node_list = ([VectorObj.transNodes[node][:2] for node in surface.nodes])
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
        self.angles = np.array([0.0, 0.0, 0.0])
        self.angleScale = (2.0 * np.pi) / 360.0             # to scale degrees. 
        self.rotationMatrix = np.zeros((3,3))
        self.rotateSpeed = np.array([0.0, 0.0, 0.0])
        self.nodes = np.zeros((0, 4))                       # nodes will have unrotated X,Y,Z coordinates plus a column of ones for position handling
        self.rotatedNodes = np.zeros((0, 3))                # rotatedNodes will have X,Y,Z coordinates after rotation ("final 3D coordinates")
        self.transNodes = np.zeros((0, 2))                  # transNodes will have X,Y coordinates
        self.surfaces = []
        self.minShade = 0.2                                 # shade (% of color) to use when surface is parallel to light source

    def initObject(self):
        self.updateSurfaceZPos()
        self.updateSurfaceCrossProductVector()
        self.updateSurfaceCrossProductLen()

    def setPosition(self, position):
        # move object by giving it a rotated position.
        self.position = position 

    def setRotateSpeed(self, angles):
        # set object rotation speed.
        self.rotateSpeed = angles 

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
    
    def increaseAngles(self):
        self.angles += self.rotateSpeed
        for i in range(3):
            if self.angles[i] >= 360: self.angles[i] -= 360
            if self.angles[i] < 0: self.angles[i] += 360

    def setRotationMatrix(self):
        """ Set matrix for rotation using angles. """
        
        (sx, sy, sz) = np.sin((self.angles) * self.angleScale)
        (cx, cy, cz) = np.cos((self.angles) * self.angleScale)
 
        # build a matrix for X, Y, Z rotation (in that order, see Wikipedia: Euler angles) including position shift. 
        # add a column of zeros for later position use
        self.rotationMatrix = np.array([[cy * cz               , -cy * sz              , sy      ],
                                        [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
                                        [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]])

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
        matrix = np.vstack((self.rotationMatrix, self.position[0:3]))   # add position to rotation matrix to move object at the same time
        self.rotatedNodes = np.dot(self.nodes, matrix)

    def transform(self, zScale, midScreen):
        """ 
        Add screen center.
        """
        # add midScreen to center on screen to get to transNodes.
        self.transNodes = (self.rotatedNodes[:, 0:2] * zScale) / (self.rotatedNodes[:, 2:3]) + midScreen

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
    
       
if __name__ == '__main__':
    """ 
    Prepare screen, objects etc.
    """

    # set screen size
    # first check available full screen modes
    pygame.display.init()
    # disp_modes = pygame.display.list_modes(0, pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    # disp_size = disp_modes[4] # selecting display size from available list. Assuming the 5th element is nice...
    disp_size = (1280, 800)

    vv = VectorViewer(disp_size[0], disp_size[1])

    # set up a simple cube
    vobj = VectorObject()
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
    # then define surfaces
    node_list = [0, 3, 2, 1] # node_list defines the four nodes forming a cube surface, in clockwise order
    vobj.addSurfaces(0, (255,255,255), 0, 0, (50,50,50), node_list)
    node_list = [4, 5, 6, 7]
    vobj.addSurfaces(1, (255,255,255), 0, 0, (50,50,50), node_list)
    node_list = [0, 1, 5, 4]
    vobj.addSurfaces(2, (255,255,255), 0, 0, (50,50,50), node_list)
    node_list = [3, 7, 6, 2]
    vobj.addSurfaces(3, (255,255,255), 0, 0, (50,50,50), node_list)
    node_list = [0, 4, 7, 3]
    vobj.addSurfaces(4, (255,255,255), 0, 0, (50,50,50), node_list)
    node_list = [1, 2, 6, 5]
    vobj.addSurfaces(5, (255,255,255), 0, 0, (50,50,50), node_list)

    speed_angles = np.array([1.0, -.3, 0.55])
    vobj.setRotateSpeed(speed_angles)
    position = np.array([0.0, 0.0, 1500.0, 1.0])
    vobj.setPosition(position)
 
    # add the object
    vv.addVectorObj(vobj)
     
    # run the main program
    vv.run()
