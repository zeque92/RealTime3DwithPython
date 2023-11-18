# -*- coding: utf-8 -*-
import pygame
import numpy as np
import random
from os import chdir
from sys import exit


class Landscape:
    """
    Generates zoomable fractal landscapes.

    @author: kalle
    """

    def __init__(self, screen):
        self.screen = screen
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.screenCopy = self.screen.copy()
        self.font = pygame.font.SysFont('CourierNew', 14)  # initialize font and set font size
        self.backgroundColor = (0, 0, 0)
        self.randSize = 300.0                            # randSize determines how steep or flat the landscape is (bigger = steeper)
        self.landSize = 6                                # landSize determines grid size as 2 ** landSize + 1 in both directions
        self.AASize = 7                                  # landSize <= AASize will use AntiAliasing. For bigger sizes, dropped for speed.
        self.gridSize = 0
        self.zFront = 0.0                                # scaling for z coordinates
        self.zBack = 0.0
        self.yHeight = 0.0
        self.colorScale = 0.0
        self.mountainHeight = 0.0
        self.landHeight = 0.0
        self.seaHeight = 0.0
        self.seaColorMult = 0.0
        self.yAdd = np.zeros((1, 2 ** 10 + 1))
        self.zAdd = np.zeros((1, 2 ** 10 + 1))
        self.zoomPosition = (0, 0)
        self.zoomLevel = 0
        self.zoomerPoints = np.zeros((2, 2, 2),  dtype=int)
        self.mousePosition = (0, 0)
        self.grid = np.zeros((2 ** self.landSize + 1,  2 ** self.landSize + 1),  dtype=float)
        self.tilt = 0.1                                  # tilt towards viewer; determines how much higher the far-away points (on average) are
        self.midScreen = np.array([int(self.width / 2), int(self.height / 2)],  dtype=float)
        self.iceRGB = (240, 240, 240)
        self.mountainRGB = (180, 120, 80)
        self.landRGB = (30, 220, 40)
        self.seaRGB = (60, 60, 255)
        self.zoomerRGB = (128, 128, 128)
        self.zoomerMidRGB = (255, 255, 255)
        self.autoShow = True
        self.autoScapeCount = 10
        self.autoScapes = 0
        self.autoFrameCount = 30                        # refers to how many cycles of pygame.time.wait(100) will be passed before moving on
        self.autoFrames = 0
        self.running = True
        self.stop = False

    def run(self):
        """ Main loop. """

        self.initGrid()
        self.generateGrid()
        self.drawGrid()

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
                    if event.key == pygame.K_RIGHT and self.landSize < 10:
                        self.increaseLandSize()
                        self.autoShow = False
                    if event.key == pygame.K_LEFT and self.landSize > 3:
                        self.decreaseLandSize()
                        self.autoShow = False
                    if event.key == pygame.K_UP and self.randSize < 1200:
                        self.increaseRandSize()
                        self.autoShow = False
                    if event.key == pygame.K_DOWN and self.randSize > 70:
                        self.decreaseRandSize()
                        self.autoShow = False
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                                          self.__class__.__name__ + '.jpg')
                elif event.type == pygame.MOUSEMOTION:
                    self.mousePosition = pygame.mouse.get_pos()
                    self.drawZoomer()
                    self.autoShow = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.autoShow = False
                    btns = pygame.mouse.get_pressed(num_buttons=3)
                    if btns[0] and btns[2]:
                        # left and righ mouse buttons: exit
                        self.running = False
                    # if event.button == 1:
                    elif btns[0]:
                        # left button: whole new landscape
                        self.zoomLevel = 0
                        self.initGrid()
                        self.generateGrid()
                        self.drawGrid()
                    # if event.button == 3:
                    elif btns[2]:
                        # right button: zoom in using the zoomer
                        self.zoomGrid()

            if self.autoShow:
                self.autoFrames += 1
                if self.autoFrames > self.autoFrameCount:
                    self.autoScapes += 1
                    if self.autoScapes > self.autoScapeCount:
                        self.running = False
                    else:
                        self.autoFrames = 0
                        if self.autoScapes == 4 or self.autoScapes == 6:
                            self.landSize += 1  # increase land size / precision
                            self.grid = np.zeros((2 ** self.landSize + 1,  2 ** self.landSize + 1),  dtype=float)
                        # draw a whole new landscape
                        self.initGrid()
                        self.generateGrid()
                        self.drawGrid()

            # wait to give CPU a chance to do something else...
            pygame.time.wait(100)

        return self.stop

    def initGrid(self):

        self.gridSize = 2 ** self.landSize
        self.zFront = (self.width * 0.9) / self.gridSize   # Scaling for z coordinates
        self.zBack = self.zFront / 1.5
        self.yHeight = self.height / self.zFront * 1.5
        self.mountainHeight = (2 ** self.zoomLevel) * self.randSize / 2
        self.landHeight = (2 ** self.zoomLevel) * self.randSize / 4
        self.seaHeight = 0
        self.seaColorMult = 1.0 / 64.0
        # zAdd is the vector (size: grid width) with increasing width of polygons (starting with smaller, far away).
        self.zAdd = np.linspace(self.zBack, self.zFront, num=self.gridSize + 1, endpoint=True)
        # yAdd is the vector (size: grid width) with increasing (0 is at the top) "basic height" for each row of polygons.
        self.yAdd = (self.yHeight * self.zAdd) - self.yHeight * (self.zAdd[0] + self.zAdd[self.gridSize]) / 2 + self.midScreen[1] * 1.3

    def generateGrid(self, last_iter_only=0):
        """
        Generates a new landscape grid by creating a random altitude map.
        Uses a simple fractal method called mid-point replacement.

        @author: kalle
        """

        if last_iter_only == 0:
            # full grid generation. Start from corners.
            rSize = self.randSize
            startSize = self.landSize
            # set corner values. Tilt: Use higher altitudes for back of grid.
            self.grid[0, 0] = (random.random() - 0.5 + self.tilt * 2) * rSize
            self.grid[0, self.gridSize] = (random.random() - 0.5 + self.tilt * 2) * rSize
            self.grid[self.gridSize, 0] = (random.random() - 0.5 - self.tilt) * rSize
            self.grid[self.gridSize, self.gridSize] = (random.random() - 0.5 - self.tilt) * rSize
        else:
            # just the last iteration. The grid is built from the previous grid but only has "every other point" so needs one round.
            rSize = self.randSize / (2.0 ** (self.landSize - 1))
            startSize = 1

        # go through grid by adding a mid point first on axis 0 (X), then on axis 1 (Z), as average of end points + a random shift
        # each round the rSize will be halved as the distance between end points (step) is halved as well

        for s in range(startSize, 0, -1):
            halfStep = 2 ** (s - 1)
            step = 2 * halfStep
            # generate mid point in x for each z
            for z in range(0, self.gridSize + 1, step):
                for x in range(step, self.gridSize + 1, step):
                    self.grid[x - halfStep, z] = (self.grid[x - step, z] + self.grid[x, z]) / 2 + (random.random() - 0.5) * rSize
            # generate mid point in z for each x (including the nex x just created, so using halfStep)
            for x in range(0, self.gridSize + 1, halfStep):
                for z in range(step, self.gridSize + 1, step):
                    self.grid[x, z - halfStep] = (self.grid[x, z - step] + self.grid[x, z]) / 2 + (random.random() - 0.5) * rSize
            rSize = rSize / 2
        # store the last steepness component for color scaling
        self.colorScale = 6.0 * rSize

    def drawGrid(self):
        """
        Draws the grid.

        @author: kalle
        """

        self.screen.lock()
        # clear screen
        self.screen.fill(self.backgroundColor)
        # add information text
        self.plotInfo()
        self.plotInstructions()
        # first transform the 3D grid to 2D screen coordinates.
        midGrid = int(self.gridSize / 2)
        yAdd = self.yAdd
        zAdd = self.zAdd

        # z goes through each grid row ("vertical line")
        for z in range(self.gridSize):
            ySea1 = yAdd[z] - self.seaHeight
            ySea2 = yAdd[z + 1] - self.seaHeight
            # x goes thriugh each grid column for line z
            for x in range(self.gridSize):
                # coordinates for square, two triangles: 0 - 1 - 2 and 1 - 3 - 2.
                # top left.
                x0 = zAdd[z] * (x - midGrid) + self.midScreen[0]
                y0 = yAdd[z] - self.grid[x, z]
                # top right.
                x1 = x0 + zAdd[z]
                y1 = yAdd[z] - self.grid[x + 1, z]
                # bottom left.
                x2 = zAdd[z + 1] * (x - midGrid) + self.midScreen[0]
                y2 = yAdd[z + 1] - self.grid[x, z + 1]
                # bottom right.
                x3 = x2 + zAdd[z + 1]
                y3 = yAdd[z + 1] - self.grid[x + 1, z + 1]
                # minimum Y for each triangle.
                yMin = min(y0, y1, y2, y3)
                if yMin >= ySea1:
                    # if sea (all coordinates below sea level), draw a flat square
                    color = self.setColorSea(yAdd[z] - yMin)
                    node_list = [(x0, ySea1), (x1, ySea1), (x3, ySea2), (x2, ySea2)]
                    self.drawPolygon(self.screen, color, node_list)
                else:
                    # if land, draw two polygons, if they are visible (i.e. "higher" (smaller y) at the back than at the front).
                    # first triangle 0 - 1 - 2
                    if y0 < y2 and y2 > 0:
                        color = self.setColorLand(yAdd[z] - yMin, self.grid[x, z] - self.grid[x + 1, z], self.grid[x, z + 1] - self.grid[x, z])
                        node_list = [(x0, min(y0, ySea1)), (x1, min(y1, ySea1)), (x2, min(y2, ySea2))]
                        self.drawPolygon(self.screen, color, node_list)
                    # then triangle 1 - 3 - 2
                    if y1 < y3 and y3 > 0:
                        color = self.setColorLand(yAdd[z] - yMin, self.grid[x + 1, z + 1] - self.grid[x, z + 1], self.grid[x + 1, z + 1] - self.grid[x + 1, z])
                        node_list = [(x1, min(y1, ySea1)), (x3, min(y3, ySea2)), (x2, min(y2, ySea2))]
                        self.drawPolygon(self.screen, color, node_list)
            # every other horizontal set of triangles, flip screen to show landscape being built.
            # this is not mandatory, but used to mimick the original 1992 code.
            if z % 2 == 0:
                pygame.display.flip()

        # clean the front
        node_list = [(zAdd[self.gridSize] * (0 - midGrid) + self.midScreen[0], self.height)]
        for x in range(self.gridSize + 1):
            node_list.append((zAdd[self.gridSize] * (x - midGrid) + self.midScreen[0], yAdd[self.gridSize] - self.grid[x, self.gridSize]))
        node_list.append((zAdd[self.gridSize] * (self.gridSize - midGrid) + self.midScreen[0], self.height))
        self.drawPolygon(self.screen, self.backgroundColor, node_list)

        self.plotInfo()

        # create a copy for cleaning up after zoomer. Not very memory efficient but easy.
        # first release any locks on screenCopy
        while self.screenCopy.get_locked():
            self.screenCopy.unlock()
        self.screenCopy = self.screen.copy()

        # release any locks on screen
        while self.screen.get_locked():
            self.screen.unlock()

        # switch between currently showed and the next screen (prepared in "buffer")
        pygame.display.flip()

        # clear event queue
        self.clearEvents()

    def setColorLand(self, y, diffX, diffZ):

        # defines color. diffX and diffZ are the differences in altitude in x and z coordinates, respectively,  going right and towards the viewer
        s = max(1.0 - np.sqrt(diffX ** 2 + diffZ ** 2) / self.colorScale, 0.2)
        if y > self.mountainHeight:
            use_color = ([int(s * x + 0.5) for x in self.iceRGB])
        elif y > self.landHeight:
            use_color = ([int(s * x + 0.5) for x in self.mountainRGB])
        else:
            use_color = ([int(s * x + 0.5) for x in self.landRGB])
        return use_color

    def setColorSea(self, y):

        # defines color for sea, based on depth.
        s = max(1.0 - (-y + self.seaHeight) * self.seaColorMult, 0.2)
        use_color = ([int(s * x + 0.5) for x in self.seaRGB])
        return use_color

    def drawZoomer(self):

        # adds zoomer sighting points.
        # use mouse coordinates to locate zoomer on grid. Use "zero height" ie. independent of grid / landscape height.
        midGrid = int(self.gridSize / 2)
        # first figure out Z coordinate using mouse Y.
        zoomTopZ = max(0, min(midGrid, np.searchsorted(self.yAdd, self.mousePosition[1]) - int(midGrid / 2)))
        # then, get X on that Z horizontal line using mouse X
        zoomLeftX = max(0, min(midGrid, (self.mousePosition[0] - self.midScreen[0]) / self.zAdd[zoomTopZ + int(midGrid / 2)] + int(midGrid / 2)))
        self.zoomPosition = (zoomLeftX, zoomTopZ)

        zoomX = np.array([zoomLeftX, zoomLeftX + midGrid], dtype=int)
        zoomZ = np.zeros((2, 2), dtype=int)
        zoomZ[0, 0] = zoomTopZ                  # beginning of top Z
        zoomZ[0, 1] = zoomTopZ + midGrid        # end of top Z = beginning of bottom Z
        zoomZ[1, 0] = zoomTopZ + midGrid        # beginning of bottom Z
        zoomZ[1, 1] = self.gridSize             # end of bottom Z = end of grid
        # loop through the four zoomer corners.
        for x in [0, 1]:
            minY = self.height
            for z in [1, 0]:
                ySea = self.yAdd[zoomZ[z, 0]] - self.seaHeight
                y0 = min(ySea, self.yAdd[zoomZ[z, 0]] - self.grid[zoomX[x], zoomZ[z, 0]])
                # get minimum Z in front of current zoomer corner. No need to chack for sea level here.
                # for bottom corners, minimum of all points from it to grid front end; for top corners, minimum of bottom result and points between top and bottom.
                minY = min(minY, min(self.yAdd[zoomZ[z, 0]: zoomZ[z, 1] + 1] - self.grid[zoomX[x], zoomZ[z, 0]: zoomZ[z, 1] + 1]))
                # test if zoomer corner is at or above respective minY - if so, its is not hidden behind the landscape, draw it
                if y0 <= minY:
                    x0 = self.zAdd[zoomZ[z, 0]] * (zoomX[x] - midGrid) + self.midScreen[0]
                    self.drawZoomerPoint(x0, y0, x, z)
                else:
                    # zoomer point is hidden - call drawZoomerPoint to remove the previous point
                    self.drawZoomerPoint(0, 0, x, z)

        # switch between currently showed and the next screen (prepared in "buffer")
        pygame.display.flip()

    def drawZoomerPoint(self, x0, y0, x, z):

        # draws a zoomer point
        # first remove previous zoomer point by copying landscape image under it from screenCopy
        if self.zoomerPoints[x, z, 0] > 0 and self.zoomerPoints[x, z, 1] > 0:
            rect = pygame.Rect(self.zoomerPoints[x, z, 0] - 3, self.zoomerPoints[x, z, 1] - 3, 7, 7)
            self.screen.blit(self.screenCopy, rect, rect)
        # then add the new zoomer point
        if x0 > 0 and y0 > 0:
            self.screen.fill(self.zoomerRGB, rect=[(x0 - 3, y0 - 3), (7, 7)])
            self.screen.fill(self.zoomerMidRGB, rect=[(x0 - 1, y0 - 1), (3, 3)])
        # and store its coordinates
        self.zoomerPoints[x, z, 0] = x0
        self.zoomerPoints[x, z, 1] = y0

    def drawPolygon(self, screen, color, node_list):

        # draws a filled polygon
        if self.landSize <= self.AASize:
            # for small land/grid sizes, use antialiasing
            pygame.draw.aalines(screen, color, True, node_list)
        pygame.draw.polygon(screen, color, node_list)

    def zoomGrid(self):

        # zoom in using the current grid zoomer area.
        midGrid = int(self.gridSize / 2)
        zoomPosX = int(self.zoomPosition[0])
        zoomPosZ = int(self.zoomPosition[1])
        zoomMult = 2
        # copy data from both sides of the zoomer area to the middle so no overlap destroys data.
        # multiply the height with zoomMult to preserve steepness, but only once (with z, not again with x).
        for z in range(midGrid, -1, -1):
            if z * 2 > zoomPosZ + z:
                self.grid[:, z * 2] = zoomMult * self.grid[:, zoomPosZ + z]
            else:
                break
        for z in range(0, midGrid + 1):
            if z * 2 <= zoomPosZ + z:
                self.grid[:, z * 2] = zoomMult * self.grid[:, zoomPosZ + z]
            else:
                break
        for x in range(midGrid, -1, -1):
            if x * 2 > zoomPosX + x:
                self.grid[x * 2, :] = self.grid[zoomPosX + x, :]
            else:
                break
        for x in range(0, midGrid + 1):
            if x * 2 <= zoomPosX + x:
                self.grid[x * 2, :] = self.grid[zoomPosX + x, :]
            else:
                break
        # adjust zoom level to preserve colors as they were
        self.zoomLevel += 1
        self.mountainHeight *= zoomMult
        self.landHeight *= zoomMult
        self.seaHeight *= zoomMult
        self.seaColorMult = 1.0 / 64.0

        # add the final grid iteration
        self.generateGrid(1)
        self.drawGrid()

    def increaseLandSize(self):

        # increase land size (double the number of polygons both horizontally and vertically)
        self.landSize += 1
        self.initGrid()
        self.zoomPosition = (int(2 * self.zoomPosition[0]), int(2 * self.zoomPosition[1]))
        gSize = int(self.gridSize / 2)
        # use current grid as a basis for a new, more accurate grid. "Expand" from top left quartile.
        self.grid = np.hstack((self.grid, self.grid[:, 0:gSize]))  # double size horizontally
        self.grid = np.vstack((self.grid, self.grid[0:gSize, :]))  # double size vertically
        for z in range(gSize, 0, -1):
            self.grid[0:gSize + 1, z * 2] = self.grid[0:gSize + 1, z]
        for x in range(gSize, 0, -1):
            self.grid[x * 2, :] = self.grid[x, :]
        # # add the final grid iteration
        self.generateGrid(1)
        self.drawGrid()

    def decreaseLandSize(self):

        # decrease land size (halve the number of polygons both horizontally and vertically)
        self.landSize -= 1
        self.initGrid()
        self.zoomPosition = (int(self.zoomPosition[0] / 2), int(self.zoomPosition[1] / 2))
        gSize = self.gridSize + 1
        # use current grid as a basis for a new, less accurate grid. "Compress" to top left quartile.
        for z in range(1, gSize):
            self.grid[:, z] = self.grid[:, z * 2]
        for x in range(1, gSize):
            self.grid[x, 0:gSize] = self.grid[x * 2, 0:gSize]
        self.grid = np.split(np.split(self.grid, [gSize, gSize * 2 - 1], axis=0)[0], [gSize, gSize * 2 - 1], axis=1)[0]  # use the upper left quartile of grid as the new grid
        self.colorScale *= 2  # adjust colorScale accordingly
        self.drawGrid()

    def increaseRandSize(self):

        # increase randomizer (steepness)
        self.randSize *= 1.4
        self.initGrid()
        # self.generateGrid()
        self.grid *= 1.4
        self.colorScale *= 1.4
        self.drawGrid()

    def decreaseRandSize(self):

        # decrease randomizer (steepness)
        self.randSize /= 1.4
        self.initGrid()
        # self.generateGrid()
        self.grid /= 1.4
        self.colorScale /= 1.4
        self.drawGrid()

    def plotInfo(self):
        """
        Add info on screen.
        This can obviously be skipped, just for information.
        """

        # release any locks on screen
        while self.screen.get_locked():
            self.screen.unlock()

        # self.screen.fill(self.backgroundColor, rect=[(10, 10), (200, 120)])   # clear info area
        self.plotInfoBlit(10,  10, "land size   " + (" " * 10 + "2^" + str(int(self.landSize)))[-10:])
        self.plotInfoBlit(10,  25, "grid size   " + (" " * 10 + str(int(self.gridSize + 1)) + "x" + str(int(self.gridSize + 1)))[-10:])
        self.plotInfoBlit(10,  40, "grid points " + (" " * 10 + str(int((self.gridSize) + 1) ** 2))[-10:])
        self.plotInfoBlit(10,  55, "surfaces    " + (" " * 10 + str(int((self.gridSize) ** 2) * 2))[-10:])
        self.plotInfoBlit(10,  70, "steepness   " + (" " * 10 + str(int(self.randSize)))[-10:])
        self.plotInfoBlit(10,  85, "zoom level  " + (" " * 10 + str(int(self.zoomLevel)))[-10:])

        # switch between currently showed and the next screen (prepared in "buffer")
        pygame.display.flip()

    def plotInstructions(self):
        """
        Add use instructions on screen.
        This can obviously be skipped, just for information.
        """

        # release any locks on screen
        while self.screen.get_locked():
            self.screen.unlock()

        self.screen.fill(self.backgroundColor, rect=[(400, 10), (200, 120)])   # clear info area
        self.plotInfoBlit(400,  10, "cursor left/right:  decrease/increase land size")
        self.plotInfoBlit(400,  25, "cursor down/up:     decrease/increase steepness")
        self.plotInfoBlit(400,  40, "left mouse button:  new landscape")
        self.plotInfoBlit(400,  55, "right mouse button: zoom selected area")
        self.plotInfoBlit(400,  70, "both mouse buttons: exit")
        self.plotInfoBlit(400,  85, "f key:              toggle full screen mode on/off")

        # switch between currently showed and the next screen (prepared in "buffer")
        pygame.display.flip()

    def plotInfoBlit(self, x, y, msg):

        f_screen = self.font.render(msg, False, [255, 255, 255])
        self.screen.blit(f_screen, (x, y))

    def clearEvents(self):

        # only accept QUIT event and ESC key, otherwise clear queue.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def toggleFullScreen(self):

        # toggle between fulls creen and windowed mode
        pygame.display.toggle_fullscreen()
        self.screenCopy = self.screen.copy()
        # redraw.
        self.drawGrid()


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
    disp_size = disp_modes[9]  # selecting display size from available list. Assuming the 9th element is nice...
    # disp_size = (1920, 1080) # to force display size
    pygame.font.init()

    # initialize mixer
    pygame.mixer.init()
    music_file = "sinking2.ogg"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/release.php?id=45536
    # start music player
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play(loops=-1)

    screen = pygame.display.set_mode(disp_size)
    pygame.display.set_caption('Landscape')
    Landscape(screen).run()

    # exit; close display, stop music
    pygame.quit()
    exit()


