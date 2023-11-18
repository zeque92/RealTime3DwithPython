# -*- coding: utf-8 -*-
import pygame
import numpy as np
from os import chdir
from sys import exit

class MilkyWay:
    """
    Displays a morphing / blurring red-yellow image.

    @author: kalle
    """

    def __init__(self, screen, target_fps, run_seconds):
        self.run_seconds = run_seconds
        self.screen = screen
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.screenCopy = self.screen.copy()
        self.screenColorize = self.screen.copy()                        # used if not using palette
        self.screenCount = 127                                          # nr of separately stored images. Images are stored so the "oldest" one can be subtracted from final image.
                                                                        # obviously, it would be possible to store e.g. only the "transNodes" and redraw these images as needed - slower but memory-friendlier.
        self.screenCounter = self.screenCount - 1
        self.screenList = []
        self.rectList = []
        self.font = pygame.font.SysFont('CourierNew', 14)               # initialize font and set font size
        self.clock = pygame.time.Clock()
        self.backgroundColor = (0, 0, 0)
        self.barColor = (160, 30, 0)
        # self.objColor = (2, 0, 0)                                       # using red colour to set the shadecolor from the palette - the R component must be integer < 256 / screenCount
        self.objColor = (1, 2, 2)                                       # not using the palette, faster --> this will result in continuous red, "clipped" green and blue later
        self.nrImagesPerFrame = 4                                       # how many images to draw per frame. Larger number will make the effect smoother.
        self.palette = np.zeros((256, 3), dtype=np.uint8)
        self.midScreen = np.array([self.width / 2, self.height / 2], dtype=int)
        self.angleScale = (2.0 * np.pi) / 360.0
        self.nodes = np.zeros((4, 3), dtype=float)                      # nodes will have unrotated X,Y,Z coordinates.
        self.rotatedNodes = np.zeros((4, 3), dtype=float)               # rotatedNodes will have X,Y,Z coordinates after rotation
        self.transNodes = np.zeros((4, 2), dtype=int)                   # transNodes will have X,Y coordinates
        self.angles = np.zeros((3), dtype=float)                        # angles
        self.angleAdd = np.zeros((3), dtype=float)                      # rotation
        self.surfaces = []                                              # surfaces connect nodes. Note every other surface will be drawn with background color!
        self.zPos = 3000.0
        self.zScale = self.height
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

        self.timer_names.append("draw image")
        self.timer_names.append("blend images")
        self.timer_names.append("colorize")
        self.timer_names.append("plot info")
        self.timer_names.append("display flip")
        self.timer_names.append("wait")
        self.timers = np.zeros((len(self.timer_names), self.timer_avg_frames), dtype=int)

        self.angleAdd = np.array([0.7, 0.2, -3.0])
        self.nodes = np.array([
                [ 800,  800, 0],
                [ 800, -800, 0],
                [-800, -800, 0],
                [-800,  800, 0],
                [ 500,  500, 0],
                [ 500, -500, 0],
                [-500, -500, 0],
                [-500,  500, 0],
                [ 350,  350, 0],
                [ 400, -400, 0],
                [-350, -350, 0],
                [-400,  400, 0],
                [ 250,  250, 0],
                [ 300, -300, 0],
                [-250, -250, 0],
                [-300,  300, 0],
                [ 200,  200, 0],
                [ 200, -200, 0],
                [-200, -200, 0],
                [-200,  200, 0]
                ])
        self.surfaces.append((0, 3, 2, 1))
        self.surfaces.append((4, 7, 6, 5))
        self.surfaces.append((8, 11, 10, 9))
        self.surfaces.append((12, 15, 14, 13))
        self.surfaces.append((16, 19, 18, 17))

        # generate palette. This 256 color palette will be used for object colors even if display is full color.
        for i in range(0, 100):
            self.palette[i, :] = np.array([int(min(i, 127) * 2), 0, 0],  dtype=np.uint8)
        for i in range(100, 200):
            self.palette[i, :] = np.array([int(min(i, 127) * 2), int(min(i - 100, 127) * 2), 0],  dtype=np.uint8)
        for i in range(200, 256):
            self.palette[i, :] = np.array([int(min(i, 127) * 2), int(min(i - 100, 127) * 2), int(min(i - 200, 127) * 2)],  dtype=np.uint8)

        # create screen copies and init Rect list
        for i in range(self.screenCount):
            self.screenList.append(self.screen.copy())
            self.rectList.append(pygame.Rect(self.midScreen[0], self.midScreen[1], 0, 0))

        self.screen.fill(self.backgroundColor)
        self.addBars()
        self.screenColorize.fill((0, 127, 220))                           # when not using the palette, this will be subtracted and result in continuous red, "clipped" green and blue
        # initialize timers
        self.start_timer = pygame.time.get_ticks()
        self.millisecs = self.start_timer

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

            for i in range(self.nrImagesPerFrame):
                active_screen = self.screenList[self.screenCounter]
                active_screen.fill(self.backgroundColor, self.rectList[self.screenCounter])  # clear
                self.addAngles()
                self.rotateAndTransformPlanar()
                self.draw(active_screen, self.objColor)
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

            if pygame.time.get_ticks() > self.start_timer + 1000 * self.run_seconds:
                self.running = False

        return self.stop

    def addAngles(self):

        # rotate by changing "angles".
        if self.frameCount < 200:
            # in the beginning, rotate arounf Z axis only.
            self.angles[2] += self.angleAdd[2] / self.nrImagesPerFrame
        else:
            self.angles += self.angleAdd / self.nrImagesPerFrame

        for j in range(3):
            if self.angles[j] > 360:
                self.angles[j] -= 360
            elif self.angles[j] < 0:
                self.angles[j] += 360

    def rotateAndTransformPlanar(self):

        # PLANAR: Assumes Z coordinate for all nodes is zero. Hence no need for row 3 of the matrix and a simpler dot product.
        # Set matrix for rotation and rotate and transform nodes.
        (sx, sy, sz) = np.sin(self.angles * self.angleScale)
        (cx, cy, cz) = np.cos(self.angles * self.angleScale)
        # build a matrix for X, Y, Z rotation (in that order, see Wikipedia: Euler angles).
        matrix = np.array([
            [cy * cz               , -cy * sz              , sy      ],
            [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx] #,
            # [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]
            ])

        # rotate nodes - Planar
        self.rotatedNodes = np.matmul(self.nodes[:, 0:2], matrix)
        # transform 3D to 2D by dividing X and Y with Z coordinate
        self.transNodes = (self.rotatedNodes[:, 0:2] / ((self.rotatedNodes[:, 2:3] + self.zPos)) * self.zScale) + self.midScreen

    def draw(self, screen, color):

        # draw the rotated object
        i = 0
        color = self.objColor
        for surface in self.surfaces:
            node_list = ([self.transNodes[node][:2] for node in surface])
            rect = self.drawPolygon(screen, color, node_list)
            if i == 0:
                # first and largest shape: store its Rect
                self.rectList[self.screenCounter] = rect
                i = 1
            # switch color for next surface
            if color == self.objColor:
                color = self.backgroundColor
            else:
                color = self.objColor

    def blend(self, screen, rect, mode):

        # first test if rect is something to process
        if rect is not None:
            # blit a surface to the "complete image" surface, additive (the newest) or subtract (the oldest) mode
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
        self.rectUsed = rect

        # blit the screen copy to the visible screen.
        while self.screenCopy.get_locked():
            self.screenCopy.unlock()
        while self.screen.get_locked():
            self.screen.unlock()
        self.screen.blit(self.screenCopy, rect, rect)

        # subtract colors to get the "red first, then yellow"
        self.screen.blit(self.screenColorize, rect, rect, pygame.BLEND_SUB)
        # double the colors to get the final blends
        self.screen.blit(self.screen, rect, rect, pygame.BLEND_ADD)

        # the ~same can be achieved using a more strictly defined 256 color palette:
        # x0 = rect[0]
        # x1 = rect[0] + rect[2]
        # y0 = rect[1]
        # y1 = rect[1] + rect[3]
        # # use surfarray.pixels3d to process image as a numpy array
        # rgb_array = pygame.surfarray.pixels3d(self.screen)
        # # pick a new color from the 256 color palette for each pixel, based on its shade (0 to 255) of RED color
        # rgb_array[x0:x1, y0:y1, 0:3] = self.palette[rgb_array[x0:x1, y0:y1, 0]]

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

        self.plotInfoBlit(10, 0 * 15 + 10, "frame  " + ("     " + str(self.frameCount))[-6:])
        self.plotInfoBlit(10, 1 * 15 + 10, "angles " + ("     " + str(int(self.angles[0])))[-6:]
                          + ("     " + str(int(self.angles[1])))[-6:] + ("     " + str(int(self.angles[2])))[-6:])
        self.plotInfoBlit(10, 2 * 15 + 10, "rect   " + ("     " + str(int(self.rectUsed[0])))[-6:]
                          + ("     " + str(int(self.rectUsed[1])))[-6:] + ("     " + str(int(self.rectUsed[2])))[-6:]
                          + ("     " + str(int(self.rectUsed[3])))[-6:])

        # add Frames Per Second - avg frame rate using the last ten frames
        self.plotInfoBlit(10, 3 * 15 + 10, "fps    " + ("     " + str(round(self.clock.get_fps(), 1)))[-6:])

        # add measured times as percentage of total
        tot_time = np.sum(self.timers)
        if tot_time > 0:
            for i in range(len(self.timer_names)):
                info_msg = (self.timer_names[i] + ' '*16)[:16] + (' '*10 + str(round(np.sum(self.timers[i, :]) * 100 / tot_time, 1)))[-7:]
                self.plotInfoBlit(10, (i + 4) * 15 + 10, info_msg)

    def addBars(self):

        # draw horizontal bars
        bar_height = int(4.0 * self.height / 320)
        for i in range(bar_height):
            color_level = np.sin((i + 1) / (bar_height + 1) * np.pi)
            color = ([round(x * color_level, 0) for x in self.barColor])
            pygame.draw.line(self.screen, color, (0, self.height * 0.05 + i),
                             (self.width - 1, self.height * 0.05 + i))
            pygame.draw.line(self.screen, color, (0, self.height * 0.95 - i),
                             (self.width - 1, self.height * 0.95 - i))


    def plotInfoBlit(self, x, y, msg):

        f_screen = self.font.render(msg, False, [255, 255, 255])
        f_screen.set_colorkey(None)
        self.screen.blit(f_screen, (x, y))

    def toggleInfoDisplay(self):

        # switch between a windowed display and full screen
        if self.infoDisplay:
            self.infoDisplay = False
        else:
            self.infoDisplay = True

    def toggleFullScreen(self):

        # toggle between fulls creen and windowed mode
        pygame.display.toggle_fullscreen()
        self.addBars()

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
    # disp_modes = pygame.display.list_modes()
    # disp_size = disp_modes[9]  # selecting display size from available list. Assuming the 9th element is nice...
    # disp_size = (800, 600)  # to force display size
    # disp_size = (1280, 800)  # to force display size
    disp_size = (1920, 1080)  # to force display size
    pygame.font.init()

    # initialize mixer
    pygame.mixer.init()
    music_file = "firepower.ogg"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/release.php?id=45537
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play()

    screen = pygame.display.set_mode(disp_size)
    pygame.display.set_caption('Milky Way')
    MilkyWay(screen, 60, 120).run()

    # exit; close everything
    pygame.quit()
    exit()
