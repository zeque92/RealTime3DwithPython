# -*- coding: utf-8 -*-
import pygame
import numpy as np
from os import chdir
from sys import exit


class TheStars:
    """
    Displays a moving star field.

    @author: kalle
    """

    def __init__(self, screen, target_fps):
        self.screen = screen
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.screenSize = (self.width, self.height)
        self.midScreen = (int(self.width / 2), int(self.height / 2))
        self.screenTitles = pygame.Surface((self.width, 2 * self.height))  # create a surface for title texts
        self.font = pygame.font.SysFont('CourierNew', 14)   # initialize font and set font size
        self.fontTitles = pygame.font.SysFont('GillSansUltraCondensed,Arial', int(self.height / 12 - 1))   # initialize font for Titles
        self.titles = (
            ('Amiga Coding by:', 'Overlander', (238, 170, 85)),
            ('', 'Zeque', (238, 170, 85)),
            ('Python Coding by:', 'Zeque', (85, 170, 85)),
            ('Graphics by:', 'Nighthawk', (238, 238, 238)),
            ('', 'Red Baron', (238, 238, 238)),
            ('Music by:', 'Jellybean', (170, 0, 85))
            )
        self.titleData = []
        self.backgroundColor = (0, 0, 0)
        self.nrStars = 1
        self.zRange = (50, 2000)                           # range for Z coordinates of stars
        self.stars = np.zeros((0, 3))
        self.angles = np.zeros((0, 3))
        self.movementAvg = 60
        self.movement = np.zeros((0, 6), dtype=float)
        self.currMove = np.zeros((0, 5), dtype=float)
        self.moveNr = 0
        self.moveFrames = 0
        self.clock = pygame.time.Clock()
        self.target_fps = target_fps                       # sets maximum refresh rate
        self.running = True
        self.stop = False
        # the following for checking performance only
        self.infoDisplay = False
        self.paused = False
        self.millisecs = 0
        self.timer_avg_frames = 60
        self.timer_names = []
        self.timers = np.zeros((1, 1), dtype=int)
        self.timer_frame = 0
        self.start_timer = 0

        # initialize timers
        self.start_timer = pygame.time.get_ticks()
        self.millisecs = self.start_timer

        # set up timers
        self.timer_names.append("clear")
        self.timer_names.append("move")
        self.timer_names.append("rotate")
        self.timer_names.append("draw")
        self.timer_names.append("add titles")
        self.timer_names.append("plot info")
        self.timer_names.append("display flip")
        self.timer_names.append("wait")
        self.timers = np.zeros((len(self.timer_names), self.timer_avg_frames), dtype=int)

        # set number of stars depending on screen size
        self.nrStars = int(self.width * self.height / 100)

        self.setupStars()
        self.screen.fill(self.backgroundColor)

    def run(self):

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
                    if event.key == pygame.K_SPACE:
                        if self.paused:
                            self.paused = False
                        else:
                            self.paused = True
                    if event.key == pygame.K_i:
                        if self.infoDisplay:
                            self.infoDisplay = False
                        else:
                            self.infoDisplay = True
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                                          self.__class__.__name__ + '.jpg')
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # left button: exit
                        self.running = False

            if not self.paused:
                self.getMovement()
                self.moveStars()
            self.measureTime("move")
            self.plotStars()
            self.addTitles()

            if self.infoDisplay:
                self.plotInfo()

            # switch between currently showed and the next screen (prepared in "buffer")
            pygame.display.flip()
            self.measureTime("display flip")
            self.clock.tick(self.target_fps)  # this keeps code running at max target_fps
            self.measureTime("wait")

        return self.stop

    def setupStars(self):

        # set up a random batch of stars and the title texts
        self.stars = np.random.rand(self.nrStars, 3) * np.array([self.width - 2, self.height - 2, 1.0]) \
            + np.array([-self.midScreen[0], -self.midScreen[1], 0.0])
        # adjust Z coordinates as more stars needed at distance for a balanced view
        self.stars[:, 2] = (self.stars[:, 2] ** 0.5) * (self.zRange[1] - self.zRange[0]) + self.zRange[0]

        self.screenTitles.fill(self.backgroundColor)
        self.screenTitles.set_colorkey(self.backgroundColor)
        self.screenTitles.set_alpha(0)
        y = 0
        y_pos = int(-self.height / 6)
        for title in self.titles:
            # plot titles and add to titleData: source surface, destination position, source rect. Second item: center align
            if title[0] != '':
                # new item
                y_pos += int(self.height / 4)
                y_add = 0
                size = self.plotTitle(title[0], title[2], (0, y))
                self.titleData.append((self.screenTitles,
                                       (int(self.width / 8), y_pos),
                                       (0, y, size[0], size[1])
                                       ))
                y += size[1] + 1
            else:
                # just add another name
                y_add = size[1]  # this is the previous row height
            size = self.plotTitle(title[1], title[2], (0, y))
            self.titleData.append((self.screenTitles,
                                   (int(self.width * 3 / 4 - size[0] / 2), y_pos + y_add),
                                   (0, y, size[0], size[1])
                                   ))
            y += size[1] + 1

        # set movement data: X add, Y add, Z add, angle add, show titles, frames
        self.movement = np.array([
            [0, 0, 0, 0, 0, 120],
            [0, 0, -8, 0, 0, 360],
            [-2700, 0, 0, 0, 0, 120],
            [0, 0, 6, 0, 1, 60],
            [2700, 0, 0, 0, 1, 220],
            [2700, 2500, 0, 0, 1, 60],
            [0, 2500, 0, 0, 1, 60],
            [-2700, 2500, 0, 0, 1, 60],
            [-2700, 0, 0, 0, 1, 60],
            [-2700, -2500, 0, 0, 1, 60],
            [0, -2500, 0, 0, 1, 60],
            [2700, -2500, 0, 0, 1, 60],
            [2700, 0, 0, 0, 1, 60],
            [2700, 2500, 0, 0, 1, 60],
            [0, 2500, 0, 0, 1, 60],
            [-2700, 2500, 0, 0, 1, 60],
            [-2700, 0, 0, 0, 1, 260],
            [0, 0, -6, 0, 1, 120],
            [0, 0, -6, 0.03, 1, 360],
            [0, 0, 12, -0.05, 0, 480],
            [0, 0, 0, 0, 0, 0]
            ], dtype=float)

    def plotTitle(self, msg, color, coord):

        f_screen = self.fontTitles.render(msg, True, color, self.backgroundColor)
        self.screenTitles.blit(f_screen, coord)
        return f_screen.get_size()

    def getMovement(self):

        # use the movement array to define current movement pattern (X add, Yadd, Z add, angle add, title alpha)
        self.moveFrames += 1
        if self.moveFrames > self.movement[self.moveNr, 5]:
            self.moveFrames = 0
            self.moveNr += 1
            if self.moveNr >= np.shape(self.movement)[0]:
                self.moveNr = 0
                self.running = False

        # apply averaging when switching to new movement for smooth appearance.
        if self.moveFrames <= self.movementAvg:
            self.currMove = (self.moveFrames / self.movementAvg) * self.movement[self.moveNr, 0:5] \
                + (1.0 - self.moveFrames / self.movementAvg) * self.movement[self.moveNr - 1, 0:5]
            self.screenTitles.set_alpha(int(self.currMove[4] * 255))
        else:
            self.currMove = self.movement[self.moveNr, 0:5]

    def moveStars(self):

        # move the stars. If they go outside of screen or zRange, return them ... returning does not always work very well, though.

        # rotate stars around Z
        self.rotateStars(self.currMove[3])
        self.measureTime("rotate")

        star_move = self.currMove[0:3]
        # move stars in X,Y depending on their Z coordinate - the closer the faster / bigger move. Hence divide star_move X & Y by star Z
        self.stars += star_move / np.hstack((self.stars[:, 2:3], self.stars[:, 2:3], np.ones((self.nrStars, 1))))

        # return stars outside of X, Y, Z range to the other edge
        self.stars[:, 0][self.stars[:, 0] < -self.midScreen[0]] += self.width - 2
        self.stars[:, 0][self.stars[:, 0] > self.midScreen[0] - 2] -= self.width - 2
        self.stars[:, 1][self.stars[:, 1] < -self.midScreen[1]] += self.height - 5
        self.stars[:, 1][self.stars[:, 1] > self.midScreen[1] - 2] -= self.height - 2

        # move stars using Z coordinate and Z move
        self.stars[:, 0:2] *= self.stars[:, 2:3] / (self.stars[:, 2:3] + star_move[2])

        # Z is too far
        if self.moveNr >= np.shape(self.movement)[0] - 2 \
                and self.moveFrames > self.movement[self.moveNr, 5] - int((self.zRange[1] - self.zRange[0]) / (self.movement[self.moveNr, 2] + 0.0001)):
            # last stage - eliminate stars..
            self.stars[(self.stars[:, 2] > self.zRange[1])] = np.ones((np.shape(self.stars[(self.stars[:, 2] > self.zRange[1])])[0], 3)) \
                                                                      * np.array([0.0, 0.0, self.zRange[1]])
        else:
            # normally replace with a new random star at a random X, Y edge and random Z
            nr_half = int(self.nrStars / 2)
            # first half: vertical edge
            self.stars[0:nr_half, :][(self.stars[0:nr_half, 2] > self.zRange[1])] = np.hstack((
                np.random.randint(0, 2, (np.shape(self.stars[0:nr_half, :][(self.stars[0:nr_half, 2] > self.zRange[1])])[0], 1)) * (self.width - 2) - self.midScreen[0],
                np.random.rand(np.shape(self.stars[0:nr_half, :][(self.stars[0:nr_half, 2] > self.zRange[1])])[0], 1) * (self.height - 2) - self.midScreen[1],
                np.random.rand(np.shape(self.stars[0:nr_half, :][(self.stars[0:nr_half, 2] > self.zRange[1])])[0], 1) * (self.zRange[1] - self.zRange[0]) + self.zRange[0]
                ))
            # second half: horizontal edge
            self.stars[nr_half:, :][(self.stars[nr_half:, 2] > self.zRange[1])] = np.hstack((
                np.random.rand(np.shape(self.stars[nr_half:, :][(self.stars[nr_half:, 2] > self.zRange[1])])[0], 1) * (self.width - 2) - self.midScreen[0],
                np.random.randint(0, 2, (np.shape(self.stars[nr_half:, :][(self.stars[nr_half:, 2] > self.zRange[1])])[0], 1)) * (self.height - 2) - self.midScreen[1],
                np.random.rand(np.shape(self.stars[nr_half:, :][(self.stars[nr_half:, 2] > self.zRange[1])])[0], 1) * (self.zRange[1] - self.zRange[0]) + self.zRange[0]
                ))
        # if Z too close OR X, Y out of bounds due to Z move, replace with a new random star at maximum Z
        self.stars[(self.stars[:, 2] < self.zRange[0]) | (abs(self.stars[:, 0] + 1) > self.midScreen[0] - 1) | (abs(self.stars[:, 1] + 1) > self.midScreen[1] - 1)] \
            = np.random.rand(np.shape(self.stars[(self.stars[:, 2] < self.zRange[0]) | (abs(self.stars[:, 0] + 1) > self.midScreen[0] - 1) | (abs(self.stars[:, 1] + 1) > self.midScreen[1] - 1)])[0], 3) \
            * np.array([self.width - 2, self.height - 2, 0]) + np.array([-self.midScreen[0], -self.midScreen[1], self.zRange[1]])

    def plotStars(self):

        # clear screen
        self.screen.fill(self.backgroundColor)
        self.measureTime("clear")

        # setting colors pixel by pixel in a loop is very inefficient
        # for i in range(self.nrStars):
        #     color = int(255 - min(255, max(0, self.stars[i, 2] / (self.zRange[1] - self.zRange[0]) * 215 + 40)))
        #     self.screen.set_at((int(self.stars[i, 0] + self.midScreen[0]), int(self.stars[i, 1] + self.midScreen[1])), (color, color, color))

        while self.screen.get_locked():
            self.screen.unlock()

        # using a surfarray is way faster
        # define color as a function of distance
        colors = ((1.0 - self.stars[:, 2:3] / (self.zRange[1] - self.zRange[0])) * 200 + 55).astype(np.uint8)
        rgb_array = pygame.surfarray.pixels3d(self.screen)
        rgb_array[(self.stars[:, 0] + self.midScreen[0]).astype(np.int16), (self.stars[:, 1] + self.midScreen[1]).astype(np.int16), 0:3] \
            = np.hstack((colors, colors, colors))
        # add pixels to those which are closest (color is above a threshold)
        rgb_array[(self.stars[:, 0][colors[:, 0] > 150] + self.midScreen[0] + 1).astype(np.int16), (self.stars[:, 1][colors[:, 0] > 150] + self.midScreen[1]).astype(np.int16), 0:3] \
            = np.hstack((colors[colors[:, 0] > 150], colors[colors[:, 0] > 150], colors[colors[:, 0] > 150]))
        rgb_array[(self.stars[:, 0][colors[:, 0] > 200] + self.midScreen[0]).astype(np.int16), (self.stars[:, 1][colors[:, 0] > 200] + self.midScreen[1] + 1).astype(np.int16), 0:3] \
            = np.hstack((colors[colors[:, 0] > 200], colors[colors[:, 0] > 200], colors[colors[:, 0] > 200]))
        rgb_array[(self.stars[:, 0][colors[:, 0] > 230] + self.midScreen[0] + 1).astype(np.int16), (self.stars[:, 1][colors[:, 0] > 230] + self.midScreen[1] + 1).astype(np.int16), 0:3] \
            = np.hstack((colors[colors[:, 0] > 230], colors[colors[:, 0] > 230], colors[colors[:, 0] > 230]))

        self.measureTime("draw")

    def rotateStars(self, angle):

        # will rotate the stars around the Z axis. Note as modifies the original stars this is cumulative.
        if angle != 0.0:
            # set matrix for rotation using angles.
            sz = np.sin(angle)
            cz = np.cos(angle)
            # matrix for rotating around Z only is simple
            matrix = np.array([[cz, -sz],
                               [sz,  cz]])
            # apply to X and Y coordinates only.
            self.stars[:, 0:2] = np.dot(self.stars[:, 0:2], matrix)

    def addTitles(self):

        # add pre-plotted title texts
        self.screen.blits(self.titleData)
        self.measureTime("add titles")

    def plotInfo(self):
        """
        Add use instructions & data on screen.
        This can obviously be skipped, just for information.
        """

        # release any locks on screen
        while self.screen.get_locked():
            self.screen.unlock()

        pygame.draw.rect(self.screen, self.backgroundColor, (10, 10, 200, 180), 0)
        # add Frames Per Second
        fps = self.clock.get_fps()  # avg frame rate using the last ten frames
        self.plotInfoBlit(10,  10, ("fps" + ' '*16)[:16] + (' '*10 + str(round(fps, 1)))[-7:])
        tot_time = np.sum(self.timers)
        # add measured times as percentage of total
        if tot_time > 0:
            for i in range(len(self.timer_names)):
                self.plotInfoBlit(10, 25 + i * 15, (self.timer_names[i] + ' '*16)[:16]
                                  + (' '*10 + str(round(np.sum(self.timers[i, :]) * 100 / tot_time, 1)))[-7:])
        self.measureTime("plot info")

    def plotInfoBlit(self, x, y, msg):

        f_screen = self.font.render(msg, False, [255, 255, 255])
        self.screen.blit(f_screen, (x, y))

    def toggleFullScreen(self):

        # toggle between full screen and windowed mode
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
    # disp_size = disp_modes[0] # selecting display size from available list. Assuming the 9th element is nice...
    disp_size = (1920, 1080)  # to force display size
    pygame.font.init()

    screen = pygame.display.set_mode(disp_size)
    pygame.display.set_caption('The Stars')

    TheStars(screen, 60).run()

    # exit; close display, stop music
    pygame.quit()
    exit()
