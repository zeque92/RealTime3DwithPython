# -*- coding: utf-8 -*-
import pygame
import numpy as np
from scipy.interpolate import interp1d
from sys import exit


class ShadowBobs:
    """
    Displays shadow bobs.

    @author: kalle
    """

    def __init__(self, screen, target_fps):
        self.screen = screen
        self.screenCopy = self.screen.copy()
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.font = pygame.font.SysFont('CourierNew', 14)               # initialize font and set font size
        self.clock = pygame.time.Clock()
        self.backgroundColor = (0, 0, 0)
        self.bobRadius = int(min(self.width, self.height) / 18)         # size of "bob" used
        self.bobSize = self.bobRadius * 2 - 1
        self.bobNr = 0
        self.bobCount = 500                                             # nr of bobs to add before colorizing
        self.bobPosition = np.zeros((1, 4))
        self.bobMoves = []
        self.bobMove = np.zeros((1, 4))
        self.palette = np.zeros((256, 3), dtype=np.uint8)
        self.palettes = np.zeros((0, 3), dtype=np.uint8)
        self.screenData = np.zeros((self.width, self.height), dtype=np.uint8)   # a numpy array storing the 8-bit (256) color for each pixel
        self.bobData = np.zeros((self.bobSize, self.bobSize), dtype=np.uint8)   # a numpy array storing the "bob" ie. filled circle
        self.midScreen = np.array([self.width / 2, self.height / 2], dtype=int)
        self.bobTimeLimit = 5500                                        # milliseconds for each set of bobs
        self.bobTimeFade = 600                                          # milliseconds to fade out (at the end of bobTimeLimit)
        self.bobTime = 0
        self.target_fps = target_fps
        self.running = True
        self.stop = False
        self.infoDisplay = False
        self.millisecs = 0
        self.timer_avg_frames = 180
        self.timer_names = []
        self.timers = np.zeros((1, 1), dtype=int)
        self.timer_frame = 0
        self.start_timer = 0

        self.timer_names.append("blend bobs")
        self.timer_names.append("colorize")
        self.timer_names.append("plot info")
        self.timer_names.append("display flip")
        self.timer_names.append("wait")
        self.timers = np.zeros((len(self.timer_names), self.timer_avg_frames), dtype=int)

        self.setPalette()
        self.screen.fill(self.backgroundColor)

        # draw the BOB
        self.setUpBob()

    def run(self):
        """ Main loop. """

        # initialize timers
        self.start_timer = pygame.time.get_ticks()
        self.millisecs = self.start_timer
        self.bobTime = self.start_timer

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

            bobTimeUsed = pygame.time.get_ticks() - self.bobTime
            if bobTimeUsed >= self.bobTimeLimit:
                # next bob setup
                self.bobNr += 1
                if self.bobNr >= len(self.bobMoves) or self.bobNr >= np.shape(self.palettes)[0] / 256:
                    self.running = False  # out of bob moves or palettes
                else:
                    # clear data
                    self.screenData = np.zeros((self.width, self.height), dtype=np.uint8)
                    # get new palette and bob move data
                    self.palette = self.palettes[self.bobNr * 256: (self.bobNr + 1) * 256, :]
                    self.bobMove = np.array(self.bobMoves[self.bobNr])
                    self.bobPosition = np.zeros((np.shape(self.bobMove)))
                    self.bobTime = pygame.time.get_ticks()
            elif bobTimeUsed >= self.bobTimeLimit - self.bobTimeFade:
                # adjust palette to fade out at the end
                self.palette = (self.palettes[self.bobNr * 256: (self.bobNr + 1) * 256, :]
                                * ((self.bobTimeLimit - bobTimeUsed) / self.bobTimeFade)).astype(np.uint8)

            for i in range(0, self.bobCount, np.shape(self.bobPosition)[0]):
                # move the bobs and add them
                self.bobPosition += self.bobMove
                for j in range(np.shape(self.bobPosition)[0]):
                    self.addBob(j)

            self.measureTime("blend bobs")

            # copy finished data to the visible screen and update the colors
            self.colorize()
            self.measureTime("colorize")

            if self.infoDisplay:
                self.plotInfo()
                self.measureTime("plot info")
            self.nextTimeFrame()

            pygame.display.flip()
            self.measureTime("display flip")
            self.clock.tick(self.target_fps)  # this keeps code running at max target_fps
            self.measureTime("wait")

        return self.stop

    def addBob(self, bob):

        screenPos = (self.midScreen + ((np.sin(self.bobPosition[bob, 0:2]) + np.cos(self.bobPosition[bob, 2:4])) / 2)
                     * (self.midScreen - [self.bobRadius, self.bobRadius]) - [self.bobRadius, self.bobRadius]).astype(np.int16)

        # add Bob. screenData will also automatically drop back to 0 after 255 (as is uint8)
        self.screenData[screenPos[0]:screenPos[0] + self.bobSize, screenPos[1]:screenPos[1] + self.bobSize] += self.bobData

    def colorize(self):

        # get screen to a numpy array
        rgb_array = pygame.surfarray.pixels3d(self.screen)
        # pick a new color from the 256 color palette for each pixel, based on its color number (0 to 255) in screenData
        rgb_array[:, :, 0:3] = self.palette[self.screenData]

    def plotInfo(self):
        """
        Add info on screen.
        This can obviously be skipped, just for information.
        """

        # release any locks on screen
        while self.screen.get_locked():
            self.screen.unlock()

        self.plotInfoBlit(10, 0 * 15 + 10, "bobs/frame      " + ("      " + str(self.bobCount))[-7:])
        # add Frames Per Second - avg frame rate using the last ten frames
        self.plotInfoBlit(10, 1 * 15 + 10, "fps             " + ("      " + str(round(self.clock.get_fps(), 1)))[-7:])

        # add measured times as percentage of total
        tot_time = np.sum(self.timers)
        if tot_time > 0:
            for i in range(len(self.timer_names)):
                info_msg = (self.timer_names[i] + ' '*16)[:16] + (' '*10 + str(round(np.sum(self.timers[i, :]) * 100 / tot_time, 1)))[-7:]
                self.plotInfoBlit(10, (i + 2) * 15 + 10, info_msg)

    def plotInfoBlit(self, x, y, msg):

        f_screen = self.font.render(msg, False, [255, 255, 255])
        f_screen.set_colorkey(None)
        self.screen.blit(f_screen, (x, y))

    def setUpBob(self):

        # set up the bob. Simply draw a filled circle.
        for i in range(self.bobSize):
            for j in range(self.bobSize):
                # if i,j distance from center smaller than circle radius, set to 1, else 0
                if (i - self.bobRadius + 1) ** 2 + (j - self.bobRadius + 1) ** 2 < self.bobRadius ** 2:
                    self.bobData[i, j] = 1
                else:
                    self.bobData[i, j] = 0

    def setPalette(self):

        # generate palettes. These 256 color palettes will be used for object colors even if display is full color.
        palettes = ([
            [[  0,  0,  0],
             [ 16,  0,  0],
             [ 48,  0,  0],
             [ 80,  0,  0],
             [ 96, 16,  0],
             [112, 32,  0],
             [128, 48,  0],
             [144, 64,  0],
             [160, 80,  0],
             [176, 96,  0],
             [192,112,  0],
             [208,128,  0],
             [224,160,  0],
             [208,176,  0],
             [208,208,  0],
             [224,224,  0],
             [240,240,  0],
             [224,240, 16],
             [208,240, 32],
             [192,240, 48],
             [176,240, 64],
             [160,240, 80],
             [144,240, 96],
             [128,240,112],
             [112,240, 96],
             [ 96,224, 80],
             [ 80,208, 64],
             [ 64,192, 64],
             [ 64,176, 48],
             [ 64,160, 32],
             [ 64,144, 16],
             [ 64,128,  0],
             [ 48, 96,  0],
             [ 24, 48,  0],
             [  0,  0,  0]],
            [[  0,  0,  0],
             [ 48, 32, 16],
             [ 32, 32,  0],
             [ 80, 16, 16],
             [ 96, 16, 32],
             [112, 16, 48],
             [128, 16, 64],
             [144, 16, 80],
             [160, 16,112],
             [160,  0,128],
             [176,  0,160],
             [192,  0,176],
             [208,  0,192],
             [224,  0,176],
             [208,  0,144],
             [224,  0,128],
             [208,  0, 96],
             [224, 16, 80],
             [208, 16, 64],
             [224, 16, 48],
             [208, 32, 48],
             [224, 80, 48],
             [208, 96, 32],
             [208,160, 32],
             [224,144, 48],
             [240,128, 64],
             [240,192, 64],
             [224,208, 48],
             [208,240, 48],
             [192,224, 48],
             [176,240, 48],
             [160,224, 32],
             [144,208, 32],
             [128,208, 16],
             [112,224, 16],
             [ 80,208, 16],
             [ 64,192,  0],
             [ 16,176,  0],
             [  0,160,  0],
             [  0,144, 16],
             [  0,128, 32],
             [  0,112, 48],
             [  0, 96, 32],
             [  0, 64, 24],
             [  0, 32, 12],
             [  0,  0,  0]],
            [[  0,  0,  0],
             [  0,  0, 32],
             [  0,  0, 64],
             [ 16,  0, 96],
             [ 32,  0,112],
             [ 48,  0,128],
             [ 64,  0,144],
             [ 80,  0,160],
             [ 96,  0,176],
             [112,  0,192],
             [128,  0,208],
             [144,  0,224],
             [160,  0,240],
             [176,  0,224],
             [192,  0,208],
             [176,  0,192],
             [160,  0,176],
             [160, 16,160],
             [144, 32,160],
             [144, 48,144],
             [128, 64,128],
             [112, 96,112],
             [ 96,112, 96],
             [ 80,128, 80],
             [ 64,144, 64],
             [ 48,128, 48],
             [ 32,112, 32],
             [  0, 80,  0],
             [  0,  0,  0]],
            [[  0,  0,  0],
             [  0, 32,  0],
             [  0, 64,  0],
             [ 16, 80,  0],
             [  0, 96,  0],
             [ 16,112, 16],
             [ 32,128, 32],
             [ 48,128, 48],
             [ 64,144, 64],
             [ 80,160, 64],
             [ 96,176, 64],
             [112,192, 64],
             [128,208, 64],
             [144,224, 64],
             [160,240, 64],
             [176,240, 48],
             [192,240, 16],
             [208,224, 16],
             [224,208, 32],
             [208,192, 32],
             [208,160, 48],
             [208,144, 64],
             [192,128, 80],
             [176,112, 96],
             [160, 96, 96],
             [160, 80,112],
             [144, 64,128],
             [128, 48,144],
             [112, 32,144],
             [ 96, 16,144],
             [ 80,  0,128],
             [ 64,  0,112],
             [ 48,  0, 96],
             [ 32,  0, 64],
             [  0,  0,  0]],
            [[  0,  0,  0],
             [ 48,  0,  0],
             [ 80,  0,  0],
             [112,  0, 16],
             [128,  0, 48],
             [144,  0, 64],
             [160,  0, 80],
             [176,  0, 96],
             [192, 16,112],
             [208, 32,128],
             [224, 48,144],
             [224, 80,160],
             [224, 96,176],
             [208,112,192],
             [192,128,208],
             [176,144,208],
             [176,160,224],
             [160,176,224],
             [144,208,224],
             [112,224,224],
             [192,144,224],
             [ 96, 72,112],
             [  0,  0,  0]],
            [[  0,  0,  0],
             [ 48, 48,  0],
             [ 80, 48,  0],
             [ 96, 64,  0],
             [112, 96,  0],
             [144,112,  0],
             [176,128,  0],
             [160,144,  0],
             [176,160, 32],
             [192,192, 32],
             [224,208, 32],
             [240,192, 32],
             [240,176, 16],
             [240,160,  0],
             [224,144,  0],
             [208,128,  0],
             [160,128, 16],
             [112, 80, 16],
             [ 80, 80,  0],
             [ 64, 64,  0],
             [ 64, 32,  0],
             [ 32, 16,  0],
             [  0,  0,  0]],
            [[  0,  0,  0],
             [ 32,  0, 48],
             [ 48,  0, 32],
             [ 64,  0, 64],
             [ 80,  0, 80],
             [112,  0, 96],
             [128,  0,112],
             [144,  0, 96],
             [176,  0, 96],
             [192,  0,112],
             [208,  0, 80],
             [224,  0, 64],
             [224,  0, 32],
             [240,  0, 16],
             [224,  0,  0],
             [240, 16,  0],
             [224, 32,  0],
             [224, 80,  0],
             [240, 96,  0],
             [240,112, 16],
             [240,128,  0],
             [240,144, 16],
             [240,160,  0],
             [224,176,  0],
             [240,192, 16],
             [240,208,  0],
             [240,240, 16],
             [224,144,  0],
             [208,112,  0],
             [160, 64,  0],
             [ 96, 48, 16],
             [ 48, 24, 16],
             [  0,  0,  0]],
            ])

        self.palettes = np.zeros((0, 3), dtype=np.uint8)
        for palette in palettes:
            pal_nodes = np.array(palette)
            # out of palette nodes, build a full palette using interpolation.
            nodes = np.arange(0, np.shape(pal_nodes)[0], 1) * 255 / (np.shape(pal_nodes)[0] - 1)
            fx = interp1d(nodes, pal_nodes, kind='quadratic', axis=0)
            # data needed for the whole range of time series. Use speed = "time" per second.
            pal_data = fx(np.arange(0, 256, 1))
            # make sure palette stays within 0 and 255 boundaries, and is of correct type
            self.palettes = np.vstack((self.palettes, np.minimum(np.maximum(pal_data, np.zeros((256, 3))), np.ones((256, 3)) * 255).astype(np.uint8)))
        self.palette = self.palettes[0:256, :]

        # set up bob move in terms of radius (for sin, cos)
        qpi = np.pi / 4
        self.bobMoves = ([
            [[qpi * 2 + 0.0043, qpi * 2 + 0.0021, -0.0017, +0.00563],
             [qpi * 2 - 0.0023, qpi * 2 + 0.0038, +0.0031, -0.00141]],
            [[qpi * 0 + 0.0043, qpi * 0 + 0.0021, -0.0017, +0.00563],
              [qpi * 0 - 0.0023, qpi * 0 + 0.0038, +0.0031, -0.00141]],
            [[qpi * 2 + 0.0008, qpi * 0 + 0.0011, -0.00063, +0.00063],
              [qpi * 0 - 0.0006, qpi * 2 + 0.0004, +0.00055, -0.00041],
              [qpi * 2 - 0.0005, qpi * 0 + 0.0009, -0.00031, -0.00061]],
            [[qpi * 1 + 0.0013, qpi * 0 + 0.0011, -0.0031, +0.00077],
              [qpi * 3 - 0.0003, qpi * 0 + 0.0029, -0.0007, -0.00041]],
            # [[qpi * 2 - 0.0093, qpi * 1 - 0.0051, +0.0027, -0.00363]],
            # [[qpi * 1 - 0.0123, qpi * 1 + 0.0168, +0.0022, -0.00188]],
            [[qpi * 2 + 0.0013, qpi * 1 + 0.0031, -0.0007, +0.00754],
              [qpi * 3 - 0.0027, qpi * 2 + 0.0018, +0.0043, -0.00087]],
            [[qpi * 1 + 0.0043, qpi * 2 + 0.0011, -0.0089, +0.00333],
             [qpi * 2 - 0.0011, qpi * 3 + 0.0028, +0.0013, -0.00291]],
            [[qpi * 0 + 0.0008, qpi * 0 + 0.0011, -0.00063, +0.00071],
             [qpi * 0 - 0.0006, qpi * 0 + 0.0014, +0.00055, -0.00041],
             [qpi * 0 - 0.0005, qpi * 0 + 0.0009, -0.00031, -0.00061]]
            ])
        self.bobMove = np.array(self.bobMoves[0])
        self.bobPosition = np.zeros((np.shape(self.bobMove)))

    def toggleInfoDisplay(self):

        # switch between a windowed display and full screen
        if self.infoDisplay:
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


if __name__ == '__main__':
    """
    Prepare screen, etc.
    """

    # set screen size
    # first check available full screen modes
    pygame.display.init()
    # pick disp0lay mode from list or set a specific resolution
    disp_modes = pygame.display.list_modes()
    # disp_size = disp_modes[0]  # selecting display size from available list. Assuming the 9th element is nice...
    # disp_size = (800,600)  # to force display size
    # disp_size = (1280, 800)  # to force display size
    disp_size = (1920,1080) # to force display size
    pygame.font.init()

    # initialize mixer
    pygame.mixer.init()
    music_file = "firepower.ogg"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/release.php?id=45537
    # start music player
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play()

    pygame.display.set_caption('Shadow Bobs')
    screen = pygame.display.set_mode(disp_size)

    ShadowBobs(screen, 30).run()

    # exit; close everything
    pygame.quit()
    exit()
