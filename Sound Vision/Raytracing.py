# -*- coding: utf-8 -*-
import pygame
import numpy as np
from os import chdir
from sys import exit


class Raytracing:
    """
    Displays an animated raytracing.

    @author: kalle
    """

    def __init__(self, screen, target_fps):
        self.screen = screen
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.screenSize = (self.width, self.height)
        self.midScreen = (int(self.width / 2), int(self.height / 2))
        self.backgroundColor = (0, 0, 0)
        self.barColor = (100, 40, 220)
        self.logoFileName = "Reflect.jpg"
        self.logoSize = (0, 0)
        self.logoPosition = (0, 0)
        self.scrollSize = (0, 0)
        self.scrollPosition = 0.0
        self.imageFileName = "Raytracing_##.jpg"
        self.imageNr = 10
        self.imageSize = (0, 0)
        self.imagePosition = (0, 0)
        self.imageList = []
        self.imageListItem = 0
        self.frameCount = 30 * 50    # maximum nr of frames
        self.frameShowCount = 80
        self.frameNr = 0
        self.running = True
        self.stop = False
        self.target_fps = target_fps
        self.clock = pygame.time.Clock()

    def run(self):

        self.loadImages()
        self.generateScroller()
        self.copyLogo()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.frameNr < self.frameCount - self.frameShowCount:
                            self.frameNr = self.frameCount - min(self.frameNr, self.frameShowCount)  # start exiting
                        self.stop = True
                    if event.key == pygame.K_f:
                        self.toggleFullScreen()
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                                          self.__class__.__name__ + '.jpg')
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # left button
                        if self.frameNr < self.frameCount - self.frameShowCount:
                            self.frameNr = self.frameCount - min(self.frameNr, self.frameShowCount)  # start exiting

            self.animate()
            self.showScroller()
            self.frameNr += 1
            if self.frameNr > self.frameCount:
                self.running = False  # exit

            self.clock.tick(self.target_fps)  # this keeps code running at max target_fps
            # switch between currently showed and the next screen (prepared in "buffer")
            pygame.display.flip()

        return self.stop

    def animate(self):
        """
        Animate the raytracing images.
        """

        self.imageListItem += 1
        if self.imageListItem >= self.imageNr:
            self.imageListItem = 0
        # at beginning and end, show part of image only
        lines = int(self.imageSize[1] * min(1.0, min(self.frameNr, self.frameCount - self.frameNr) / self.frameShowCount))
        self.screen.blit(self.imageList[self.imageListItem], self.imagePosition, (0, 0, self.imageSize[0], lines))
        if lines < self.imageSize[1]:
            self.screen.fill(self.backgroundColor, (self.imagePosition[0], self.imagePosition[1] + lines, self.imageSize[0], self.imageSize[1] - lines))

    def showScroller(self):

        if self.frameNr > self.frameShowCount:
            if self.scrollPosition <= self.scrollSize[1] - self.imageSize[1]:
                self.scrollPosition += self.imageSize[1] / self.frameShowCount
                lines = int(self.imageSize[1] * min(1.0, (self.frameCount - self.frameNr) / self.frameShowCount))
                self.screen.blit(self.screenScroll, (self.imagePosition[0] + self.imageSize[0] + 2, self.imagePosition[1]),
                                 (0, int(self.scrollPosition), self.scrollSize[0], lines))
                if lines < self.imageSize[1]:
                    self.screen.fill(self.backgroundColor, (self.imagePosition[0] + self.imageSize[0] + 2, self.imagePosition[1] + lines,
                                                            self.scrollSize[0], self.imageSize[1] - lines))
            else:
                # start exiting
                if self.frameNr < self.frameCount - self.frameShowCount:
                    self.frameNr = self.frameCount - min(self.frameNr, self.frameShowCount)

    def loadImages(self):
        """
        Load the images and scale them for screen use.
        """

        # load and scale logo
        surf = pygame.image.load(self.logoFileName)
        surf.convert()
        # image should be maximum 1/3 of screen height or 4/5 of screen width
        image_size = surf.get_size()
        image_ratio = max(image_size[0] / (self.width * 4 / 5), image_size[1] / (self.height / 3))
        # scale image
        self.logo_size = (int(image_size[0] / image_ratio), int(image_size[1] / image_ratio))
        self.logoScreen = pygame.transform.scale(surf, self.logo_size)

        # load and scale raytracing images
        for i in range(self.imageNr):
            image_name = self.imageFileName.replace('##', ('00' + str(i + 1))[-2:])
            surf = pygame.image.load(image_name)
            surf.convert()
            # image should be maximum 1/2 of screen height or 2/3 of screen width
            image_size = surf.get_size()
            image_ratio = max(image_size[0] / (self.width * 2 / 3), image_size[1] / (self.height / 2))
            # scale image
            self.imageSize = (int(image_size[0] / image_ratio), int(image_size[1] / image_ratio))
            self.imageList.append(pygame.transform.scale(surf, self.imageSize))
        self.imagePosition = (int((self.width - self.imageSize[0] * 3 / 2) / 2), int(self.height * 0.43))

    def generateScroller(self):

        # generate scroll screen with predefined scroll text
        # scroll width = 50 % of image width, height depends on font size and nr of text rows.
        scroller = [
            'some greetings',
            'from Zeque:',
            ' ',
            'Overlander',
            'a.k.a.',
            'Mikko',
            ' ',
            'Nighthawk',
            'a.k.a.',
            'Teemu',
            ' ',
            'Devil',
            'a.k.a.',
            'Janne',
            ' ',
            'Jellybean',
            'a.k.a.',
            'Vegard',
            ' ',
            'Red Baron',
            ' ',
            'Vibrillion'
        ]
        font_size = int(self.height / 15)

        color = (240, 240, 200)
        unsuccessful = True
        # loop through until scroller is successful - checking that the selected font size fits the scroller width
        while unsuccessful:
            unsuccessful = False  # if no problems, one loop is sufficient
            scroll_line = self.imageSize[1]
            self.font = pygame.font.SysFont('GillSansUltraCondensed,Arial', font_size)   # initialize font
            self.scrollSize = (int(self.imageSize[0] / 2), self.imageSize[1] * 2.1 + int(font_size * 1.05) * len(scroller))
            self.screenScroll = pygame.Surface(self.scrollSize)
            for stext in scroller:
                f_screen = self.font.render(stext, True, color, self.backgroundColor)
                if f_screen.get_size()[0] > 0.98 * self.scrollSize[0]:
                    # scroll text too wide. Make font smaller and try again.
                    font_size = min(int(font_size * 0.9), font_size - 1)
                    unsuccessful = True
                self.screenScroll.blit(f_screen, (int((self.scrollSize[0] - f_screen.get_size()[0]) / 2), scroll_line))
                scroll_line += int(font_size * 1.05)

        # set picture and scroller speed so that will be smoother
        self.frameShowCount = int(max(1, int(self.imageSize[1] / 300 + 0.5)) * self.target_fps)

    def copyLogo(self):

        # copy logo and add horizontal bars
        self.screen.blit(self.logoScreen, (int((self.width - self.logo_size[0]) / 2), int(self.height * 0.05)))

        # draw horizontal bars
        bar_height = int(6.0 * self.height / 320)
        for i in range(bar_height):
            color_level = np.sin((i + 1) / (bar_height + 1) * np.pi)
            color = ([round(x * color_level, 0) for x in self.barColor])
            pygame.draw.line(self.screen, color, (0, self.imagePosition[1] - bar_height * 2 + i),
                             (self.width - 1, self.imagePosition[1] - bar_height * 2 + i))
            pygame.draw.line(self.screen, color, (0, self.imagePosition[1] + self.imageSize[1] + bar_height + i),
                             (self.width - 1, self.imagePosition[1] + self.imageSize[1] + bar_height + i))

    def toggleFullScreen(self):

        # toggle between full screen and windowed mode
        pygame.display.toggle_fullscreen()
        self.copyLogo()


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
    # disp_size = (800, 600)
    pygame.font.init()

    screen = pygame.display.set_mode(disp_size)
    pygame.display.set_caption('Raytracing')
    pygame.font.init()

    Raytracing(screen, 60).run()

    # exit; close display, stop music
    pygame.quit()
    exit()
