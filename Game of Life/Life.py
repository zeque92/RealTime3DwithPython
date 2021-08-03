# -*- coding: utf-8 -*-
import pygame
import numpy as np
from sys import exit


class Life:
    """
    Game of Life.

    @author: kalle
    """

    def __init__(self, screen, target_fps):

        self.screen = screen
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.background_color = (0, 0, 0)
        self.color = (200, 200, 255)
        self.use_color = self.color[0] * 256 ** 2 + self.color[1] * 256 + self.color[2]
        self.target_fps = target_fps
        self.running = True
        self.paused = False
        self.clock = pygame.time.Clock()
        self.start_timer = pygame.time.get_ticks()

        self.life_probability = 0.23  # probability of any single cell being alive at the beginning.
        self.rng = np.random.default_rng()  # initiate without seed
        # self.random_seed = 12345678  # random seed for reproducing a "game"
        # self.rng = np.random.default_rng(self.random_seed)  # initiate with seed
        self.generation = 0
        self.life_amount = 0
        self.fade_cnt = 60

        # the following for checking performance only
        self.info_display = True
        self.millisecs = 0
        self.timer_avg_frames = 180
        self.timer_names = []
        self.timers = np.zeros((1, 1), dtype=int)
        self.timer_frame = 0
        self.start_timer = 0
        self.font = pygame.font.SysFont('CourierNew', 15)

        # set up timers
        self.timer_name = []
        self.timer_names.append("calculate")
        self.timer_names.append("plot life")
        self.timer_names.append("plot info")
        self.timer_names.append("display flip")
        self.timer_names.append("wait")
        self.timers = np.zeros((len(self.timer_names), self.timer_avg_frames), dtype=int)

        self.setup_life_array()

        # initialize timers
        self.start_timer = pygame.time.get_ticks()
        self.millisecs = self.start_timer

    def run(self):
        """
        Main loop.
        """
        self.timer = pygame.time.get_ticks()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.key == pygame.K_f:
                        self.toggle_fullscreen()
                    if event.key == pygame.K_n:
                        self.fade_cnt -= 1  # start fading out
                    if event.key == pygame.K_SPACE:
                        self.pause()
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                                          self.__class__.__name__ + '.jpg')
                    if event.key == pygame.K_i:
                        self.toggle_info_display()
                # elif event.type == pygame.MOUSEBUTTONDOWN:
                #     if event.button == 1:
                #         # left button: exit
                #         self.running = False

            if self.paused:
                pygame.time.wait(100)
                self.millisecs = pygame.time.get_ticks()

            else:
                # main components executed here
                self.fade_out()
                self.new_generation()
                self.measure_time("calculate")
                self.plot_life()
                self.measure_time("plot life")
                if self.info_display:
                    self.plot_info()
                    self.measure_time("plot info")

            # release any locks on screen
            while self.screen.get_locked():
                self.screen.unlock()

            # switch between currently showed and the next screen (prepared in "buffer")
            pygame.display.flip()
            self.measure_time("display flip")
            self.clock.tick(self.target_fps)  # this keeps code running at max target_fps
            self.measure_time("wait")
            self.next_time_frame()

    def fade_out(self):

        # fade out and setup new life IF fading started

        if self.fade_cnt <= 0:
            self.setup_life_array()
            self.use_color = self.color[0] * 256 ** 2 + self.color[1] * 256 + self.color[2]
        elif self.fade_cnt < 60:
            fade = self.fade_cnt / 60
            self.use_color = int(self.color[0] * fade) * 256 ** 2 + int(self.color[1] * fade) * 256 + int(self.color[2] * fade)
            self.fade_cnt -= 1

    def new_generation(self):

        # calculate the next generation.

        self.generation += 1
        (w, h) = (self.width, self.height)

        # calculate the number of neighbours + cell itself so that the result is always between 0 and 9.
        # life_array has one extra row and column on both sides of screen area for "continuing" on the opposite side.
        nb_array = self.life_array[1:w + 1, 1:h + 1] \
            + self.life_array[1:w + 1, 0:h] \
            + self.life_array[1:w + 1, 2:h + 2] \
            + self.life_array[0:w, 1:h + 1] \
            + self.life_array[0:w, 0:h] \
            + self.life_array[0:w, 2:h + 2] \
            + self.life_array[2:w + 2, 1:h + 1] \
            + self.life_array[2:w + 2, 0:h] \
            + self.life_array[2:w + 2, 2:h + 2]

        # apply the rules:
        #   1. if cell is alive and has 2 or 3 live neighbours, it stays alive.
        #   2. if a dead cell has exactly 3 live neighours, it becomes live
        #   3. otherwise, cell is/becomes dead.
        # translated to:
        #   A. if cell + neighbours count = 3 --> it is alive (either live + 2 neighbours (1) or dead + three neighbours (2))
        #   B. if cell + neighbours count = 4 --> it stays as it is (either live + 3 neighbours (1) or dead + four neighbours (3))
        #   C. otherwise it is dead.
        self.life_array[1:w + 1, 1:h + 1][nb_array[:, :] == 3] = 1  # applying (A)   - (B) needs no action
        self.life_array[1:w + 1, 1:h + 1][(nb_array[:, :] < 3) | (nb_array[:, :] > 4)] = 0  # applying (C)

        # copy edge data from actual area other side
        self.life_array[0:1, :] = self.life_array[w:w + 1, :]
        self.life_array[w + 1:w + 2, :] = self.life_array[1:2, :]
        self.life_array[:, 0:1] = self.life_array[:, h:h + 1]
        self.life_array[:, h + 1:h + 2] = self.life_array[:, 1:2]

        check_freq = 30  # how often to check between this and the last check generation
        if self.generation % check_freq == 0:
            self.life_amount = (self.check_array != self.life_array).sum()  # differences = amount of life
            if self.life_amount == 0:
                # no changes, generate new life. check_freq should be an even number as many "almost dead forms" rotate every other frame.
                self.fade_cnt -= 1  # start fading out
            else:
                self.check_array = self.life_array.copy()

    def plot_life(self):

        # transfer life array to screen as rgb_array

        while self.screen.get_locked():
            self.screen.unlock()

        (w, h) = (self.width, self.height)
        rgb_array = pygame.surfarray.pixels2d(self.screen)
        rgb_array[:, :] = self.life_array[1:w + 1, 1:h + 1] * self.use_color

    def setup_life_array(self):

        # setup the initial array of zeroes (dead) and ones (alive) at the shape and size of screen + extra row & column at both ends

        (w, h) = (self.width, self.height)
        self.life_array = (self.rng.random((w + 2, h + 2)) + self.life_probability).astype(np.uint8)
        # copy edge data from actual area other side
        self.life_array[0:1, :] = self.life_array[w:w + 1, :]
        self.life_array[w + 1:w + 2, :] = self.life_array[1:2, :]
        self.life_array[:, 0:1] = self.life_array[:, h:h + 1]
        self.life_array[:, h + 1:h + 2] = self.life_array[:, 1:2]

        self.check_array = self.life_array.copy()
        self.generation = 0
        self.life_amount = 0
        self.fade_cnt = 60

    def toggle_fullscreen(self):

        # toggle between fullscreen and windowed mode
        pygame.display.toggle_fullscreen()

    def pause(self):

        if self.paused:
            self.paused = False
            self.timer += pygame.time.get_ticks() - self.pause_timer  # adjust timer for pause time
        else:
            self.paused = True
            self.pause_timer = pygame.time.get_ticks()

    def toggle_info_display(self):

        # switch between a windowed display and full screen
        if self.info_display:
            self.info_display = False
        else:
            self.info_display = True

    def plot_info(self):

        # show info on object and performance
        while self.screen.get_locked():
            self.screen.unlock()

        self.plot_info_msg(self.screen, 10, 10, 'frames per sec: ' + str(int(self.clock.get_fps())))
        self.plot_info_msg(self.screen, 10, 25, 'generation nr:  ' + str(int(self.generation)))
        self.plot_info_msg(self.screen, 10, 40, 'life amount:    ' + str(int(self.life_amount)))

        # add measured times as percentage of total
        tot_time = np.sum(self.timers)
        if tot_time > 0:
            for i in range(len(self.timer_names)):
                info_msg = (self.timer_names[i] + ' '*16)[:16] + (' '*10 + str(round(np.sum(self.timers[i, :]) * 100 / tot_time, 1)))[-7:]
                self.plot_info_msg(self.screen, 10, 60 + i * 15, info_msg)

        self.plot_info_msg(self.screen, 10, 80 + i * 15, 'info on/off:    i')
        self.plot_info_msg(self.screen, 10, 95 + i * 15, 'full screen:    f')
        self.plot_info_msg(self.screen, 10, 110 + i * 15, 'new breed:      n')
        self.plot_info_msg(self.screen, 10, 125 + i * 15, 'pause:          SPACE')
        self.plot_info_msg(self.screen, 10, 140 + i * 15, 'exit:           ESC')

    def plot_info_msg(self, screen, x, y, msg):
        f_screen = self.font.render(msg, False, (255, 255, 255))
        f_screen.set_colorkey(self.background_color)
        screen.blit(f_screen, (x, y))

    def measure_time(self, timer_name):

        # add time elapsed from previous call to selected timer
        i = self.timer_names.index(timer_name)
        new_time = pygame.time.get_ticks()
        self.timers[i, self.timer_frame] += (new_time - self.millisecs)
        self.millisecs = new_time

    def next_time_frame(self):

        # move to next timer and clear data
        self.timer_frame += 1
        if self.timer_frame >= self.timer_avg_frames:
            self.timer_frame = 0
        self.timers[:, self.timer_frame] = 0


if __name__ == '__main__':
    """
    Prepare screen, objects etc.
    """

    # set screen size
    # first check available full screen modes
    pygame.display.init()
    # disp_modes = pygame.display.list_modes(0, pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
    # disp_size = disp_modes[0] # selecting display size from available list.
    # disp_size = (1920, 1080)
    disp_size = (1280, 720)
    # disp_size = (800, 600)
    # disp_size = (640, 480)

    pygame.font.init()
    # pygame.mixer.init()
    # music_file = "alacrity.mod"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/
    # pygame.mixer.music.load(music_file)
    # pygame.mixer.music.play(loops=-1)

    screen = pygame.display.set_mode(disp_size, pygame.FULLSCREEN | pygame.DOUBLEBUF)
    pygame.display.set_caption('Life')
    Life(screen, 60).run()

    # exit; close display, stop music
    pygame.quit()
    exit()
