# -*- coding: utf-8 -*-
import pygame
import numpy as np
from sys import exit


class Ball:
    """
    Rotating sphere with texture mapping.

    @author: kalle
    """

    def __init__(self, screen, target_fps):

        self.screen = screen
        self.width = self.screen.get_size()[0]
        self.height = self.screen.get_size()[1]
        self.mode = 1
        self.rounds = 3
        self.round = 0
        self.background_color = (0, 0, 0)
        self.screen_copy = screen.copy()
        self.screen_copy.set_colorkey((255, 255, 255))
        self.screen_copy_rect = pygame.Rect(100, 100, 1, 1)
        self.image = pygame.image.load('Normal_Mercator_map_85deg.jpg').convert()
        self.image_type = 'Mercator'
        self.image_array = pygame.surfarray.pixels2d(self.image)
        self.image_flat_x = np.zeros((0), dtype=float)
        self.image_flat_y = np.zeros((0), dtype=np.int16)
        self.image_size = np.asarray(self.image.get_size())
        self.image_mercator_R = 1.0 / np.log(np.tan(np.pi / 4 + (85.0 / 90.0) * np.pi / 4))  # with this R y is in [-1,1] between -85 and +85 degrees
        self.mid_screen = np.array([self.width // 2, self.height // 2], dtype=float)
        self.target_fps = target_fps
        self.running = True
        self.paused = False
        self.info_display = True
        self.clock = pygame.time.Clock()

        self.radius = self.height // 3
        self.point_nr = 3 * self.radius  # nr of points on the equator
        self.nodes = np.zeros((0, 3), dtype=float)
        self.nodes_flat = np.zeros((0, 3), dtype=float)
        self.nodes_flat_x = np.zeros((0), dtype=np.int16)
        self.nodes_flat_y = np.zeros((0), dtype=np.int16)
        self.rotated_nodes = np.zeros((0, 4), dtype=float)
        self.rotated_nodes_flat = np.zeros((0, 5), dtype=float)
        self.node_colors = np.zeros((0), dtype=np.int32)
        self.angles = np.array([0.0, 0.0, 0.0])
        self.rotate_speed = np.array([0.021, -0.017, 0.012])
        self.bounce = int((self.height - self.radius * 2) / 2)
        self.bounce_speed = 1000  # bounce every n milliseconds
        self.x_move = int((self.width - self.radius) / 6.0)
        self.z_move = self.radius * 3
        self.z_pos = self.radius * 10
        self.prev_z = 0.0
        self.xz_move_speed = 5370  # go round every n milliseconds
        self.size = 1.0
        self.position = self.mid_screen

        # the following for checking performance only
        self.info_display = True
        self.millisecs = 0
        self.plot_count = 0
        self.dot_count = 0
        self.timer_avg_frames = 180
        self.timer_names = []
        self.timers = np.zeros((1, 1), dtype=int)
        self.timer_frame = 0
        self.start_timer = 0
        self.font = pygame.font.SysFont('CourierNew', 15)

        # set up timers
        self.timer_name = []
        self.timer_names.append("rotate")
        self.timer_names.append("clear")
        self.timer_names.append("select plotted")
        self.timer_names.append("plot")
        self.timer_names.append("plot info")
        self.timer_names.append("display flip")
        self.timer_names.append("wait")
        self.timers = np.zeros((len(self.timer_names), self.timer_avg_frames), dtype=int)

        self.setup_ball()

        # initialize timers
        self.move_timer = pygame.time.get_ticks()
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
                    if event.key == pygame.K_SPACE:
                        self.pause()
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                                          self.__class__.__name__ + '.jpg')
                    if event.key == pygame.K_i:
                        self.toggle_info_display()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # left button: exit
                        self.running = False

            if self.paused:
                pygame.time.wait(100)
                self.millisecs = pygame.time.get_ticks()

            else:
                # main components executed here
                if self.mode == 2:
                    self.clear()
                self.move()
                self.rotate()
                if self.mode == 1:
                    self.clear_perimeter()
                    self.plot()
                else:
                    self.plot_backwards()
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

    def move(self):

        # move the object - add x move, z move (distance = size change), y bounce
        time = pygame.time.get_ticks()
        z_pos = np.cos(((time - self.move_timer) % self.xz_move_speed) / self.xz_move_speed * np.pi * 2.0) * self.z_move
        if np.sign(z_pos) == -1 and np.sign(self.z_prev) == 1:
            self.round += 1
            if self.round == self.rounds:
                self.round = 0
                # change mode when crossing z axis
                if self.mode == 1:
                    self.size = 1.0
                    self.mode = 2
                else:
                    self.mode = 1
        if self.mode == 1:
            self.size = self.z_pos / (self.z_pos + z_pos)
        y_pos = self.bounce / 2 - np.sin(((time - self.move_timer) % self.bounce_speed) / self.bounce_speed * np.pi) * self.bounce * self.size
        x_pos = np.sin(((time - self.move_timer) % self.xz_move_speed) / self.xz_move_speed * np.pi * 2.0) * self.x_move * self.size
        self.position = (self.mid_screen + np.array([x_pos, y_pos])).astype(np.int16)
        self.z_prev = z_pos

    def rotate(self):

        # rotate object
        self.angles += self.rotate_speed
        matrix = self.rotate_matrix(self.angles)

        if self.mode == 1:
            # normal 3D rotation
            self.rotated_nodes[:, 0:3] = np.matmul(self.nodes, matrix) * self.size
        else:
            # invert rotation matrix - "backwards rotation" of the required 2D nodes (flat nodes) to "original" to get colors
            matrix_inv = np.linalg.inv(matrix)
            self.rotated_nodes_flat[:, 0:3] = np.matmul(self.nodes_flat, matrix_inv)

        self.measure_time("rotate")

    def rotate_matrix(self, angles):

        # define rotation matrix for given angles
        (sx, sy, sz) = np.sin(angles)
        (cx, cy, cz) = np.cos(angles)

        # build a matrix for X, Y, Z rotation (in that order, see Wikipedia: Euler angles).
        return np.array([[cy * cz               , -cy * sz              , sy      ],
                         [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
                         [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy ]])

    def clear(self):

        # clear plot info area
        self.screen.fill(self.background_color, (10, 10, 315, 200))
        # clear area
        rad = int(self.radius * self.size)
        rect = np.hstack((self.position - np.array([rad + 5, rad + 5]), np.array([(rad + 5) * 2, (rad + 5) * 2])))
        # clear the area
        self.screen.fill(self.background_color, rect)
        self.measure_time("clear")

    def clear_perimeter(self):

        # clear plot info area
        self.screen.fill(self.background_color, (10, 10, 315, 200))
        # clear area with mask Rect, leaving the ball area as it is - smoother picture
        rad = int(self.radius * self.size)
        rect = np.hstack((self.position - np.array([rad + 20, rad + 20]), np.array([(rad + 20) * 2, (rad + 20) * 2])))
        # either clear the whole area OR leave a hole not cleared
        self.screen.fill(self.background_color, rect)
        # # clear mask area
        # self.screen_copy.fill(self.background_color, rect)
        # # draw circular mask - a hole, in effect - enable this to avoid any black dots (but will have some "shadows" of former images)
        # pygame.draw.circle(self.screen_copy, (255, 255, 255), self.position, rad)
        # # clear area except mask hole
        # self.screen.blit(self.screen_copy, rect, rect)
        self.measure_time("clear")

    def plot(self):

        # plot the sphere dots
        while self.screen.get_locked():
            self.screen.unlock()

        # using a surfarray for plotting. Select plotted nodes using coordinates: z < 0, y within screen (x could be tested as well).
        rot_nodes = (self.rotated_nodes[(self.rotated_nodes[:, 2] < 0)
                                        & (self.rotated_nodes[:, 1] >= -self.position[1])
                                        & (self.rotated_nodes[:, 1] < self.height - self.position[1] - 1)
                                        ] + np.array([self.position[0], self.position[1], 0, 0])
                     ).astype(np.int16)
        self.plot_count = np.shape(rot_nodes)[0]
        self.measure_time("select plotted")

        rgb_array = pygame.surfarray.pixels2d(self.screen)

        # color_mult = -225.0 / (self.radius * self.size)
        # color_mult2 = 256**2 + 256 + 1
        # color_add = 10 * color_mult2

        # rgb_array[rot_nodes[:, 0], rot_nodes[:, 1]] = (rot_nodes[:, 2] * color_mult).astype(np.int16) * color_mult2 + color_add
        # # add nearby dots to have less black spaces
        # rgb_array[rot_nodes[:, 0] + 1, rot_nodes[:, 1]] = (rot_nodes[:, 2] * color_mult).astype(np.int16) * color_mult2 + color_add
        # rgb_array[rot_nodes[:, 0], rot_nodes[:, 1] + 1] = (rot_nodes[:, 2] * color_mult).astype(np.int16) * color_mult2 + color_add
        # rgb_array[rot_nodes[:, 0] + 1, rot_nodes[:, 1] + 1] = (rot_nodes[:, 2] * color_mult).astype(np.int16) * color_mult2 + color_add

        rgb_array[rot_nodes[:, 0], rot_nodes[:, 1]] = rot_nodes[:, 3]
        # add nearby dots to have less black spaces
        rgb_array[rot_nodes[:, 0] + 1, rot_nodes[:, 1]] = rot_nodes[:, 3]
        rgb_array[rot_nodes[:, 0], rot_nodes[:, 1] + 1] = rot_nodes[:, 3]
        rgb_array[rot_nodes[:, 0] + 1, rot_nodes[:, 1] + 1] = rot_nodes[:, 3]

        self.measure_time("plot")

    def plot_backwards(self):

        # plot the sphere dots. Working backwards ie. plotting all pixels within the circle, and finding the color using rotated_nodes_flat

        self.plot_count = np.shape(self.nodes_flat_x)[0]

        while self.screen.get_locked():
            self.screen.unlock()

        # # transform Cartesian coordinates (X. Y. Z) to Polar (longitude, latitude) between 0 and 1
        # polar_x = np.arctan2(self.rotated_nodes_flat[:, 0], self.rotated_nodes_flat[:, 2]) / (2.0 * np.pi) + 0.5
        # polar_y = np.arccos(self.rotated_nodes_flat[:, 1] / self.radius) / np.pi

        # if self.image_type == 'Mercator':
        #     # Mercator mode; y in image assumed to be between -85 and 85 degrees. (https://en.wikipedia.org/wiki/Mercator_projection)
        #     R = 1.0 / np.log(np.tan(np.pi / 4 + (85.0 / 90.0) * np.pi / 4))  # Mercator R for +-85 degree image
        #     surf_coord_x = (polar_x * self.image_size[0]).astype(np.int16)  # x as in "normal"
        #     surf_coord_y = (np.minimum(0.9999, np.maximum(0.0, ((1.0 + R * np.log(np.tan(np.pi / 4 + 0.4999 * (np.arccos(self.rotated_nodes_flat[:, 1] / self.radius)
        #                    - np.pi / 2)))) / 2.0))) * self.image_size[1]).astype(np.int16)
        # color = self.image_array[surf_coord_x, surf_coord_y]

        # same as above but condensed, quicker
        rgb_array = pygame.surfarray.pixels2d(self.screen)
        rgb_array[self.nodes_flat_x + self.position[0], self.nodes_flat_y + self.position[1]] = self.image_array[
            ((np.arctan2(self.rotated_nodes_flat[:, 0], self.rotated_nodes_flat[:, 2]) / (2.0 * np.pi) + 0.5)
             * self.image_size[0]).astype(np.int16),  # x calculated
            # alternative X using a precalculated coordinate - but too complex to help much
            # (self.image_flat_x[(4.0 * self.rotated_nodes_flat[:, 1]).astype(np.int16) + 4 * self.radius] * self.rotated_nodes_flat[:, 0]
            #                    + (np.sign(self.rotated_nodes_flat[:, 2]) + 2) * self.image_size[0] / 4).astype(np.int16),  # x using precalcs
            # y calculated from Mercator - rather complex and hence slow
            # (np.minimum(0.9999, np.maximum(0.0, ((1.0 + R * np.log(np.tan(np.pi / 4 + 0.4999 * (np.arccos(self.rotated_nodes_flat[:, 1] / self.radius)
            #                 - np.pi / 2)))) / 2.0))) * self.image_size[1]).astype(np.int16)  # y calculated
            self.image_flat_y[(4.0 * self.rotated_nodes_flat[:, 1]).astype(np.int16) + 4 * self.radius]  # y using precalcs
           ]

        self.measure_time("plot")

    def setup_ball(self):

        # generate 3D nodes for a sphere.

        a = np.pi / 180.0  # for scaling degrees to radiuses
        R = self.image_mercator_R

        # first generate "traditional" 3D nodes with ~even spacing on sphere surface
        c = int(self.point_nr / 4)  # nr of circles of points needed for half a sphere.
        for i in range(c):
            lat = 90.0 / (2 * c) + 90.0 * (i / c)  # latitude north
            rad = np.cos(lat * a)  # radius scale at this latitude
            p = int(self.point_nr * rad)  # nr of points needed at this latitude
            j = np.arange(p)[:, None]
            long = 360.0 / (2 * p) + 360.0 * (j / p)  # longitudes at this latitude; most at the equator, least close to pole(s)
            x = np.cos(long * a) * rad * self.radius
            y = np.ones((p, 1)) * np.sin(lat * a) * self.radius  # y = latitude is constant
            z = np.sin(long * a) * rad * self.radius
            # add nodes both north and south
            self.nodes = np.vstack((self.nodes, np.hstack((x, y, z)), np.hstack((x, -y, z))))

            # pick colors from image
            if self.image_type == 'Mercator':
                # Mercator mode (https://en.wikipedia.org/wiki/Mercator_projection). Crop at 0.9999 - R defines "maximum" latitude (85 deg)
                lat_coord = np.minimum(0.9999, np.ones((p, 1)) * ((1.0 + R * np.log(np.tan(np.pi / 4.0 + 0.4999 * (lat / 90.0) * np.pi / 2.0)))
                                                                  / 2.0)) * self.image_size[1]
            else:
                # Normal mode: picture is simply wrapped
                lat_coord = np.ones((p, 1)) * (lat / (90.0 * 2.0) + 0.5) * self.image_size[1]

            image_coord = (np.vstack((np.hstack((long / 360.0 * self.image_size[0], lat_coord)),
                                      np.hstack((long / 360.0 * self.image_size[0], self.image_size[1] - lat_coord - 1))))).astype(np.int16)
            self.node_colors = np.hstack((self.node_colors, self.image_array[image_coord[:, 0], image_coord[:, 1]]))

        # preset rotated nodes so that colors are attached to them - faster to handle later.
        self.rotated_nodes = np.hstack((self.nodes, self.node_colors[:, None]))
        self.dot_count = np.shape(self.nodes)[0]

        # generate "2D" nodes for half a sphere. This time work backwards from 2D image to 3D nodes.
        # these then represent a set of rotated nodes, and the idea is to figure out which nodes they came from (and hence their color)
        # this should avoid the issue where due to rounding errors etc. rotating 3D nodes and converting to 2D leaves gaps.
        # however, more precalculation - not so easy to e.g. change the size of the image later.
        for y_val in range(-self.radius + 1, 0):
            rad = int(np.sqrt(self.radius ** 2 - y_val ** 2))  # radius (pixels) at this latitude
            x = np.arange(-rad, rad + 1)[:, None]  # x simply covers all pixels on this line (y = constant)
            y = np.ones((rad * 2 + 1, 1)) * y_val  # y = constant on this line
            z = -np.sqrt(self.radius ** 2 - y_val ** 2 - x ** 2)  # z from sphere formula x**2 + y**2 + z**2 = r**2; negative as facing the viewer
            self.nodes_flat = np.vstack((self.nodes_flat, np.hstack((x, y, z)), np.hstack((x, -y - 1, z))))

        # store as integers, as anyway plotting pixels
        self.nodes_flat_x = (self.nodes_flat[:, 0]).astype(np.int16)
        self.nodes_flat_y = (self.nodes_flat[:, 1]).astype(np.int16)
        # precalculate [Mercator] conversion for each (integer) y and the distance between x's. Add precision by using 4 times the number of radius lines
        flat_y_range = np.arange(-self.radius * 4.0, self.radius * 4.0 + 1)
        self.image_flat_x = (self.image_size[0] / 4.0) / np.maximum(2.0, np.sqrt(self.radius ** 2 - (flat_y_range / 4.0) ** 2))
        if self.image_type == 'Mercator':
            self.image_flat_y = (np.minimum(0.9999, np.maximum(0.0, ((1.0 + R * np.log(np.tan(np.pi / 4.0 + 0.49999 * (np.arccos(flat_y_range /
                        (4.0 * self.radius))- np.pi / 2.0)))) / 2.0))) * self.image_size[1]).astype(np.int16)
        else:
            self.image_flat_y = (np.minimum(0.9999, np.arctan2(flat_y_range, np.sqrt((self.radius * 4.0) ** 2 - flat_y_range ** 2)) / np.pi + 0.5)
                                 * self.image_size[1]).astype(np.int16)

        self.rotated_nodes_flat = np.zeros((np.shape(self.nodes_flat)[0], 4), dtype=float)

    def toggle_fullscreen(self):

        # toggle between fullscreen and windowed mode
        pygame.display.toggle_fullscreen()

    def pause(self):

        if self.paused:
            self.paused = False
            self.timer += pygame.time.get_ticks() - self.pause_timer  # adjust timer for pause time
            self.move_timer += pygame.time.get_ticks() - self.pause_timer  # adjust timer for pause time
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

        self.plot_info_msg(self.screen, 10, 10, 'frames per sec:   ' + str(int(self.clock.get_fps())))
        if self.mode == 1:
            self.plot_info_msg(self.screen, 10, 25, 'mode:             normal 3D')
            self.plot_info_msg(self.screen, 10, 40, 'dots plotted:     ' + str(int(self.plot_count)) + ' / ' + str(int(self.dot_count)))
        else:
            self.plot_info_msg(self.screen, 10, 25, 'mode:             backwards 3D')
            self.plot_info_msg(self.screen, 10, 40, 'dots plotted:     ' + str(int(self.plot_count)))
        self.plot_info_msg(self.screen, 10, 55, 'sphere radius:    ' + str(int(self.radius * self.size)) + ' pixels')
        self.plot_info_msg(self.screen, 10, 70, 'sphere area:      ' + str(int(np.pi * (self.radius * self.size) ** 2)) + ' pixels')
        self.plot_info_msg(self.screen, 10, 85, '"spherical" area: ' + str(int(2 * np.pi * (self.radius * self.size) ** 2)) + ' pixels')

        # add measured times as percentage of total
        tot_time = np.sum(self.timers)
        if tot_time > 0:
            for i in range(len(self.timer_names)):
                info_msg = (self.timer_names[i] + ' '*16)[:16] + (' '*10 + str(round(np.sum(self.timers[i, :]) * 100 / tot_time, 1)))[-7:]
                self.plot_info_msg(self.screen, 10, 105 + i * 15, info_msg)

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
    # disp_size = disp_modes[4] # selecting display size from available list. Assuming the 5th element is nice...
    # disp_size = (1920, 1080)
    disp_size = (1280, 720)
    # disp_size = (800, 600)

    pygame.font.init()
    pygame.mixer.init()
    music_file = "alacrity.ogg"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play(loops=-1)

    screen = pygame.display.set_mode(disp_size)  #, pygame.FULLSCREEN | pygame.DOUBLEBUF)
    pygame.display.set_caption('Ball')
    Ball(screen, 60).run()

    # exit; close display, stop music
    pygame.quit()
    exit()
