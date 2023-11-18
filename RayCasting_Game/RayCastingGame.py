# -*- coding: utf-8 -*-
import pygame
import numpy as np
from sys import exit


class RayCastingGame:
    """
    2.5 dimensional environment - raycasting / Doom style.

    This game calculates and draws a ray cast environment based on a map of rectangular blocks.
    Each wall, floor and ceiling pixel is mapped and drawn with NumPy using pygame surface's pixelarray.
    Coin and light bulb images are added on top. A map is drawn to help navigating.

    As time passes, your lamp grows dimmer - unless you collect light bulbs.
    The game ends when visibility is almost zero.

    Keys: cursor keys to steer, A and Z to adjust speed.

    @author: kalle
    """

    def __init__(self, screen, target_fps):

        self.screen = screen
        self.target_fps = target_fps
        self.screen_size = np.asarray(self.screen.get_size(), dtype=np.int16)
        self.width = self.screen_size[0]
        self.height = self.screen_size[1]
        self.mid_screen = self.screen_size // 2
        self.background_color = (0, 0, 0)
        self.screen_info = pygame.Surface((300, 200), 0, self.screen)
        self.screen_info.set_colorkey(self.background_color)
        self.move_speed = 0.0
        self.move_time = 0
        self.moving_time = 0
        self.anim_time = 0
        self.angle_speed = 25  # defines turning speed
        self.channel = 1
        self.nr_channels = 8
        self.running = True
        self.paused = False
        self.show_map = True
        self.clock = pygame.time.Clock()

        self.pic_theend = pygame.image.load('TheEnd.jpg').convert()
        self.pic_guru2 = pygame.image.load('Guru2.jpg').convert()
        self.pic_ball2 = pygame.image.load('Ball2.jpg').convert()
        self.pic_alphaks2 = pygame.image.load('Alphaks2.jpg').convert()
        self.pic_vector3d = pygame.image.load('Vector3D.jpg').convert()
        self.pic_shadowbobs = pygame.image.load('ShadowBobs.jpg').convert()
        self.pic_milkyway2 = pygame.image.load('Milky_Way2.jpg').convert()
        self.pic_ray01 = pygame.image.load('Raytracing_01.jpg').convert()
        self.pic_ray02 = pygame.image.load('Raytracing_02.jpg').convert()
        self.pic_ray03 = pygame.image.load('Raytracing_03.jpg').convert()
        self.pic_ray04 = pygame.image.load('Raytracing_04.jpg').convert()
        self.pic_ray05 = pygame.image.load('Raytracing_05.jpg').convert()
        self.pic_ray06 = pygame.image.load('Raytracing_06.jpg').convert()
        self.pic_ray07 = pygame.image.load('Raytracing_07.jpg').convert()
        self.pic_ray08 = pygame.image.load('Raytracing_08.jpg').convert()
        self.pic_ray09 = pygame.image.load('Raytracing_09.jpg').convert()
        self.pic_ray10 = pygame.image.load('Raytracing_10.jpg').convert()
        self.pic_coin = pygame.image.load('goldcoin2.png').convert()
        self.pic_coin.set_colorkey((0, 0, 0))
        self.pic_lightbulb = pygame.image.load('lightbulb3.png').convert()
        self.pic_lightbulb.set_colorkey((0, 0, 0))
        self.sound_collide = pygame.mixer.Sound('Raycast_collide.wav')
        self.sound_bounce = pygame.mixer.Sound('Raycast_bounce.wav')
        self.sound_start = pygame.mixer.Sound('Raycast_start.wav')
        self.sound_pling = pygame.mixer.Sound('Raycast_pling.wav')
        self.sound_win = pygame.mixer.Sound('Raycast_win.wav')
        self.sound_lose = pygame.mixer.Sound('Raycast_lose.wav')

        self.wall_blocks = []   # list of available blocks (class Block)
        self.map_blocks = {}    # dictionary connecting map block (alphanumeric character in self.map) to wall_blocks list index
        self.items = []         # list of items (class Item)
        self.setup_blocks()
        self.coin_blocks = 'cdefg'
        self.light_block = 'L'
        self.map = ['11122211111111111115551111111111111111111111111',
                    '1   f                           2222          1',
                    '1   f   1f     L      efgfedcde  22    ceg L  1',
                    '1   e  13f11     22              66    gec    1',
                    '1   e  1ee1      22      55                   3',
                    '1   d   d41           L      efefef    131    3',
                    '1  6d                                  131    3',
                    '4     ccccddee      11       2111111          3',
                    '4                            2                3',
                    '4   f         g        2  c   2    4444    f  1',
                    '1   g   11    f  L111 1   c  2eeee L 555   e  1',
                    '1   f  15451  e  16 611   c  23335     6   d  1',
                    '1   g         d      ee    c      54444 6   c  1',
                    '1      L  11     cgcgcgcg                     1',
                    '11166666111111111111444411111111111111114444444']
        self.map_array = np.zeros((len(self.map[0]), len(self.map)), dtype=np.uint8)
        self.block_size = int(self.width / 120)
        self.make_map_array(self.map)
        self.map_size = np.array(np.shape(self.map_array))
        self.map_minimum_size = np.array([7, 7])  # map will not decrease to smaller size than this
        self.position = self.map_size / 2  # initial position at center
        self.pos_angle = 0.0
        self.view_angle = 0.0
        self.view_height = 0.5  # "eye height" as percentage of view
        self.view_width = 60    # view width angle in degrees (horizontally)
        self.wall_height = self.width  # this adjusts wall height.
        self.stuck = False

        self.view_blocks = 5.5   # how many blocks away can see
        self.view_blocks_target = self.view_blocks
        self.view_lightbulb_add = 3.5  # how many blocks of view are added by getting a light bulb
        self.view_blocks_decrease = 20  # how many seconds for a one block decrease in view_blocks
        self.view_blocks_time = 0
        self.view_blocks_target_time = 0
        self.view_shade = 0.7   # fraction of view after which shading starts
        self.map_show_size = np.minimum(self.map_size, self.view_blocks * 2).astype(np.int16)  # the size of map area shown - limit to 2 x view_blocks

        self.proj_dist = np.tan(self.view_width * np.pi / 180) * self.width
        self.angle_add = self.view_width / self.width * np.pi / 180

        # item data copied from Item class to speed up handling
        self.item_position = np.zeros((0), float)
        self.item_size = np.zeros((0), float)
        self.item_y_level = np.zeros((0), float)
        self.item_ratio = np.zeros((0), float)
        self.item_itype = np.zeros((0), np.int16)
        self.item_active = np.ones((0), np.bool_)
        self.setup_items()
        self.item_data = np.zeros((len(self.items), 6), dtype=float)  # item data has visible item nr, angle, distance, x_coord, y_floor, y_ceiling
        self.items_visible = 0
        self.coin_rect = np.array([self.width * 5 / 6 - 4, 4, self.width * 1 / 6, self.width * 1 / 60], dtype=np.int16)  # defines coin progress bar

        # define ceiling and floor pics
        self.pic_ceiling = self.pic_milkyway2
        self.ceiling_block_size = (2, 2)
        self.pic_floor = self.pic_shadowbobs
        self.floor_block_size = (2, 2)

        # ray_data is an array of grid item, side (0 = up, 1 = right, 2 = down, 3 = left) map x, map y,
        #   distance from viewer, distance corrected for fishbowl, y_top and y_bottom for each ray
        self.ray_data = np.zeros((self.width, 8), dtype=float)
        # distance correction multiplier for each angle to counter the fishbowl effect
        self.distance_correction = np.cos(np.linspace(-self.view_width / 2, self.view_width / 2, self.width) / 180.0 * np.pi)
        # grid_blocks will hold the data on where each block of rays (for single processing run) starts and ends
        self.grid_blocks = np.zeros((0))

        self.screen_map = pygame.Surface(self.map_size * self.block_size, 0, self.screen)
        self.screen_map_visited = self.screen_map.copy()
        self.screen_shade = pygame.Surface((self.view_blocks * self.block_size * 2, self.view_blocks * self.block_size * 2), 0, self.screen)
        self.screen_shade_yellow = self.screen_shade.copy()
        self.screen_view_area = self.screen_shade.copy()
        self.screen_view_area.set_colorkey(self.background_color)
        self.screen_view_area_size = np.asarray(self.screen_view_area.get_size(), dtype=np.int16)
        self.setup_shade(self.block_size)
        self.setup_map(self.screen_map, self.block_size)
        self.map_position = (self.screen_size - self.map_show_size * self.block_size - 4)

        self.score = 0
        self.font_score = pygame.font.SysFont('Arial Black', 24)
        self.game_over = False
        self.font_game_over = pygame.font.SysFont('Arial Black', 48)

        # the following for checking performance only
        self.info_display = False
        self.millisecs = 0
        self.timer_avg_frames = 30
        self.timer_names = []
        self.timers = np.zeros((1, 1), dtype=int)
        self.timer_frame = 0
        self.font = pygame.font.SysFont('CourierNew', 15)

        # set up timers
        self.timer_name = []
        # self.timer_names.append("clear")
        self.timer_names.append("calculate")
        self.timer_names.append("draw walls")
        self.timer_names.append("draw ceiling")
        self.timer_names.append("draw floor")
        self.timer_names.append("item handling")
        self.timer_names.append("draw map")
        self.timer_names.append("plot info")
        self.timer_names.append("display flip")
        self.timer_names.append("wait")
        self.timers = np.zeros((len(self.timer_names), self.timer_avg_frames), dtype=int)

        # initialize timers
        self.millisecs = pygame.time.get_ticks()

    def run(self):
        """
        Main loop.
        """
        time = pygame.time.get_ticks()
        self.millisecs = time
        self.move_time = time
        self.anim_time = time
        self.view_blocks_time = time
        self.view_blocks_target_time = time

        # start in full screen mode
        self.toggle_fullscreen()

        while self.running:

            time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.key == pygame.K_f:
                        self.toggle_fullscreen()
                    if event.key == pygame.K_m:
                        self.toggle_map()
                    if event.key == pygame.K_v:
                        # if self.view_blocks == 5:
                        #     self.set_view_blocks(9)
                        # else:
                        #     self.set_view_blocks(5)
                        self.view_blocks_target += 3.0
                    if event.key == pygame.K_i:
                        self.toggle_info_display()
                    # if event.key == pygame.K_SPACE:
                    #     self.pause()
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        # pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                        #                   self.__class__.__name__ + '.jpg')
                        pygame.image.save(self.screen, self.__class__.__name__ + '.jpg')
                # elif event.type == pygame.MOUSEBUTTONDOWN:
                #     if event.button == 1:
                #         # left button: exit
                #         self.running = False

            self.adjust_view_blocks(time)

            if not self.paused:
                keys = pygame.key.get_pressed()
                self.move(keys, time)
            self.animate()

            # calculate the walls in view
            self.raycast(self.screen, self.block_size)

            # drawing operations; clear is part of drawing walls so not required separately. Two options for floor and ceiling: "whole" or "visible"
            # self.clear(self.screen)
            self.draw_floor_or_ceiling_whole(self.screen, self.ceiling_block_size, self.pic_ceiling, False)
            self.draw_floor_or_ceiling_whole(self.screen, self.floor_block_size, self.pic_floor, True)
            self.draw_walls(self.screen)
            # self.draw_floor_or_ceiling_visible(self.screen, self.ceiling_block_size, self.pic_ceiling, False)
            # self.draw_floor_or_ceiling_visible(self.screen, self.floor_block_size, self.pic_floor, True)

            self.item_handling(self.screen, time)

            if self.show_map:
                self.draw_map_view(self.screen, self.block_size, self.map_position)
            self.add_coin_bar_and_score(self.screen)
            if self.info_display:
                self.plot_info()

            # release any locks on screen
            while self.screen.get_locked():
                self.screen.unlock()

            # switch between currently showed and the next screen (prepared in "buffer")
            pygame.display.flip()
            self.measure_time("display flip")
            self.clock.tick(self.target_fps)  # this keeps code running at max target_fps
            self.measure_time("wait")
            self.next_time_frame()

    def adjust_view_blocks(self, time):

        # automatically decrease view_blocks as time passes
        # use view_blocks_target to change actual view_blocks only gradually
        if time - self.view_blocks_time > self.view_blocks_decrease * 100:
            self.view_blocks_target -= 0.1
            self.view_blocks_time += self.view_blocks_decrease * 100

        if self.view_blocks <= 1 and not self.game_over:
            # visibility gone --> game over
            self.game_over = True
            self.view_blocks_target = 10.0
            self.play_sound(self.sound_lose)

        if abs(self.view_blocks_target - self.view_blocks) < 0.01:
            self.view_blocks_target_time = time + 0
        elif time - self.view_blocks_target_time > 250:
            if self.view_blocks_target > self.view_blocks:
                self.set_view_blocks(min(self.view_blocks + 0.25, self.view_blocks_target))
            elif self.view_blocks_target < self.view_blocks:
                self.set_view_blocks(max(self.view_blocks - 0.25, self.view_blocks_target))
            self.view_blocks_target_time += 250

    def animate(self):

        # animate wall images; keep speed constant with frame rate differing.
        new_time = pygame.time.get_ticks()
        img_add = (new_time - self.anim_time) / 33.33  # 33.33 means 30 frames per second
        self.anim_time = new_time

        for block in self.wall_blocks:
            block.animate(img_add)

    def move(self, keys, time):

        # adjust move for time to keep speed constant with frame rate differing.
        # if game over, adjust speed to zero.

        new_time = pygame.time.get_ticks()
        move_time_adj = (new_time - self.move_time) / 2000.0
        self.move_time = new_time

        turn = 0
        up_down = 0
        speed = 0
        if keys[pygame.K_LEFT]:
            turn = -1
        if keys[pygame.K_RIGHT]:
            turn = +1
        if keys[pygame.K_UP]:
            up_down = +1
        if keys[pygame.K_DOWN]:
            up_down = -1
        if keys[pygame.K_a]:
            speed = +10
        if keys[pygame.K_z]:
            speed = -10

        if self.game_over:
            speed = -np.sign(self.move_speed) * 10

        if self.move_speed == 0:
            if time - self.moving_time > 1000 and speed != 0:
                self.play_sound(self.sound_start)
                self.moving_time = time
            else:
                if speed < 0:
                    speed = 0
        else:
            if self.move_speed > 0 and self.move_speed + speed * move_time_adj <= 0:
                self.move_speed = 0
                speed = 0
            self.moving_time = time

        self.move_speed = max(-2.0, min(10.0, self.move_speed + speed * move_time_adj))  # keep speed in this range
        turn_speed = max(self.move_speed, 3.0)  # but for turning and moving up/down use minimum 3.0

        # moving up/down
        self.view_height = max(0.04, min(0.96, self.view_height + up_down * (turn_speed / 2) * move_time_adj))

        # turn
        self.pos_angle += turn * self.angle_speed * turn_speed * move_time_adj * np.sign(self.move_speed + 0.001)
        if self.pos_angle < 0:
            self.pos_angle += 360
        elif self.pos_angle >= 360:
            self.pos_angle -= 360

        bounce_angle = 30  # if collision angle is less than this, then bounce
        # check that new position is not within a wall (a non-zero block) OR that the distance to wall is very small ("corner" on the way)
        sp, cp = np.sin(self.pos_angle * np.pi / 180), -np.cos(self.pos_angle * np.pi / 180)
        new_pos = self.position + np.array([sp, cp]) * self.move_speed * move_time_adj
        if self.map_array[int(new_pos[0]), int(new_pos[1])] != 0 or self.ray_data[self.width // 2, 4] < self.move_speed * move_time_adj:
            # new position inside a wall. Check if angle to grid is max bounce_angle degrees
            new_angle = -999.0
            if (self.pos_angle + bounce_angle) % 180 <= 2 * bounce_angle and self.ray_data[self.width // 2, 1] in (1, 3):
                # going up or down and the wall is left or right side wall - apply bounce to x coordinate
                new_pos[0] += 2 * (round(new_pos[0], 0) - new_pos[0])
                new_angle = self.pos_angle - 2 * ((self.pos_angle + bounce_angle) % 180 - bounce_angle)  # "bounce angle"
            elif (self.pos_angle + 90 + bounce_angle) % 180 <= 2 * bounce_angle and self.ray_data[self.width // 2, 1] in (0, 2):
                # going left or right and the wall is up or down side wall - apply bounce to y coordinate
                new_pos[1] += 2 * (round(new_pos[1], 0) - new_pos[1])
                new_angle = self.pos_angle - 2 * ((self.pos_angle + 90 + bounce_angle) % 180 - bounce_angle)  # "bounce angle"
            # check bounced position for walls, do not apply bounce if stuck
            if not self.stuck and self.map_array[int(new_pos[0]), int(new_pos[1])] == 0 and new_angle > -999.0:
                # bounce possible. Decrease speed and set new angle
                self.move_speed *= 0.7
                self.pos_angle = new_angle
                self.play_sound(self.sound_bounce)
            else:
                # no bounce, get "stuck". Try to "bounce back".
                new_pos = self.position - np.array([sp, cp]) * self.move_speed * 0.025
                if self.map_array[int(new_pos[0]), int(new_pos[1])] != 0:
                    # bounce back not possible, just stuck...
                    new_pos = self.position
                # stop
                self.stuck = True
                self.move_speed = 0.0
                self.play_sound(self.sound_collide)
        else:
            # clear path
            self.stuck = False

        self.position = new_pos
        self.view_angle = self.pos_angle

    def raycast(self, screen, block_size):

        # go through view horizontally i.e. from position and view_angle go from left view_angle_witdh / 2 to right view_angle_witdh / 2.
        # below "horizontal" and "vertical" refer to a map on the screen, not the view itself.
        # angle 0 degrees = "up", rotation clockwise

        # position rounded to grid and the decimals (ie. position within the grid it is in)
        pos = np.floor(self.position)
        pos_dec = self.position - pos

        width = self.width  # this is the number of rays cast - ideally the width of the program window in pixels, so one ray per vertical line of pixels.
        view = int(self.view_blocks)  # nr of blocks: how far is visible (and, hence, how far must be calculated)
        rng_width = np.arange(width)
        # rng_view goes from zero to maximum view length in blocks.
        rng_view = np.arange(view + 1)  # add 1 to make sure covers the viewable distance as blocks are "rounded down"
        # calculate all angles etc. in one go - faster than looping
        # rads is an array (size: width) of all view angles in radians. Add 0.0001 to avoid exact 0/90/180/270 deg (and hence 0 sin/cos - div by zero)
        # rads are not evenly spaced, as they are on a line, not a circle radius. Their tangent is evenly spaced (when view_angle not yet applied),
        rad_tan = np.tan(self.view_width * np.pi / 360)  # rad_tan is the tangent of half the viewing angle
        rads = np.arctan((rng_width / (width - 1) - 0.5) * rad_tan * 2) + self.view_angle * np.pi / 180 + 0.0001
        sin_a = np.sin(rads)
        cos_a = np.cos(rads)
        tan_a = sin_a / cos_a  # np.tan(rads)
        sign_x = np.sign(sin_a)   # sign_x = +1 for angles > 0 and < 180, -1 for angles > 180
        sign_y = np.sign(cos_a)   # sign_y = +1 for angles < 90 or > 270, -1 for angles > 90 and < 270

        # get the first encountered coordinates when moving to next grid along ray line
        # if sign_y is negative, ray is cast downwards and the y distance is pos_dec[1] - 1. The same for x...
        x_first = (pos_dec[1] + (sign_y - 1) / 2) * tan_a
        y_first = (pos_dec[0] - (sign_x + 1) / 2) / tan_a

        # vertical intersections of ray and grid; y coordinate increases/decreases by one, calculate respective x coordinate
        # x_coord_y is the "grid point" (no info on which side of grid) while x_coord_y_adj is the y coordinate adjusted for viewer position relative to it (includes side).
        #   i.e. if grid point is higher than viewer (sign_y = -1), 1 is added to bring x_coord_y_adj to the bottom of that grid point.
        vert_grid_range = rng_view
        x_coord = self.position[0] + x_first[:, None] + vert_grid_range * (sign_y * tan_a)[:, None]
        x_coord_y = pos[1] - (vert_grid_range + 1) * sign_y[:, None]
        x_coord_y_adj = x_coord_y - (np.sign(x_coord_y - pos[1]) - 1) / 2
        # get the respective grid cell data from the map. Make sure not to exceed map boundaries. Pick only "view" nr of blocks; then add a "stop block" as last block
        # stop block is -1 as it is not a real block.
        vert_grid = np.hstack((
            self.map_array[(np.minimum(np.maximum(x_coord[:, :view], 0), self.map_size[0] - 1)).astype(np.int16),
                           (np.minimum(np.maximum(x_coord_y[:, :view], 0), self.map_size[1] - 1)).astype(np.int16)],
            -np.ones((width, 1), dtype=np.int16)  # this is the farthest block and always marked as "1" to make sure argmax finds something.
            ))
        # find the first nonzero item's index for each angle. Since distance is growing with each item, this is the closest wall encountered for each ray.
        vert_grid_first = (vert_grid != 0).argmax(axis=1)
        # construct an array of grid item, side (0 = up, 1 = right, 2 = down, 3 = left) map x, map y, and distance from viewer
        vert_data = (np.vstack((
            vert_grid[rng_width, vert_grid_first],
            sign_y + 1,
            x_coord[rng_width, vert_grid_first],
            x_coord_y_adj[rng_width, vert_grid_first],
            np.abs((x_coord[rng_width, vert_grid_first] - self.position[0]) / sin_a),
            ))).transpose()

        # horizontal intersections of ray and grid; x coordinate increases/decreases by one, calculate respective y coordinate
        horiz_grid_range = rng_view
        y_coord = self.position[1] + y_first[:, None] - horiz_grid_range * (sign_x / tan_a)[:, None]
        y_coord_x = pos[0] + (horiz_grid_range + 1) * sign_x[:, None]
        y_coord_x_adj = y_coord_x - (np.sign(y_coord_x - pos[0]) - 1) / 2
        horiz_grid = np.hstack((
            self.map_array[(np.minimum(np.maximum(y_coord_x[:, :view], 0), self.map_size[0] - 1)).astype(np.int16),
                           (np.minimum(np.maximum(y_coord[:, :view], 0), self.map_size[1] - 1)).astype(np.int16)],
            -np.ones((width, 1), dtype=np.int16)  # this is the farthest block and always marked as "1" to make sure argmax finds something.
            ))
        horiz_grid_first = (horiz_grid != 0).argmax(axis=1)
        horiz_data = (np.vstack((
            horiz_grid[rng_width, horiz_grid_first],
            sign_x + 2,
            y_coord_x_adj[rng_width, horiz_grid_first],
            y_coord[rng_width, horiz_grid_first],
            np.abs((y_coord[rng_width, horiz_grid_first] - self.position[1]) / cos_a)
            ))).transpose()

        # combine data on vertical and horizontal first encountered walls
        comb_data = np.stack((vert_data, horiz_data))
        # find the closest of these two - distance being item 4
        comb_closest = np.argmin(comb_data[:, :, 4], axis=0)
        # pick these for the closest data
        pick_data = comb_data[comb_closest, rng_width, :]
        # y_top and y_bottom are screen y top and bottom coordinates for each x coordinate. Wall height is divided by its distance for each ray (corrected for "fish bowl")
        dist_corr = pick_data[:, 4] * self.distance_correction
        # mark the blocks as "-1" when distance > self.view_blocks; these should not be drawn abd data are useless.
        pick_data[:, 0:2][dist_corr > self.view_blocks] = -1
        y_top = self.height / 2 - (self.wall_height * (1.0 - self.view_height)) / dist_corr
        y_bottom = self.height / 2 + (self.wall_height * self.view_height) / dist_corr
        # store these in self.ray_data
        self.ray_data[:, :] = np.hstack((pick_data, dist_corr[:, None], y_top[:, None], y_bottom[:, None]))

        # figure out where each block starts and ends. If the same block and side are used continuously, this will be treated as one block.
        # grid_changes contains the beginning of each block. These blocks can later be drawn in one go each as the source image is the same. Always include first and last.
        # 1. np.diff: get the difference of each block to the preceding in each of [block, side, x_coord, y_coord].
        # 2. sum the absolutes of these for each ray.
        # 3. convert to int16 and get the indexes of each non-zero change. This means block and side changes are always accounted for (are >= 1), but small changes
        #    (same or the very next) in grid position are not (are < 1).
        # 4. combine the changes so that also the last ray of the preceding block is included, and add the first and last rays.
        grid_changes = (np.sum(np.abs(np.diff(self.ray_data[:, :4], n=1, axis=0)), axis=1)).astype(np.int16).nonzero()[0] + 1
        self.grid_blocks = np.zeros((np.shape(grid_changes)[0] * 2 + 2), np.int16)
        # grid_blocks[0] = 0                    # first ray - this is already zero
        self.grid_blocks[1:-1:2] = grid_changes - 1  # the last ray in the block before change
        self.grid_blocks[2:-1:2] = grid_changes      # the first ray of the block after change
        self.grid_blocks[-1] = width - 1             # the last ray

        self.measure_time("calculate")

    def item_handling(self, screen, time):

        # check if items are in view. Store their item number, angle from viewer, distance from viewer, x coord, y floor, and y ceiling - if visible
        active = self.item_active
        item_nr = np.arange(np.size(self.item_active))[active]
        angle = np.arctan2(self.item_position[active, 0] - self.position[0], self.position[1] - self.item_position[active, 1]) * 180.0 / np.pi
        angle[angle < 0] += 360
        dist = np.sqrt(np.sum((self.item_position[active, :] - self.position) ** 2, axis=1)) + 0.000001  # using + 0.000001 to ensure dist > 0
        size = self.item_size[active]
        y_level = self.item_y_level[active]
        view_min_angle = self.pos_angle - self.view_width / 2
        # as items have width, check a wider angle. Calculate item image width at its distance, divide by screen width and multiply by view angle to get adjustment
        view_size_adj = ((self.wall_height / dist) * size * self.item_ratio[active] / self.width) * self.view_width

        # visible = True if close enough and within view angle. Note DOES NOT check for walls in between.
        # items has the item number of each visible item, collided has the item number of each item collided with.
        items = ((dist > size / 2) &
                 (dist <= self.view_blocks) &
                 ((angle - (view_min_angle - view_size_adj / 2)) % 360 <= self.view_width + view_size_adj)).nonzero()[0]
        collided = ((dist <= size / 2) &
                    (self.view_height >= y_level - size / 2) &
                    (self.view_height <= y_level + size / 2)).nonzero()[0]
        self.items_visible = np.size(items)

        if self.items_visible > 0:
            # x coordinate based on angle. Not really defined if not visible.
            angle_diff = angle[items] - self.pos_angle
            angle_diff[angle_diff > 180] -= 360  # adjust angle diff when e.g. angle = 350 and pos_angle = 10 --> true diff = -20
            angle_diff[angle_diff < -180] += 360  # adjust angle diff when e.g. angle = 10 and pos-angle = 350 --> true diff = +20
            x_coord = (angle_diff / self.view_width + 0.5) * self.width
            # y coordinate can be calculated using the (fish bowl corrected) distance
            y_floor = self.height / 2 + (self.wall_height * self.view_height) / (dist[items] * np.cos((self.pos_angle - angle[items]) / 180.0 * np.pi))
            y_ceiling = self.height / 2 - (self.wall_height * (1.0 - self.view_height)) / (dist[items] * np.cos((self.pos_angle - angle[items]) / 180.0 * np.pi))
            self.item_data[:self.items_visible, :] = np.vstack((
                item_nr[items],
                angle[items],
                dist[items],
                x_coord,
                y_floor,
                y_ceiling
                )).transpose()
            # find a sorted order so that the most distant item is first
            sorted_items = self.item_data[:self.items_visible, 2].argsort()[::-1]

            for i in sorted_items:
                item = self.items[int(self.item_data[i, 0])]
                if item.image_cnt > 0:
                    item.animate(time)  # first animate so that the item.image_used is selected from list correctly.
                    y_size = int((self.item_data[i, 4] - self.item_data[i, 5]) * item.size)  # use room height (in pixels) at item location and item size to determine image height
                    x_size = int(y_size * item.image_size[item.image_used, 0] / item.image_size[item.image_used, 1])  # keep aspect ratio by scaling x_size accordingly
                    # image screen x coordinates. Use offsets when image goes outside of screen
                    x_left = int(self.item_data[i, 3] - x_size / 2)
                    x_right = x_left + x_size
                    x_left_offset = -min(x_left, 0)
                    x_right_offset = min(self.width - x_right, 0)
                    if x_size >= 2 and y_size >= 2 and x_left + x_left_offset < x_right + x_right_offset:
                        # find distance to wall for each image x (on screen).
                        wall_dist = self.ray_data[x_left + x_left_offset:x_right + x_right_offset, 4]
                        # check first that item is not totally behind walls (ie. at least for some x the wall distance > item distance)
                        if np.max(wall_dist) > self.item_data[i, 2]:
                            # if not, we need a scaled item image. check if shading required
                            if self.item_data[i, 2] > self.view_blocks * self.view_shade:
                                scaled_img_pre = pygame.transform.scale(item.images[item.image_used], (x_size, y_size))
                                scaled_img_pre.set_alpha(int((self.view_blocks - self.item_data[i, 2]) / (self.view_blocks * (1.0 - self.view_shade)) * 255))
                                scaled_img = pygame.Surface((x_size, y_size))
                                scaled_img.blit(scaled_img_pre, (0, 0))
                                scaled_img.set_colorkey(self.background_color)
                            else:
                                scaled_img = pygame.transform.scale(item.images[item.image_used], (x_size, y_size))
                            # figure out the x coordinate blocks not behind walls (ie. closer to viewer than wall at x)
                            # pad the check with False on either side to help np.diff find all blocks. Use nonzero()[0] to pick only the changes.
                            wall_check = np.diff(np.hstack((False, (wall_dist > self.item_data[i, 2]), False)), n=1).nonzero()[0]
                            # copy image slices to screen. screen_y = ceiling + mid_point y_level - y_size / 2
                            screen_y = int(self.item_data[i, 5] + (1.0 - item.y_level) * (self.item_data[i, 4] - self.item_data[i, 5]) - y_size / 2)
                            for s in range(0, np.size(wall_check), 2):
                                x_pos = wall_check[s] + x_left_offset
                                x_end = wall_check[s + 1] + x_left_offset
                                screen_x = int(self.item_data[i, 3] - x_size / 2 + x_pos)
                                # blit the scaled (and shaded) image slice in place
                                screen.blit(scaled_img, (screen_x, screen_y), (x_pos, 0, x_end - x_pos, y_size))

        # handle any collisions to items
        for i in collided:
            item = self.items[item_nr[i]]
            item.collide()
            self.play_sound(item.collide_sound)
            self.item_active[item_nr[i]] = item.active
            if self.item_itype[item_nr[i]] == 1:
                # a coin: add to score
                self.score += 1
                # test if game won to play fanfare
                coins_left_nr = np.count_nonzero(self.item_itype[self.item_active == 1] == 1)  # active coins
                if coins_left_nr == 0:
                    self.play_sound(self.sound_win)
            elif self.item_itype[item_nr[i]] == 2:
                # a lightbulb: add view_blocks
                self.view_blocks_target += self.view_lightbulb_add

        self.measure_time("item handling")

    def clear(self, screen):

        # clear is only required when walls are so far away they will not be drawn at all
        y_floor = self.height / 2 + (self.wall_height * self.view_height) / ((self.view_blocks - 1) * self.distance_correction[0])  # this is the most distant visible floor y
        y_ceiling = self.height / 2 - (self.wall_height * (1.0 - self.view_height)) / ((self.view_blocks - 1) * self.distance_correction[0])  # this is the most distant visible ceiling y
        for i in range(0, np.size(self.grid_blocks), 2):
            # check if the wall is not visible
            block_nr = int(self.ray_data[self.grid_blocks[i], 0])
            if block_nr <= 0:
                screen.fill(self.background_color, (self.grid_blocks[i], y_ceiling, self.grid_blocks[i + 1] - self.grid_blocks[i] + 1, y_floor - y_ceiling + 1))

        self.measure_time("clear")

    def draw_walls(self, screen):

        # draw the walls, grid block by grid block.

        # obtain a surfarray on screen to modify it pixel by pixel with NumPy
        while screen.get_locked():
            screen.unlock()
        rgb_array_dst = pygame.surfarray.pixels3d(screen)

        # calculate where floor and ceiling end to be able to clear grid blocks not drawn as walls
        y_floor = self.height / 2 + (self.wall_height * self.view_height) / self.view_blocks  # this is the most distant visible floor y
        y_ceiling = self.height / 2 - (self.wall_height * (1.0 - self.view_height)) / self.view_blocks  # this is the most distant visible ceiling y

        # process each grid block in one go
        for i in range(0, np.size(self.grid_blocks), 2):

            # check if the wall is visible
            block_nr = int(self.ray_data[self.grid_blocks[i], 0])
            if block_nr > 0:

                x_left = self.grid_blocks[i]
                x_right = self.grid_blocks[i + 1]
                x_num = x_right - x_left + 1
                # y_top and y_bottom are screen y coordinates for each x coordinate.
                y_top = self.ray_data[x_left:x_right + 1, 6:7]
                y_bottom = self.ray_data[x_left:x_right + 1, 7:8]
                y_min = int(max(0.0, min(y_top[0], y_top[-1])))
                y_max = int(min(self.height - 1, max(y_bottom[0], y_bottom[-1]) + 0.999999))
                y_num = y_max - y_min + 1

                if y_num > 1:

                    # figure out which image to use
                    block = self.wall_blocks[block_nr]
                    side_nr = int(self.ray_data[self.grid_blocks[i], 1])
                    rgb_array_src = pygame.surfarray.pixels3d(block.images[side_nr][int(block.image_used[side_nr])])
                    (x_size, y_size) = np.shape(rgb_array_src)[0:2]

                    # map y coordinates on source image. This will be a (y_num, x_num) array transposed.
                    # if y_top is negative i.e. picture starts above screen area, y_map will be positive for y = 0 (i.e. starting in mid-image).
                    # if y_top > y_min (i.e. image height at x is smaller than the drawing area height, y_map will be negative for y = 0.
                    # ...and the same for the bottom side.
                    y_map = ((np.arange(y_min, y_max + 1, dtype=np.int16) - y_top) * y_size / (y_bottom - y_top)).astype(np.int16)

                    # map x coordinates on image based on side (0 = up, 1 = right, 2 = down, 3 = left) and the respective map coordinate.
                    # as block size = 1, use the decimal part of map coordinate to find the respective image x coordinate.
                    # resize the map from one line to all lines (as all lines are the same). This will be a (y_num, x_num) array transposed.
                    if side_nr == 0:
                        x_map = (np.resize(((0.999999 - (self.ray_data[x_left:x_right + 1, 2] - np.floor(self.ray_data[x_left:x_right + 1, 2])))
                                            * x_size).astype(np.int16), (y_num, x_num))).transpose()
                    elif side_nr == 2:
                        x_map = (np.resize(((self.ray_data[x_left:x_right + 1, 2] - np.floor(self.ray_data[x_left:x_right + 1, 2]))
                                            * x_size).astype(np.int16), (y_num, x_num))).transpose()
                    elif side_nr == 1:
                        x_map = (np.resize(((0.999999 - (self.ray_data[x_left:x_right + 1, 3] - np.floor(self.ray_data[x_left:x_right + 1, 3])))
                                            * x_size).astype(np.int16), (y_num, x_num))).transpose()
                    else:
                        x_map = (np.resize(((self.ray_data[x_left:x_right + 1, 3] - np.floor(self.ray_data[x_left:x_right + 1, 3]))
                                            * x_size).astype(np.int16), (y_num, x_num))).transpose()

                    # pick the valid pixels i.e. y is mapped inside the source picture (valid is False when y is negative or greater than image height).
                    valid = (y_map >= 0) & (y_map < y_size)

                    # draw all pixels of destination range but again dropping out the pixels not mapped on the image.
                    # picking the mapped image coordinates from source data; these arrays will be flattened when applying [valid].

                    if np.amax(self.ray_data[x_left:x_right + 1, 4]) > self.view_shade * self.view_blocks:
                        # add shading when objects are far away. Shading applied if max distance (not corrected) > self.view_shade * self.view_blocks
                        shade = (np.resize(np.minimum(1.0, np.maximum(0.0, (self.view_blocks - self.ray_data[x_left:x_right + 1, 4])
                                                                      / (self.view_blocks * (1.0 - self.view_shade)))), (y_num, x_num, 1))).transpose(1, 0, 2)
                        # using shade array to phase the image to black in the distance.
                        rgb_array_dst[x_left:x_left + x_num, y_min:y_min + y_num][valid] = rgb_array_src[x_map[valid], y_map[valid]] * shade[valid]

                    else:
                        # no shading required
                        rgb_array_dst[x_left:x_left + x_num, y_min:y_min + y_num][valid] = rgb_array_src[x_map[valid], y_map[valid]]

            else:
                # if the wall block is not visible, clear the area.
                screen.fill(self.background_color, (self.grid_blocks[i], y_ceiling - 1, self.grid_blocks[i + 1] - self.grid_blocks[i] + 1, y_floor - y_ceiling + 3))

        self.measure_time("draw walls")

    def draw_floor_or_ceiling_visible(self, screen, block_size, pic, is_floor=True):

        # draw the floor or the ceiling - all in one go.
        # block_size defines how many map blocks the image (pic) should cover in (x, y)
        # if is_floor is False then will draw the ceiling.
        # this version only draws the visible area of floor/ceiling, so may be run after drawing the walls.

        # obtain a surfarray on screen to modify it pixel by pixel with NumPy
        while screen.get_locked():
            screen.unlock()
        rgb_array_dst = pygame.surfarray.pixels3d(screen)

        rgb_array_src = pygame.surfarray.pixels3d(pic)
        bl_x_size = block_size[0]
        bl_y_size = block_size[1]
        x_size = int(np.shape(rgb_array_src)[0] // bl_x_size)
        y_size = int(np.shape(rgb_array_src)[1] // bl_y_size)

        # using the same formula as for calculating y_top and y_bottom for wall distance (see self.raycast())
        # we can also calculate distance for each y_coord. This is the same for all rays / viewing angles.
        if is_floor:
            y_num = int(self.height - np.amin(self.ray_data[:, 7]))
            y_coord = np.arange(0, y_num)
            dist = (self.wall_height * self.view_height) / (self.height / 2 - y_coord)
        else:
            y_num = int(np.amax(self.ray_data[:, 6]))
            y_coord = np.arange(0, y_num)
            dist = (self.wall_height * (1.0 - self.view_height)) / (self.height / 2 - y_coord)

        if y_num > 1:
            # mapping these to our map coordinates simply use distance above and go linearly from viewer position to wall position for each ray
            x_map = self.position[0] + (self.ray_data[:, 2:3] - self.position[0]) * dist / self.ray_data[:, 5:6]
            y_map = self.position[1] + (self.ray_data[:, 3:4] - self.position[1]) * dist / self.ray_data[:, 5:6]

            # valid pixels are lower than wall bottom (floor) or higher than wall top (ceiling)
            valid = (np.resize(dist, (self.width, y_num)) < np.repeat(np.minimum(self.ray_data[:, 5:6], self.view_blocks), y_num, axis=1))
            x_map_valid = (np.mod(x_map[valid], bl_x_size) * x_size).astype(np.int16)
            y_map_valid = (np.mod(y_map[valid], bl_y_size) * y_size).astype(np.int16)

            if dist[-1] / self.distance_correction[0] > self.view_shade * self.view_blocks:
                # add shading when floor/ceiling far away. Shading applied if distance (not corrected for fishbowl) > self.view_shade * self.view_blocks
                shade = (np.minimum(1.0, np.maximum(0.0, (self.view_blocks - dist / self.distance_correction[:, None]) /
                                                    (self.view_blocks * (1.0 - self.view_shade)))))[:, :, None]

                if is_floor:
                    rgb_array_dst[:, -1:-1 - y_num:-1][valid] = rgb_array_src[x_map_valid, y_map_valid] * shade[valid]
                    self.measure_time("draw floor")
                else:
                    rgb_array_dst[:, :y_num][valid] = rgb_array_src[x_map_valid, y_map_valid] * shade[valid]
                    self.measure_time("draw ceiling")

            else:
                # no shading needed
                if is_floor:
                    rgb_array_dst[:, -1:-1 - y_num:-1][valid] = rgb_array_src[x_map_valid, y_map_valid]
                    self.measure_time("draw floor")
                else:
                    rgb_array_dst[:, :y_num][valid] = rgb_array_src[x_map_valid, y_map_valid]
                    self.measure_time("draw ceiling")

    def draw_floor_or_ceiling_whole(self, screen, block_size, pic, is_floor=True):

        # draw the floor or the ceiling - all in one go.
        # block_size defines how many map blocks the image (pic) should cover in (x, y)
        # if is_floor is False then will draw the ceiling.
        # this version draws the whole floor/ceiling rectangle, not testing for existing walls, so must be run before drawing walls.

        # obtain a surfarray on screen to modify it pixel by pixel with NumPy
        while screen.get_locked():
            screen.unlock()
        rgb_array_dst = pygame.surfarray.pixels3d(screen)

        rgb_array_src = pygame.surfarray.pixels3d(pic)
        bl_x_size = block_size[0]
        bl_y_size = block_size[1]
        x_size = int(np.shape(rgb_array_src)[0] // bl_x_size)
        y_size = int(np.shape(rgb_array_src)[1] // bl_y_size)
        min_block = np.min(self.ray_data[:, 0])  # to test if all blocks are walls i.e. min_block > 0

        # using the same formula as for calculating y_top and y_bottom for wall distance (see self.raycast())
        # we can also calculate distance for each y_coord. This is the same for all rays / viewing angles.
        if is_floor:
            y_top = self.height / 2 + (self.wall_height * self.view_height) / self.view_blocks  # this is the most distant visible floor y
            if min_block > 0:
                y_num = int(self.height - max(y_top, np.amin(self.ray_data[:, 7])))  # if floor is cut off before y_top by walls, use the highest cut off point
            else:
                y_num = int(self.height - y_top)
            y_coord = np.arange(0, y_num)
            dist = (self.wall_height * self.view_height) / (self.height / 2 - y_coord)
            # find the left and right edges of rectangle (ie. do not draw floor/ceiling if wall covers the whole vertical space left or right)
            x_left = (self.ray_data[:, 7] < self.height).argmax()  # x_left is the first x where y is not below screen
            x_right = self.width - (np.flip(self.ray_data[:, 7]) < self.height).argmax()  # x_right is the last x where  y is not below screen + 1
        else:
            y_bottom = self.height / 2 - (self.wall_height * (1.0 - self.view_height)) / self.view_blocks  # this is the most distant visible ceiling y
            if min_block > 0:
                y_num = int(min(y_bottom, np.amax(self.ray_data[:, 6])))  # if ceiling is cut off before y_bottom by walls, use the lowest cut off point
            else:
                y_num = int(y_bottom)
            y_coord = np.arange(0, y_num)
            dist = (self.wall_height * (1.0 - self.view_height)) / (self.height / 2 - y_coord)
            # find the left and right edges of rectangle (ie. do not draw floor/ceiling if wall covers the whole vertical space left or right)
            x_left = (self.ray_data[:, 6] > 0).argmax()  # x_left is the first x where y is positive
            x_right = self.width - (np.flip(self.ray_data[:, 6]) > 0).argmax()  # x_right is the last x where y is positive + 1

        if y_num > 1 and x_right > x_left:

            # mapping these to our map coordinates simply use distance above and go linearly from viewer position to wall position for each ray.
            # using np.mod to pick the remainder and scaling that to image coordinate.
            x_map = (np.mod(self.position[0] + (self.ray_data[x_left:x_right, 2:3] - self.position[0]) * dist / self.ray_data[x_left:x_right, 5:6],
                            bl_x_size) * x_size).astype(np.int16)
            y_map = (np.mod(self.position[1] + (self.ray_data[x_left:x_right, 3:4] - self.position[1]) * dist / self.ray_data[x_left:x_right, 5:6],
                            bl_y_size) * y_size).astype(np.int16)

            if dist[-1] / self.distance_correction[0] > self.view_shade * self.view_blocks:
                # add shading when floor/ceiling far away. Shading applied if max distance (not corrected for fishbowl) > self.view_shade * self.view_blocks
                shade = ((np.minimum(1.0, np.maximum(0.0, (self.view_blocks - dist / self.distance_correction[x_left:x_right, None]) /
                                                     (self.view_blocks * (1.0 - self.view_shade))))))[:, :, None]

                if is_floor:
                    rgb_array_dst[x_left:x_right, -1:-1 - y_num:-1] = (rgb_array_src[x_map.ravel(), y_map.ravel()]).reshape(x_right - x_left, y_num, 3) * shade
                    self.measure_time("draw floor")
                else:
                    rgb_array_dst[x_left:x_right, :y_num] = (rgb_array_src[x_map.ravel(), y_map.ravel()]).reshape(x_right - x_left, y_num, 3) * shade
                    self.measure_time("draw ceiling")

            else:
                # no shading needed
                if is_floor:
                    rgb_array_dst[x_left:x_right, -1:-1 - y_num:-1] = (rgb_array_src[x_map.ravel(), y_map.ravel()]).reshape(x_right - x_left, y_num, 3)
                    self.measure_time("draw floor")
                else:
                    rgb_array_dst[x_left:x_right, :y_num] = (rgb_array_src[x_map.ravel(), y_map.ravel()]).reshape(x_right - x_left, y_num, 3)
                    self.measure_time("draw ceiling")

    def draw_map_view(self, screen, block_size, map_position):

        # copies the map on screen and draws the player view on the map.

        # center map on player position - if possible.
        map_pos = np.minimum(np.maximum(0, self.position - self.map_show_size / 2), self.map_size - self.map_show_size)
        map_rect = pygame.Rect((map_pos * block_size + 0.5).astype(np.int16), self.map_show_size * block_size)
        player_pos = (map_position + (self.position - map_pos) * block_size + 0.5).astype(np.int16)

        # copy basic map to screen
        self.screen.blit(self.screen_map, map_position, map_rect)

        # draw the area of player view using ray data
        coords = np.vstack((
            self.screen_view_area_size / 2,  # middle of surface as start and end point
            self.screen_view_area_size / 2 + (self.ray_data[self.grid_blocks, 2:4] - self.position) * block_size + 0.5
            )).astype(np.int16)

        # make a variant to show only the map already seen
        rect = pygame.draw.polygon(self.screen_view_area, (255, 255, 255), coords, 0)
        # add shading to view area
        self.screen_view_area.blit(self.screen_shade, rect, rect, pygame.BLEND_RGB_MULT)
        self.screen_map_visited.blit(self.screen_view_area, self.position * block_size - self.screen_view_area_size // 2 + rect[0:2], rect, pygame.BLEND_RGB_MAX)

        # make another variant showing the current view in yellow
        self.screen_view_area.blit(self.screen_shade_yellow, rect, rect, pygame.BLEND_RGB_MULT)
        # copy to main screen
        screen.blit(self.screen_view_area, player_pos - self.screen_view_area_size // 2 + rect[0:2], rect, pygame.BLEND_RGB_ADD)

        # clear view area after it has been used
        self.screen_view_area.fill(self.background_color, rect)

        # add "player"
        p_size = min(max(3, block_size / 4), 8)
        pygame.draw.circle(screen, (0, 0, 200), player_pos, p_size, 0)

        item_pos = (np.array([0.5, 0.5]) + (self.item_position[self.item_active] - map_pos) * block_size).astype(np.int16)
        items_on_map = map_position + item_pos[
            (item_pos[:, 0] >= 0) &
            (item_pos[:, 0] < self.map_show_size[0] * block_size) &
            (item_pos[:, 1] >= 0) &
            (item_pos[:, 1] < self.map_show_size[1] * block_size)
            ]
        while screen.get_locked():
            screen.unlock()
        rgb_array_dst = pygame.surfarray.pixels3d(screen)
        rgb_array_dst[items_on_map[:, 0], items_on_map[:, 1], :] = np.array([255, 255, 200], dtype=np.uint8)
        rgb_array_dst = None

        # show only the visited area of the map
        self.screen.blit(self.screen_map_visited, map_position, map_rect, pygame.BLEND_RGB_MULT)

        self.measure_time("draw map")

    def add_coin_bar_and_score(self, screen):

        # draw the bar showing how large a part of coins have been collected.
        pygame.draw.rect(screen, (255, 255, 255), self.coin_rect, 1)
        pygame.draw.rect(screen, (180, 180, 180), self.coin_rect + np.array([1, 1, -2, -2]), 1)
        coin_total_nr = np.count_nonzero(self.item_itype == 1)  # itype 1 = coin
        coin_collected_nr = np.count_nonzero(self.item_itype[self.item_active == 0] == 1)  # inactive coins have been collected
        pygame.draw.rect(screen, (255, 200, 100), (
            self.coin_rect[0] + 2,
            self.coin_rect[1] + 2,
            int((self.coin_rect[2] - 4) * coin_collected_nr / coin_total_nr),
            self.coin_rect[3] - 4
            ), 0)

        # add score
        f_screen = self.font_score.render(str(self.score), False, (255, 255, 255))
        f_screen.set_colorkey(self.background_color)
        f_size = f_screen.get_size()
        screen.blit(f_screen, (self.coin_rect[0] + self.coin_rect[2] - f_size[0] - 4, self.coin_rect[1] + self.coin_rect[3] + 4))

        # if all coins collected, game won! Keep visibility...
        if coin_collected_nr == coin_total_nr:
            self.view_blocks_target = max(self.view_blocks_target, 10)
            f_screen = self.font_game_over.render('GAME WON!', True, (255, 255, 255), self.background_color)
            f_screen.set_colorkey(self.background_color)
            f_size = np.asarray(f_screen.get_size())
            screen.blit(f_screen, ((self.screen_size - f_size) / 2).astype(np.int16))

        # if game over, add game over text
        if self.game_over:
            f_screen = self.font_game_over.render('GAME OVER', True, (255, 255, 255), self.background_color)
            f_screen.set_colorkey(self.background_color)
            f_size = np.asarray(f_screen.get_size())
            screen.blit(f_screen, ((self.screen_size - f_size) / 2).astype(np.int16))

    def play_sound(self, sound, channel=None):

        # plays a sound on the next channel (all channels used in order).
        # if channel not specified, sounds will be missed as sometimes all channels are busy - rotates through channels.
        if channel is None:
            ch = pygame.mixer.Channel(self.channel)
            self.channel += 1  # move to next channel
            if self.channel == self.nr_channels:
                self.channel = 1
        else:
            ch = pygame.mixer.Channel(channel)
        ch.play(sound)

    def set_view_blocks(self, view_blocks):

        # adjusts for new vire_blocks viewing distance.
        self.view_blocks = view_blocks
        # the size of map area shown - limit to 2 x view_blocks, but keep at least map_minimum_size
        self.map_show_size = np.maximum(self.map_minimum_size, np.minimum(self.map_size, self.view_blocks * 2).astype(np.int16))
        self.screen_shade = pygame.Surface((self.view_blocks * self.block_size * 2, self.view_blocks * self.block_size * 2), 0, self.screen)
        self.screen_shade_yellow = self.screen_shade.copy()
        self.screen_view_area = self.screen_shade.copy()
        self.screen_view_area.set_colorkey(self.background_color)
        self.screen_view_area_size = np.asarray(self.screen_view_area.get_size(), dtype=np.int16)
        self.setup_shade(self.block_size)
        self.map_position = (self.screen_size - self.map_show_size * self.block_size - 4)

    def setup_shade(self, block_size):

        # draw co-centric circles to screen_shade to enable map viewing area shading.
        circle_nr = int(self.view_blocks * (1.0 - self.view_shade) * block_size)
        center = (self.view_blocks * self.block_size, self.view_blocks * block_size)
        for i in range(circle_nr + 1):
            color = (np.array([255, 255, 255]) * (0.1 + 0.9 * i / circle_nr)).astype(np.uint8)
            radius = self.view_blocks * self.block_size - i
            pygame.draw.circle(self.screen_shade, color, center, radius, 0)

        # make a fully yellow surface for coloring
        self.screen_shade_yellow.fill((255, 200, 100))

    def setup_map(self, screen, block_size):

        # draws the map - the grid and the wall blocks.
        for col in range(self.map_size[0]):
            pygame.draw.line(screen, (80, 80, 80),
                             (col * block_size, 0),
                             (col * block_size, block_size * self.map_size[1]),
                             1)
        for row in range(self.map_size[1]):
            pygame.draw.line(screen, (80, 80, 80),
                             (0, row * block_size),
                             (block_size * self.map_size[0], row * block_size),
                             1)
            for col in range(self.map_size[0]):
                if self.map_array[col, row] > 0:
                    pos = np.array([col * block_size, row * block_size], dtype=np.int16)
                    pygame.draw.rect(screen, (200, 200, 200), (pos[0], pos[1], block_size, block_size), 0)

    def make_map_array(self, map):

        # go through the map, with alphanumeric IDs for each block, and transform it to a processable map data.
        # assumes the map is rectangular ie. that each item (row) in the map has the same number of characters (blocks).
        # coin blocks are converted to a "nothing block" and a a coin item is created in the middle of them.

        if len(self.coin_blocks) > 0:
            coin_y_levels = (np.arange(len(self.coin_blocks)) + 1) / len(self.coin_blocks) * 0.6 + 0.2
        y = 0
        for row in map:
            x = 0
            for block in row:
                # block must be found in the dictionary of alphanumeric blocks
                # however, if block is in coin_blocks or a light block, make it a nothning block and add a coin/light item
                if block in self.coin_blocks:
                    self.setup_coin_item((x + 0.5, y + 0.5), coin_y_levels[self.coin_blocks.index(block)])
                    block = ' '  # make this a "nothing block"
                elif block == self.light_block:
                    self.items.append(Item(2, (x + 0.5, y + 0.5), 0.5, 0.5, [self.pic_lightbulb], 999, 1, self.sound_pling))
                    block = ' '  # make this a "nothing block"
                if block not in self.map_blocks:
                    print('Map block "' + block + '" not found in block dictionary. Set it up in setup_blocks().')
                    self.map_array[x, y] = 0  # not accepting the block...
                else:
                    self.map_array[x, y] = self.map_blocks[block]
                x += 1
            y += 1

    def setup_blocks(self):

        # set up all wall blocks by providing the alphanumeric "key" and image data for its sides.
        # simple blocks have only one image, used for all sides, others may have four (up, right, down, left in the map).
        # each image is given in a list, and if the list has many images, they will be animated.

        block_nr = 0
        # "nothing block" ie no walls or coins etc. at all. Must be the zero block. Using blank space for this.
        self.map_blocks[' '] = block_nr
        self.wall_blocks.append(Block())

        block_nr += 1
        self.map_blocks['1'] = block_nr
        self.wall_blocks.append(Block([[self.pic_alphaks2], [self.pic_vector3d], [self.pic_ball2],
                                       [self.pic_ray01, self.pic_ray02, self.pic_ray03, self.pic_ray04, self.pic_ray05,
                                        self.pic_ray06, self.pic_ray07, self.pic_ray08, self.pic_ray09, self.pic_ray10]]))
        block_nr += 1
        self.map_blocks['2'] = block_nr
        self.wall_blocks.append(Block([[self.pic_guru2]]))
        block_nr += 1
        self.map_blocks['3'] = block_nr
        self.wall_blocks.append(Block([[self.pic_alphaks2]]))
        block_nr += 1
        self.map_blocks['4'] = block_nr
        self.wall_blocks.append(Block([[self.pic_theend]]))
        block_nr += 1
        self.map_blocks['5'] = block_nr
        self.wall_blocks.append(Block([[self.pic_vector3d]]))
        block_nr += 1
        self.map_blocks['6'] = block_nr
        self.wall_blocks.append(Block([[self.pic_ray01, self.pic_ray02, self.pic_ray03, self.pic_ray04, self.pic_ray05,
                                        self.pic_ray06, self.pic_ray07, self.pic_ray08, self.pic_ray09, self.pic_ray10]]))

    def setup_coin_item(self, position, y_level):

        # set up a coin item (itype = 1). Add a random rotation speed element to make coins spin with different frequencies.
        # pic_coin contains six frames of rotating coin
        pic1 = self.pic_coin.subsurface((22, 22), (360, 360))
        pic2 = self.pic_coin.subsurface((416, 22), (732 - 416, 360))
        pic3 = self.pic_coin.subsurface((794, 22), (968 - 794, 360))
        pic4 = self.pic_coin.subsurface((190, 450), (250 - 190, 360))
        pic5 = self.pic_coin.subsurface((418, 450), (590 - 418, 360))
        pic6 = self.pic_coin.subsurface((670, 450), (984 - 670, 360))
        self.items.append(Item(1, position, y_level, 0.3, [pic1, pic2, pic3, pic4, pic5, pic6], 50 + np.random.randint(0, 33), 1, self.sound_pling))

    def setup_items(self):

        # set up items not set up in the map. Example:
        # self.setup_coin_item(self.position + np.array([0, -3]), 0.5)

        # set the item arrasy for quicker processing
        self.item_position = np.asarray([item.position for item in self.items], dtype=float)
        self.item_size = np.asarray([item.size for item in self.items], dtype=float)
        self.item_y_level = np.asarray([item.y_level for item in self.items], dtype=float)
        self.item_ratio = np.asarray([item.image_max_size[0] / (item.image_max_size[1] + 0.00001) for item in self.items], dtype=float)
        self.item_itype = np.asarray([item.itype for item in self.items], dtype=np.int16)
        self.item_active = np.asarray([item.active for item in self.items], dtype=np.bool_)

    def add_frame(self, pic, color):

        # adds a frame to the picture
        size = pic.get_size()
        pygame.draw.rect(pic, color, (1, 1, size[0] - 2, size[1] - 2), 3)
        color2 = [2 * c // 3 for c in color]
        pygame.draw.rect(pic, color2, (2, 2, size[0] - 4, size[1] - 4), 1)
        color3 = [c // 3 for c in color]
        pygame.draw.rect(pic, color3, (3, 3, size[0] - 6, size[1] - 6), 1)

    def toggle_fullscreen(self):

        # toggle between fullscreen and windowed mode.
        pygame.display.toggle_fullscreen()

    def toggle_map(self):

        # map display on/off
        self.show_map = not(self.show_map)

    def toggle_info_display(self):

        # info display on/off
        self.info_display = not(self.info_display)

    def plot_info(self):

        # show info
        while self.screen_info.get_locked():
            self.screen_info.unlock()
        self.screen_info.fill(self.background_color)

        y = 0
        info_msg = f'frames per sec: {int(self.clock.get_fps()):5.0f}'
        self.plot_info_msg(self.screen_info, 0, y, info_msg)
        y += 15
        info_msg = f'position:       {self.position[0]:5.2f}  {self.position[1]:5.2f}'
        self.plot_info_msg(self.screen_info, 0, y, info_msg)
        y += 15
        info_msg = f'angle & speed:  {self.pos_angle:5.0f}  {self.move_speed:5.2f}'
        self.plot_info_msg(self.screen_info, 0, y, info_msg)
        y += 15
        if self.ray_data[self.width // 2, 4] < self.view_blocks:
            info_msg = f'dist. to wall:  {self.ray_data[self.width // 2, 4]:5.2f}'
        else:
            info_msg = f'dist. to wall:  [n/a]'
        self.plot_info_msg(self.screen_info, 0, y, info_msg)
        y += 15
        # add measured times as percentage of total
        tot_time = np.sum(self.timers)
        if tot_time > 0:
            for i in range(len(self.timer_names)):
                info_msg = (self.timer_names[i] + ' '*16)[:16] + f'{np.sum(self.timers[i, :]) * 100 / tot_time:5.1f}'
                self.plot_info_msg(self.screen_info, 0, y + i * 15, info_msg)

        # copy to screen
        self.screen.blit(self.screen_info, (10, 10))

        self.measure_time("plot info")

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


class Block:

    def __init__(self, image_list=None):

        # if no image list given, set up as such (this is normally the " " block i.e. not a wall at all)
        if image_list is None:
            self.image_cnt = np.zeros((4))
        else:
            self.images = image_list  # image list must be a list of lists of images; but there may be only one image.
            # if given only one image (or image list) when set up, use it for all sides
            if len(image_list) == 1:
                self.images.append(image_list[0])
                self.images.append(image_list[0])
                self.images.append(image_list[0])
            self.image_cnt = np.array([len(self.images[0]), len(self.images[1]), len(self.images[2]), len(self.images[3])])

        self.image_used = np.zeros((4), dtype=np.float16)  # this is a float to keep accuracy; int() used when using this.
        self.image_cnt_max = np.amax(self.image_cnt)

    def animate(self, img_add):

        # img_add should determine how may images forward to go in the animation list (and may be e.g. 0.4) and naturally should depend on time.
        if self.image_cnt_max > 1:
            self.image_used = np.mod((self.image_used + img_add), self.image_cnt)


class Item:

    def __init__(self, itype, position, y_level, size, image_list, ms_per_image, active, collide_sound):

        # y_level and size are in percentage or room height and y_level is for the mid point of item. Hence, making sure size / 2 < y_level < (1 - size / 2)
        self.itype = itype  # defines item type: 1 = coin, 2 = light bulb
        self.position = np.asarray(position, dtype=float)  # position as map_x, map_y
        self.size = min(1.0, max(0.01, size))  # size as percentage of room height (floor to ceiling). Limited to between 1 and 100 %.
        self.y_level = min((1.0 - self.size / 2), max(self.size / 2, y_level))  # position as mid point as percentage of room height (floor to ceiling)
        self.active = active  # when setting up this is normally 1
        self.collide_sound = collide_sound
        # image list may be empty or contain one or more images. If >1 they will be animated.
        if image_list is None:
            self.image_cnt = 0
            self.image_max_size = np.array([0, 0])
        else:
            self.images = image_list  # there may be only one image.
            self.image_cnt = len(image_list)
            self.image_used = 0
            self.image_size = np.asarray([i.get_size() for i in image_list], dtype=np.int16)
            self.image_max_size = np.max(self.image_size, axis=0)
            self.ms_per_image = ms_per_image  # nr of milliseconds to show each image (approximate)

    def animate(self, time):

        # selects the image to use based on current time and how long (in milliseconds( each image is to be used.
        if self.image_cnt > 1:
            self.image_used = int((time / self.ms_per_image) % self.image_cnt)

    def collide(self):

        self.active = 0


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
    # disp_size = (1280, 720)
    disp_size = (720, 400)
    # disp_size = (640, 360)

    pygame.font.init()
    pygame.mixer.init()
    music_file = "alacrity.ogg"  # this mod by Jellybean is available at e.g. http://janeway.exotica.org.uk/
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play(loops=-1)

    screen = pygame.display.set_mode(disp_size)
    pygame.display.set_caption('RayCastingGame')
    RayCastingGame(screen, 60).run()

    # exit; close display, stop music
    pygame.quit()
    exit()
