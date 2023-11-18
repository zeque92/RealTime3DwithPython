# -*- coding: utf-8 -*-
import pygame
import numpy as np
from sys import exit


class RayCasting:
    """
    2.5 dimensional environment - raycasting / Doom style.

    Current version: Map only.

    @author: kalle
    """

    def __init__(self, screen, target_fps):

        self.screen = screen
        self.screen_map = screen.copy()
        self.screen_view = screen.copy()
        self.target_fps = target_fps
        self.screen_size = np.asarray(self.screen.get_size(), dtype=np.int16)
        self.width = self.screen_size[0]
        self.height = self.screen_size[1]
        self.mid_screen = self.screen_size // 2
        self.background_color = (0, 0, 0)
        self.screen_info = pygame.Surface((300, 200), 0, self.screen)
        self.screen_info.set_colorkey(self.background_color)
        self.mode = 1
        self.move_speed = 1.0
        self.angle_speed = 0.8
        self.running = True
        self.paused = False
        self.info_display = True
        self.clock = pygame.time.Clock()

        self.map = ['111111111111111111111111111',
                    '100000000000000000000000001',
                    '100000001000000000000000001',
                    '100000011011000001100000001',
                    '100000010010000001100000001',
                    '100000000110000000000000001',
                    '100100000000000000000000001',
                    '100000000000000000001100001',
                    '100000000000000000000000001',
                    '100000000000000000000010001',
                    '100000001100000000111010001',
                    '100000011111000001101110001',
                    '100000000000000000000000001',
                    '100000000011000000000000001',
                    '111111111111111111111111111']
        self.map_array = np.zeros((len(self.map[0]), len(self.map)), dtype=np.uint8)
        self.map_blocks = {'0': 0}  # initialize block dictionary with an empty (0) block
        self.block_size = 32
        self.make_map_array(self.map)
        self.map_size = np.array(np.shape(self.map_array))
        self.position = self.map_size / 2  # initial position at center
        self.pos_angle = 0.0
        self.stuck = False
        self.view_angle = 0.0
        self.view_height = 0.5  # "eye height" as percentage
        self.view_blocks = 10   # how many blocks away can see
        self.view_shade = 0.7   # fraction of view after which shading starts
        self.view_width = 60    # view width angle in degrees (horizontally)
        self.proj_dist = np.tan(self.view_width * np.pi / 180) * self.width
        self.angle_add = self.view_width / self.width * np.pi / 180

        # ray_data is an array of grid item, side (0 = up, 1 = right, 2 = down, 3 = left) map x, map y, and distance from viewer for each ray
        self.ray_data = np.zeros((self.width, 5), dtype=float)
        self.grid_blocks = np.zeros((0))

        self.screen_shade = pygame.Surface((self.view_blocks * self.block_size * 2, self.view_blocks * self.block_size * 2), 0, self.screen)
        self.screen_view_area = self.screen_shade.copy()
        self.screen_view_area.set_colorkey(self.background_color)
        self.screen_view_area_size = np.asarray(self.screen_view_area.get_size(), dtype=np.int16)
        self.setup_shade(self.block_size)
        self.map_rect = self.draw_map(self.screen_map, self.block_size)

        # the following for checking performance only
        self.info_display = True
        self.millisecs = 0
        self.timer_avg_frames = 180
        self.timer_names = []
        self.timers = np.zeros((1, 1), dtype=int)
        self.timer_frame = 0
        self.font = pygame.font.SysFont('CourierNew', 15)

        # set up timers
        self.timer_name = []
        self.timer_names.append("clear")
        self.timer_names.append("calculate")
        self.timer_names.append("draw")
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
        self.millisecs = pygame.time.get_ticks()

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
                    if event.key == pygame.K_m:
                        if self.mode == 1:
                            self.mode = 2
                        else:
                            self.mode = 1
                    if event.key == pygame.K_s:
                        # save screen, at half the resolution, using class name as file name
                        # pygame.image.save(pygame.transform.scale(self.screen, (int(self.screen.get_size()[0] / 2), int(self.screen.get_size()[1] / 2))),
                        #                   self.__class__.__name__ + '.jpg')
                        pygame.image.save(self.screen, self.__class__.__name__ + '.jpg')
                    if event.key == pygame.K_i:
                        self.toggle_info_display()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # left button: exit
                        self.running = False

            if not self.paused:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    self.pos_angle -= self.angle_speed * self.move_speed
                    if self.pos_angle < 0:
                        self.pos_angle += 360
                if keys[pygame.K_RIGHT]:
                    self.pos_angle += self.angle_speed * self.move_speed
                    if self.pos_angle >= 360:
                        self.pos_angle -= 360
                if keys[pygame.K_UP]:
                    self.move_speed += 0.1
                    if self.move_speed > 3.0:
                        self.move_speed = 3.0
                if keys[pygame.K_DOWN]:
                    self.move_speed -= 0.1
                    if self.move_speed < 0.0:
                        self.move_speed = 0.0

            self.clear()
            if not self.paused:
                self.move()
            self.raycast(self.screen, self.block_size)
            self.draw_map_view(self.screen, self.block_size)
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

    def clear(self):

        self.screen.fill(self.background_color, (10, 10, 310, 210))  # clear plot info area
        self.screen.blit(self.screen_map, self.map_rect, self.map_rect)
        self.measure_time("clear")

    def move(self):

        # check for collision: if any ray hits a grid block sooner than move speed, it's a collision
        # collisions = (np.minimum(self.ray_data - self.move_speed * 0.01, 0)).nonzero()[0]

        move_time_adj = 0.01
        bounce_angle = 40  # if collision angle is less than this, then bounce
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
                self.move_speed *= 0.8
                self.pos_angle = new_angle
            else:
                # no bounce, get "stuck"
                new_pos = self.position
                self.stuck = True
                if self.move_speed > 1.0:
                    self.move_speed = 1.0
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
        view = self.view_blocks  # nr of blocks: how far is visible (and, hence, how far must be calculated)
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
        tan_a = np.tan(rads)
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
        vert_grid = np.hstack((
            self.map_array[(np.minimum(np.maximum(x_coord[:, :view], 0), self.map_size[0] - 1)).astype(np.int16),
                           (np.minimum(np.maximum(x_coord_y[:, :view], 0), self.map_size[1] - 1)).astype(np.int16)],
            np.ones((width, 1), dtype=np.int16)  # this is the farthest block and always marked as "1" to make sure argmax finds something.
            ))
        # find the first nonzero item's index for each angle. Since distance is growing with each item, this is the closest wall encountered for each ray.
        vert_grid_first = (vert_grid != 0).argmax(axis=1)
        # construct an array of grid item, side (0 = up, 1 = right, 2 = down, 3 = left) map x, map y, and distance from viewer
        vert_data = (np.vstack((
            vert_grid[rng_width, vert_grid_first],
            sign_y + 1,
            x_coord[rng_width, vert_grid_first],
            x_coord_y_adj[rng_width, vert_grid_first],
            np.abs((x_coord[rng_width, vert_grid_first] - self.position[0]) / sin_a)
            ))).transpose()

        # horizontal intersections of ray and grid; x coordinate increases/decreases by one, calculate respective y coordinate
        horiz_grid_range = rng_view
        y_coord = self.position[1] + y_first[:, None] - horiz_grid_range * (sign_x / tan_a)[:, None]
        y_coord_x = pos[0] + (horiz_grid_range + 1) * sign_x[:, None]
        y_coord_x_adj = y_coord_x - (np.sign(y_coord_x - pos[0]) - 1) / 2
        horiz_grid = np.hstack((
            self.map_array[(np.minimum(np.maximum(y_coord_x[:, :view], 0), self.map_size[0] - 1)).astype(np.int16),
                           (np.minimum(np.maximum(y_coord[:, :view], 0), self.map_size[1] - 1)).astype(np.int16)],
            np.ones((width, 1), dtype=np.int16)  # this is the farthest block and always marked as "1" to make sure argmax finds something.
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
        self.ray_data = comb_data[comb_closest, rng_width, :]

        # figure out where each block starts and ends. If the same block and side are used continuously, this will be treated as one block.
        # grid_changes contains the beginning of each block. These blocks can later be drawn in one go each as the source image is the same. Always include first and last.
        # 1. np.diff: get the difference of each block to the preceding in each of [block, side, x_coord, y_coord].
        # 2. sum the absolutes of these for each ray.
        # 3. convert to int16 and get the indexes of each non-zero change. This means block and side changes are always accounted for (are >= 1(, but small changes
        #    (same or the very next) in grid position are not (are < 1).
        # 4. combine the changes so that also the last ray of the preceding block is included, and add the first and last rays.
        grid_changes = (np.sum(np.abs(np.diff(self.ray_data[:-1, :4], n=1, axis=0)), axis=1)).astype(np.int16).nonzero()[0] + 1
        self.grid_blocks = np.zeros((np.shape(grid_changes)[0] * 2 + 2), np.int16)
        # grid_blocks[0] = 0                    # first ray - this is already zero
        self.grid_blocks[1:-1:2] = grid_changes - 1  # the last ray in the block before change
        self.grid_blocks[2:-1:2] = grid_changes      # the first ray of the block after change
        self.grid_blocks[-1] = width - 1             # the last ray

        self.measure_time("calculate")

        # below for drawing calculation data

        if self.mode == 2:
            for k in range(10):
                i = int(k * (width - 1) / 9)
                # pos_x, pos_y = self.position
                # use_pos = np.array([(self.width - block_size * self.map_size[0]) // 2 + pos_x * block_size,
                #                     (self.height - block_size * self.map_size[1]) // 2 + pos_y * block_size], dtype=np.int16)
                for j in range(np.shape(x_coord)[1]):
                    if x_coord[i, j] < 0 or x_coord[i, j] > self.map_size[0] or x_coord_y_adj[i, j] < 0 or x_coord_y_adj[i, j] > self.map_size[1]:
                        break
                    # prev_pos = use_pos.copy()
                    use_pos = np.array([(self.width - block_size * self.map_size[0]) // 2 + x_coord[i, j] * block_size,
                                        (self.height - block_size * self.map_size[1]) // 2 + x_coord_y_adj[i, j] * block_size], dtype=np.int16)
                    # pygame.draw.aaline(screen, (120, 120, 120), prev_pos, use_pos)
                    if vert_grid[i, j] == 1:
                        pygame.draw.circle(screen, (200, 0, 0), use_pos, 5, 0)
                    else:
                        pygame.draw.circle(screen, (200, 0, 0), use_pos, 3, 1)

            for k in range(10):
                i = int(k * (width - 1) / 9)
                for j in range(np.shape(y_coord)[1]):
                    if y_coord_x_adj[i, j] < 0 or y_coord_x_adj[i, j] > self.map_size[0] or y_coord[i, j] < 0 or y_coord[i, j] > self.map_size[1]:
                        break
                    use_pos = np.array([(self.width - block_size * self.map_size[0]) // 2 + y_coord_x_adj[i, j] * block_size,
                                        (self.height - block_size * self.map_size[1]) // 2 + y_coord[i, j] * block_size], dtype=np.int16)
                    if horiz_grid[i, j] == 1:
                        pygame.draw.circle(screen, (0, 200, 0), use_pos, 5, 0)
                    else:
                        pygame.draw.circle(screen, (0, 200, 0), use_pos, 3, 1)

    # =============================================================================
    #             # add a yellow line from viewer to the closest wall
    #             pos_x, pos_y = closest[i, 2], closest[i, 3]
    #             use_pos = np.array([(self.width - block_size * self.map_size[0]) // 2 + pos_x * block_size,
    #                                 (self.height - block_size * self.map_size[1]) // 2 + pos_y * block_size], dtype=np.int16)
    #             pygame.draw.aaline(self.screen, (255, 200, 100), player_pos, use_pos)
    # =============================================================================

    def draw_map_view(self, screen, block_size):

        # draws the player view on the map.

        player_pos = ((self.screen_size - self.map_size * block_size) // 2 + self.position * block_size).astype(np.int16)

        # draw a yellow area of player view
        if self.mode == 1:
            coords = np.vstack((
                self.screen_view_area_size // 2,  # middle of surface
                self.screen_view_area_size // 2 + (self.ray_data[self.grid_blocks, 2:4] - self.position) * block_size
                )).astype(np.int16)

            rect = pygame.draw.polygon(self.screen_view_area, (255, 200, 100), coords, 0)

            # add shading
            self.screen_view_area.blit(self.screen_shade, rect, rect, pygame.BLEND_RGB_MULT)
            # copy to main screen
            screen.blit(self.screen_view_area, player_pos - self.screen_view_area_size // 2 + rect[0:2], rect, pygame.BLEND_RGB_ADD)
            # clear view area
            self.screen_view_area.fill(self.background_color, rect)
        # add player
        pygame.draw.circle(screen, (0, 0, 200), player_pos, 6, 0)

        self.measure_time("draw")

    def setup_shade(self, block_size):

        # draw co-centric circles to screen_shade to enable map viewing area shading.
        circle_nr = int(self.view_blocks * (1.0 - self.view_shade) * block_size)
        center = (self.view_blocks * self.block_size, self.view_blocks * block_size)
        for i in range(circle_nr + 1):
            color = (np.array([255, 255, 255]) * (0.1 + 0.9 * i / circle_nr)).astype(np.uint8)
            radius = self.view_blocks * self.block_size - i
            pygame.draw.circle(self.screen_shade, color, center, radius, 0)

    def draw_map(self, screen, block_size):

        # draws the map and returns the used rect (with a small margin)

        for col in range(self.map_size[0]):
            pygame.draw.line(screen, (80, 80, 80),
                             ((self.width - block_size * self.map_size[0]) // 2 + col * block_size, (self.height - block_size * self.map_size[1]) // 2),
                             ((self.width - block_size * self.map_size[0]) // 2 + col * block_size, (self.height + block_size * self.map_size[1]) // 2),
                             1)
        for row in range(self.map_size[1]):
            pygame.draw.line(screen, (80, 80, 80),
                             ((self.width - block_size * self.map_size[0]) // 2, (self.height - block_size * self.map_size[1]) // 2 + row * block_size),
                             ((self.width + block_size * self.map_size[0]) // 2, (self.height - block_size * self.map_size[1]) // 2 + row * block_size),
                             1)
            for col in range(self.map_size[0]):
                if self.map_array[col, row] > 0:
                    pos = np.array([(self.width - block_size * self.map_size[0]) // 2 + col * block_size,
                                    (self.height - block_size * self.map_size[1]) // 2 + row * block_size], dtype=np.int16)
                    pygame.draw.rect(screen, (200, 200, 200), (pos[0], pos[1], block_size, block_size), 0)

        return ((self.width - block_size * self.map_size[0]) // 2 - 6, (self.height - block_size * self.map_size[1]) // 2 - 6,
                block_size * self.map_size[0] + 12, block_size * self.map_size[1] + 12)

    def make_map_array(self, map):

        # go through the map, with alphanumeric IDs for each block, and tranform it to a processable map data.
        # assumes the map is rectangular ie. that each item (row) in the map has the same number of characters (blocks).
        block_nr = len(self.map_blocks)
        y = 0
        for row in map:
            x = 0
            for block in row:
                # create a dictionary of alphanumeric blocks to keep track of them and use their respective number in self.map_array
                if block not in self.map_blocks:
                    self.map_blocks[block] = block_nr
                    block_nr += 1
                self.map_array[x, y] = self.map_blocks[block]
                x += 1
            y += 1

    def toggle_fullscreen(self):

        # toggle between fullscreen and windowed mode
        pygame.display.toggle_fullscreen()

    def pause(self):

        if self.paused:
            self.paused = False
        else:
            self.paused = True
            self.pause_timer = pygame.time.get_ticks()

    def toggle_info_display(self):

        # switch between a windowed display and full screen
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

    screen = pygame.display.set_mode(disp_size)  # pygame.FULLSCREEN | pygame.DOUBLEBUF
    pygame.display.set_caption('RayCasting Map')
    RayCasting(screen, 60).run()

    # exit; close display, stop music
    pygame.quit()
    exit()