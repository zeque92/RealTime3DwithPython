# -*- coding: utf-8 -*-
import numpy as np
import random
import pygame  # needed only if running stand-alone and showing maze generation. (If not imported code must be modified, though.)
from os.path import exists
from sys import exit


class Maze:
    """
    Generate a maze See https://en.wikipedia.org/wiki/Maze_generation_algorithm
    Returns either
        a (y, x, 2) size Numpy Array with 0 as a passage and 1 as a wall for the down and right walls of each cell; outer edges are always walls.
        a (y * 2 + 1, x * 2 + 1) size Numpy Array with 0 as a corridor and 1 as a wall block; outer edges are wall blocks.

    @author: kalle
    """

    def __init__(self, size_x, size_y):

        self.screen = None  # if this is set to a display surface, maze generation will be shown in a window
        self.screen_size = None
        self.screen_block_size = None
        self.screen_block_offset = None
        self.prev_update = 0
        self.clock = pygame.time.Clock()
        self.slow_mode = False
        self.running = True

        # initialize an array of walls filled with ones, data "not visited" + if exists for 2 walls (down and right) per cell.
        self.wall_size = np.array([size_y, size_x], dtype=np.int)
        # add top, bottom, left, and right (ie. + 2 and + 2) to array size so can later work without checking going over its boundaries.
        self.walls = np.ones((self.wall_size[0] + 2, self.wall_size[1] + 2, 3), dtype=np.byte)
        # mark edges as "unusable" (-1)
        self.walls[:, 0, 0] = -1
        self.walls[:, self.wall_size[1] + 1, 0] = -1
        self.walls[0, :, 0] = -1
        self.walls[self.wall_size[0] + 1, :, 0] = -1

        # initialize an array of block data - each passage (0) and wall (1) is a block
        self.block_size = np.array([size_y * 2 + 1, size_x * 2 + 1], dtype=np.int)
        self.blocks = np.ones((self.block_size[0], self.block_size[1]), dtype=np.byte)

    def gen_maze_walls(self, corridor_len=999):

        # Generate a maze.
        # This will start with a random cell and create a corridor until corridor length >= corridor_len or it cannot continue the current corridor.
        # Then it will continue from a random point within the current maze (ie. visited cells), creating a junction, until no valid starting points exist.
        # Setting a small corridor maximum length (corridor_len) will cause more branching / junctions.
        # Returns maze walls data - a NumPy array size (y, x, 2) with 0 or 1 for down and right cell walls.

        # set a random starting cell and mark it as "visited"
        cell = np.array([random.randrange(2, self.wall_size[0]), random.randrange(2, self.wall_size[1])], dtype=np.int)
        self.walls[cell[0], cell[1], 0] = 0

        # a simple definition of the four neighboring cells relative to current cell
        up    = np.array([-1,  0], dtype=np.int)
        down  = np.array([ 1,  0], dtype=np.int)
        left  = np.array([ 0, -1], dtype=np.int)
        right = np.array([ 0,  1], dtype=np.int)

        # preset some variables
        need_cell_range = False
        round_nr = 0
        corridor_start = 0
        if corridor_len <= 4:
            corridor_len = 5  # even this is too small usually

        while np.size(cell) > 0 and self.running:

            round_nr += 1
            # get the four neighbors for current cell (cell may be an array of cells)
            cell_neighbors = np.vstack((cell + up, cell + left, cell + down, cell + right))
            # valid neighbors are the ones not yet visited
            valid_neighbors = cell_neighbors[self.walls[cell_neighbors[:, 0], cell_neighbors[:, 1], 0] == 1]

            if np.size(valid_neighbors) > 0:
                # there is at least one valid neighbor, pick one of them (at random)
                neighbor = valid_neighbors[random.randrange(0, np.shape(valid_neighbors)[0]), :]
                if np.size(cell) > 2:
                    # if cell is an array of cells, pick one cell with this neighbor only, at random
                    cell = cell[np.sum(abs(cell - neighbor), axis=1) == 1]  # cells where distance to neighbor == 1
                    cell = cell[random.randrange(0, np.shape(cell)[0]), :]
                # mark neighbor visited
                self.walls[neighbor[0], neighbor[1], 0] = 0
                # remove the wall between current cell and neighbor. Applied to down and right walls only so may be that of the cell or the neighbor
                self.walls[min(cell[0], neighbor[0]), min(cell[1], neighbor[1]), 1 + abs(neighbor[1] - cell[1])] = 0
                if self.screen is not None:
                    # if screen is set, draw the corridor from cell to neighbor
                    self.draw_cell(cell, neighbor)
                # check if more corridor length is still available
                if round_nr - corridor_start < corridor_len:
                    # continue current corridor: set current cell to neighbor
                    cell = np.array([neighbor[0], neighbor[1]], dtype=np.int)
                else:
                    # maximum corridor length fully used; make a new junction and continue from there
                    need_cell_range = True

            else:
                # no valid neighbors for this cell
                if np.size(cell) > 2:
                    # if cell already contains an array of cells, no more valid neighbors are available at all
                    cell = np.zeros((0, 0))  # this will end the while loop, the maze is finished.
                    if self.screen is not None:
                        # if screen is set, make sure it is updated as the maze is now finished.
                        pygame.display.flip()
                else:
                    # a dead end; make a new junction and continue from there
                    need_cell_range = True

            if need_cell_range:
                # get all visited cells (=0) not marked as "no neighbors" (=-1), start a new corridor from one of these (make a junction)
                cell = np.transpose(np.nonzero(self.walls[1:-1, 1:-1, 0] == 0)) + 1  # not checking the edge cells, hence needs the "+ 1"
                # check these for valid neighbors (any adjacent cell with "1" as visited status (ie. not visited) is sufficient, hence MAX)
                valid_neighbor_exists = np.array([self.walls[cell[:, 0] - 1, cell[:, 1], 0],
                                                  self.walls[cell[:, 0] + 1, cell[:, 1], 0],
                                                  self.walls[cell[:, 0], cell[:, 1] - 1, 0],
                                                  self.walls[cell[:, 0], cell[:, 1] + 1, 0]
                                                  ]).max(axis=0)
                # get all visited cells with no neighbors
                cell_no_neighbors = cell[valid_neighbor_exists != 1]
                # mark these (-1 = no neighbors) so they will no longer be actively used. This is not required but helps with large mazes.
                self.walls[cell_no_neighbors[:, 0], cell_no_neighbors[:, 1], 0] = -1
                corridor_start = round_nr + 0  # start a new corridor.
                need_cell_range = False

        # return: drop out the additional edge cells. All cells visited anyway so just return the down and right edge data.
        if self.running:
            return self.walls[1:-1, 1:-1, 1:3]

    def gen_maze_2D(self, corridor_len=999):

        # converts walls data from gen_maze_walls to a NumPy array size (y * 2 + 1, x * 2 + 1)
        # wall blocks are represented by 1 and corridors by 0.

        self.gen_maze_walls(corridor_len)

        if self.running:
            # use wall data to set final output maze
            self.blocks[1:-1:2, 1:-1:2] = 0  # every cell is visited if correctly generated
            # horizontal walls
            self.blocks[1:-1:2, 2:-2:2] = self.walls[1:-1, 1:-2, 2]  # use the right wall
            # vertical walls
            self.blocks[2:-2:2, 1:-1:2] = self.walls[1:-2, 1:-1, 1]  # use the down wall

            return self.blocks

    def draw_cell(self, cell, neighbor):

        # draw passage from cell to neighbor. As these are always adjacent can min/max be used.
        min_coord = (np.flip(np.minimum(cell, neighbor) * 2 - 1) * self.screen_block_size + self.screen_block_offset).astype(np.int16)
        max_coord = (np.flip(np.maximum(cell, neighbor) * 2 - 1) * self.screen_block_size + int(self.screen_block_size) + self.screen_block_offset).astype(np.int16)
        pygame.draw.rect(self.screen, (200, 200, 200), (min_coord, max_coord - min_coord))

        if self.slow_mode or pygame.time.get_ticks() > self.prev_update + 50:
            self.prev_update = pygame.time.get_ticks()
            pygame.display.flip()

            # when performing display flip, handle some pygame events as well.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.key == pygame.K_f:
                        self.toggle_fullscreen()
                    if event.key == pygame.K_m:
                        self.toggle_slow_mode()

        if self.slow_mode:
            pygame.time.wait(3)

    def toggle_slow_mode(self):

        # switch between a windowed display and full screen
        self.slow_mode = not(self.slow_mode)

    def toggle_fullscreen(self):

        # toggle between fullscreen and windowed mode
        screen_copy = self.screen.copy()
        pygame.display.toggle_fullscreen()
        self.screen.blit(screen_copy, (0, 0))
        pygame.display.flip()

    def save_image(self):

        # save maze as a png image. Use the first available number to avoid overwriting a previous image.
        for file_nr in range(1, 1000):
            file_name = 'Maze_' + ('00' + str(file_nr))[-3:] + '.png'
            if not exists(file_name):
                pygame.image.save(self.screen, file_name)
                break


if __name__ == '__main__':

    # Run and display the Maze.
    # Left mouse button or space bar: generate a new maze. Up and down cursor keys: change maze block size. s: save the maze image.
    # ESC or close the window: Quit.

    # set screen size and initialize it
    pygame.display.init()
    disp_size = (1920, 1080)
    rect = np.array([0, 0, disp_size[0], disp_size[1]])  # the rect inside which to draw the maze. Top x, top y, width, height.
    block_size = 10  # block size in pixels
    screen = pygame.display.set_mode(disp_size)
    pygame.display.set_caption('Maze Generator / KS 2022')
    running = True

    while running:

        # intialize a maze, given size (y, x)
        maze = Maze(rect[2] // (block_size * 2) - 1, rect[3] // (block_size * 2) - 1)
        maze.screen = screen  # if this is set, the maze generation process will be displayed in a window. Otherwise not.
        screen.fill((0, 0, 0))
        maze.screen_size = np.asarray(disp_size)
        maze.screen_block_size = np.min(rect[2:4] / np.flip(maze.block_size))
        maze.screen_block_offset = rect[0:2] + (rect[2:4] - maze.screen_block_size * np.flip(maze.block_size)) // 2

        # generate the maze - parameter: corridor length (optional)
        start_time = pygame.time.get_ticks()
        print(f'Generating a maze of {maze.wall_size[1]} x {maze.wall_size[0]} = {maze.wall_size[0] * maze.wall_size[1]} cells. Block size = {block_size}.')
        maze.gen_maze_2D()
        if maze.running:
            print('Ready. Time: {:0.2f} seconds.'.format((pygame.time.get_ticks() - start_time) / 1000.0))
            print('ESC or close the Maze window to end program. SPACE BAR for a new maze. UP & DOWN cursor keys to change block size. s to save maze image.')
        else:
            print('Aborted.')

        # wait for exit (window close or ESC key) or left mouse button (new maze) or key commands
        pygame.event.clear()  # clear the event queue
        running = maze.running
        pausing = maze.running
        while pausing:
            event = pygame.event.wait()  # wait for user input, yielding to other prcesses
            if event.type == pygame.QUIT:
                pausing = False
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pausing = False
                if event.key == pygame.K_f:
                    maze.toggle_fullscreen()
                if event.key == pygame.K_s:
                    # save screen as png image
                    maze.save_image()
                if event.key == pygame.K_DOWN:
                    block_size -= 1
                    if block_size < 1:
                        block_size = 1
                    pausing = False
                if event.key == pygame.K_UP:
                    block_size += 1
                    if block_size > min(rect[2], rect[3]) // 10:
                        block_size = min(rect[2], rect[3]) // 10
                    pausing = False
                if event.key == pygame.K_ESCAPE:
                    pausing = False
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # left button: new maze
                    pausing = False

    # exit; close display
    pygame.quit()
    exit()

