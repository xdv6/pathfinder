import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=30):

        self.screen_width = 401
        self.screen_height = 1172

        # The size of a single grid square in pixels
        self.pix_square_size = size

        # background image with tracks
        self.map = pygame.image.load("./afbeeldingen/straat_padded.png")

        # list with the four corner points of the agent
        self.four_points = []

        # bool dat weergeeft of agent nog niet gecrashed is
        self.is_dead = False

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0, 0]), high=np.array([self.screen_width-1, self.screen_height-1]), shape=(2,), dtype=int),
                "target": spaces.Box(low=np.array([0, 0]), high=np.array([self.screen_width-1, self.screen_height-1]), shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            # tijdelijke noop
            # 4: np.array([0,0])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):

        # startpunt van agent (vastgezet op bepaalde pixels), positie afhankelijk van gridcelgrootte
        coo_agent_x = int(90/ self.pix_square_size)
        coo_agent_y = int(990/self.pix_square_size)

        self._agent_location = np.array([coo_agent_x, coo_agent_y], dtype=np.int32)

        # targetlocatie
        coo_target_x = int(240/ self.pix_square_size)
        coo_target_y = int(120/self.pix_square_size)

        self._target_location = np.array([coo_target_x, coo_target_y], dtype=np.int32)

        # print(f"agent location: {self._agent_location}")
        # print(f"target location: {self._target_location}")

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def check_collision(self):
        self.is_dead = False
        for p in self.four_points:
            # kijken of punt wit is
            if self.map.get_at((int(p[0]*self.pix_square_size), int(p[1]*self.pix_square_size))) == (255, 255, 255):
                # print(f"punt: {p[0]},{p[1]} => DEAD")
                self.is_dead = True
                break

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # change position of agent (position is top left corner)
        self._agent_location = self._agent_location + direction

        # caculate 4 collision points
        left_top = self._agent_location
        right_top = self._agent_location + np.array([1, 0])
        left_bottom = self._agent_location + np.array([0, 1])
        right_bottom = self._agent_location + np.array([1, 1])
        # print(f"left_top: {left_top}, right_top: {right_top}, left_bottom: {left_bottom}, right_bottom: {right_bottom}")

        self.four_points = [left_top, right_top, left_bottom, right_bottom]
        
        # check if agent collided with the border
        self.check_collision()

        observation = self._get_obs()
        info = self._get_info()

        # An episode is done if the agent has collided with the border 
        if self.is_dead:
            
            # mogelijkheid 1: stoppen aan bordern
            # direction terug aftrekken want mag niet af pad gaan
            self._agent_location = self._agent_location - direction
            self.is_dead = False


            # mogelijkheid 2: punishen of crashen
            # reward = -1
            # terminated = True
            # return observation, reward, False, False, info
        
        # or reached the target
        terminated = np.array_equal( self._agent_location, self._target_location)

        reward = 1 if terminated else -0.1  # Binary sparse rewards


        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_width, self.screen_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.image.load("./afbeeldingen/straat_padded.png")
        

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            # syntax: Rect(top_left_corner_postion, (width, height))
            pygame.Rect(
                self.pix_square_size * self._target_location,
                (self.pix_square_size, self.pix_square_size),
            ),
        )

        # Now we draw the agent
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                self.pix_square_size * self._agent_location,
                (self.pix_square_size, self.pix_square_size),
            ),

        )

        # Finally, add some gridlines (gridlines om actiespace te visualiseren)

        # horizontale lijnen
        # for x in range(self.screen_height + 1):
        #     # syntax line(surface, color, start_pos, end_pos) -> Rect
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, self.pix_square_size * x),
        #         (self.screen_width, self.pix_square_size * x),
        #         width=3,
        #     )

        # # verticale lijnen
        # for x in range(self.screen_width + 1):

        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (self.pix_square_size * x, 0),
        #         (self.pix_square_size * x, self.screen_height),
        #         width=3,
        #     )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
