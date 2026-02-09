import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from enum import IntEnum
import os


class ACTION(IntEnum):
    """Actions available in the Socks grid environment."""
    LEFT = 0
    RIGHT = 1
    TOP = 2
    DOWN = 3
    PICK_UP = 4
    PUT = 5


class SocksGridEnv(gym.Env):
    """
    A simple grid environment where a dog needs to put socks into a target place.
    
    Grid: 4x4
    Actions: left, right, top, down, pick up, put
    Reward: +1 for putting socks into target place (episode ends)
            -1 for putting something else
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, seed=None, use_snacks=False, use_trash=False):
        super().__init__()
        
        self.grid_size = 4
        self.render_mode = render_mode
        self._seed = seed
        self.use_snacks = use_snacks
        self.use_trash = use_trash
        
        # Define action and observation spaces
        # 6 actions: left, right, top, down, pick up, put
        self.action_space = spaces.Discrete(len(ACTION))
        
        # Observation: [dog_x, dog_y, socks_x, socks_y, target_x, target_y, has_socks]
        # has_socks: 0 = no socks, 1 = has socks
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(7,), dtype=np.int32
        )
        
        # Load textures for rendering
        self._load_textures()
        
        # Initialize state
        self.reset()
    
    def _load_textures(self):
        """Load texture images from the textures directory (jack_dog_env/textures)."""
        textures_dir = os.path.join(os.path.dirname(__file__), "textures")
        
        # Load cell texture
        cell_path = os.path.join(textures_dir, "cell.png")
        self.cell_texture = Image.open(cell_path).convert("RGBA")
        self.cell_size = self.cell_texture.size[0]  # Assume square cells
        
        # Load dog texture
        dog_path = os.path.join(textures_dir, "dog.png")
        self.dog_texture = Image.open(dog_path).convert("RGBA")
        # Resize dog to fit in cell (assuming 80% of cell size)
        dog_size = int(self.cell_size * 0.8)
        self.dog_texture = self.dog_texture.resize((dog_size, dog_size), Image.Resampling.LANCZOS)
        
        # Load dog with socks texture
        dog_socks_path = os.path.join(textures_dir, "dog_socks.png")
        self.dog_socks_texture = Image.open(dog_socks_path).convert("RGBA")
        self.dog_socks_texture = self.dog_socks_texture.resize((dog_size, dog_size), Image.Resampling.LANCZOS)
        
        # Load socks texture
        socks_path = os.path.join(textures_dir, "socks.png")
        self.socks_texture = Image.open(socks_path).convert("RGBA")
        # Resize socks to fit in cell (assuming 60% of cell size)
        socks_size = int(self.cell_size * 0.6)
        self.socks_texture = self.socks_texture.resize((socks_size, socks_size), Image.Resampling.LANCZOS)
        
        # Load snack texture when use_snacks is enabled
        if getattr(self, "use_snacks", False):
            snack_path = os.path.join(textures_dir, "snack.png")
            self.snack_texture = Image.open(snack_path).convert("RGBA")
            snack_size = int(self.cell_size * 0.6)
            self.snack_texture = self.snack_texture.resize((snack_size, snack_size), Image.Resampling.LANCZOS)
        else:
            self.snack_texture = None
        # Load trash texture when use_trash is enabled
        if getattr(self, "use_trash", False):
            cola_path = os.path.join(textures_dir, "cola_.png")
            self.trash_texture = Image.open(cola_path).convert("RGBA")
            trash_size = int(self.cell_size * 0.6)
            self.trash_texture = self.trash_texture.resize((trash_size, trash_size), Image.Resampling.LANCZOS)
        else:
            self.trash_texture = None
    
    def reset(self, seed=None, options=None):
        if seed is None and self._seed is not None:
            seed = self._seed
        super().reset(seed=seed)
        
        # Randomly place dog, socks, target, and optionally snack and trash (all different positions)
        n_entities = 3 + (1 if self.use_snacks else 0) + (1 if self.use_trash else 0)
        positions = self.np_random.choice(
            self.grid_size * self.grid_size,
            size=n_entities,
            replace=False
        )
        
        dog_pos = positions[0]
        socks_pos = positions[1]
        target_pos = positions[2]
        
        self.dog_x = dog_pos // self.grid_size
        self.dog_y = dog_pos % self.grid_size
        
        self.socks_x = socks_pos // self.grid_size
        self.socks_y = socks_pos % self.grid_size
        
        self.target_x = target_pos // self.grid_size
        self.target_y = target_pos % self.grid_size
        
        self.has_socks = False
        
        idx = 3
        if self.use_snacks:
            snack_pos = positions[idx]
            idx += 1
            self.snack_x = snack_pos // self.grid_size
            self.snack_y = snack_pos % self.grid_size
        else:
            self.snack_x = self.snack_y = -1
        if self.use_trash:
            trash_pos = positions[idx]
            self.trash_x = trash_pos // self.grid_size
            self.trash_y = trash_pos % self.grid_size
        else:
            self.trash_x = self.trash_y = -1

        self._initial_state_id = f"s{self._state_to_idx(self.dog_x, self.dog_y, self.socks_x, self.socks_y, self.has_socks)}"
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def _state_to_idx(self, dog_x, dog_y, socks_x, socks_y, has_socks):
        """Convert (dog_x, dog_y, socks_x, socks_y, has_socks) to state index 0..272."""
        dog_pos = dog_x * self.grid_size + dog_y
        if has_socks:
            return 256 + dog_pos  # s256..s271
        socks_pos = socks_x * self.grid_size + socks_y
        return dog_pos * 16 + socks_pos  # s0..s255
    
    def _idx_to_state(self, idx):
        """Convert state index to (dog_x, dog_y, socks_x, socks_y, has_socks)."""
        if idx == 272:  # terminal success state
            return None, None, None, None, None
        if idx >= 256:
            dog_pos = idx - 256
            dog_x, dog_y = dog_pos // self.grid_size, dog_pos % self.grid_size
            return dog_x, dog_y, dog_x, dog_y, True
        dog_pos = idx // 16
        socks_pos = idx % 16
        dog_x, dog_y = dog_pos // self.grid_size, dog_pos % self.grid_size
        socks_x, socks_y = socks_pos // self.grid_size, socks_pos % self.grid_size
        return dog_x, dog_y, socks_x, socks_y, False
    
    def _compute_move_pos(self, dog_x, dog_y, action_int):
        """Compute next position from move action. Returns (new_x, new_y)."""
        if action_int == ACTION.LEFT:
            return dog_x, max(0, dog_y - 1)
        if action_int == ACTION.RIGHT:
            return dog_x, min(self.grid_size - 1, dog_y + 1)
        if action_int == ACTION.TOP:
            return max(0, dog_x - 1), dog_y
        if action_int == ACTION.DOWN:
            return min(self.grid_size - 1, dog_x + 1), dog_y
        return dog_x, dog_y
    
    def get_all_states(self):
        """Return tuple of all state names ('s0', ..., 's272') for the current layout."""
        return tuple(f"s{i}" for i in range(273))
    
    def get_possible_actions(self, state):
        """Return tuple of possible action names for the state."""
        return ("a0", "a1", "a2", "a3", "a4", "a5")  # LEFT, RIGHT, TOP, DOWN, PICK_UP, PUT
    
    def get_next_states(self, state, action):
        """Return dict mapping next_state -> transition probability for (state, action)."""
        idx = int(state.replace("s", ""))
        if idx == 272:  # terminal success
            return {state: 1.0}
        dog_x, dog_y, socks_x, socks_y, has_socks = self._idx_to_state(idx)
        action_int = int(action.replace("a", ""))
        result = {}
        
        if action_int in (ACTION.LEFT, ACTION.RIGHT, ACTION.TOP, ACTION.DOWN):
            new_dog_x, new_dog_y = self._compute_move_pos(dog_x, dog_y, action_int)
            new_socks_x, new_socks_y = (new_dog_x, new_dog_y) if has_socks else (socks_x, socks_y)
            s_next = f"s{self._state_to_idx(new_dog_x, new_dog_y, new_socks_x, new_socks_y, has_socks)}"
            result[s_next] = 1.0
        elif action_int == ACTION.PICK_UP:
            if dog_x == socks_x and dog_y == socks_y and not has_socks:
                s_next = f"s{self._state_to_idx(dog_x, dog_y, dog_x, dog_y, True)}"
                result[s_next] = 1.0
            else:
                result[state] = 1.0  # no change
        elif action_int == ACTION.PUT:
            if has_socks:
                if dog_x == self.target_x and dog_y == self.target_y:
                    result["s272"] = 1.0  # success terminal
                else:
                    s_next = f"s{self._state_to_idx(dog_x, dog_y, dog_x, dog_y, False)}"
                    result[s_next] = 1.0
            else:
                result[state] = 1.0  # no change
        
        return result
    
    def get_reward(self, state, action, next_state):
        """Return reward for (state, action, next_state)."""
        idx = int(state.replace("s", ""))
        if idx == 272:
            return 0.0
        dog_x, dog_y, socks_x, socks_y, has_socks = self._idx_to_state(idx)
        idx_next = int(next_state.replace("s", ""))
        action_int = int(action.replace("a", ""))
        
        if idx_next == 272:  # success
            return 1.0
        return 0.0
    
    def get_transition_prob(self, state, action, next_state):
        """Return P(next_state | state, action)."""
        next_states = self.get_next_states(state, action)
        return next_states.get(next_state, 0.0)
    
    def is_terminal(self, state):
        """Return True if state is terminal (socks successfully in target)."""
        idx = int(state.replace("s", ""))
        return idx == 272
    
    def get_initial_state(self):
        """Return the state id that the env is in right after reset()."""
        return self._initial_state_id
    
    def load_from_state_id(self, state_id):
        """Load the environment into the given state. Sets dog_x, dog_y, socks_x, socks_y, has_socks."""
        idx = int(state_id.replace("s", ""))
        if idx == 272:
            raise ValueError("Cannot load terminal success state s272")
        dog_x, dog_y, socks_x, socks_y, has_socks = self._idx_to_state(idx)
        self.dog_x = dog_x
        self.dog_y = dog_y
        self.socks_x = socks_x
        self.socks_y = socks_y
        self.has_socks = has_socks
    
    def seed(self, seed=None):
        """Set the seed for the environment's random number generator.
        For compatibility with older Gym API. Returns the seed in a list."""
        self._seed = seed
        if hasattr(self, "np_random") and self.np_random is not None:
            self.np_random = np.random.default_rng(seed)
        return [seed]
    
    def _get_observation(self):
        """Get current observation."""
        return np.array([
            self.dog_x,
            self.dog_y,
            self.socks_x,
            self.socks_y,
            self.target_x,
            self.target_y,
            1 if self.has_socks else 0
        ], dtype=np.int32)
    
    def step(self, action):
        terminated = False
        truncated = False
        reward = 0
        
        if action == ACTION.LEFT:
            self.dog_y = max(0, self.dog_y - 1)
        elif action == ACTION.RIGHT:
            self.dog_y = min(self.grid_size - 1, self.dog_y + 1)
        elif action == ACTION.TOP:
            self.dog_x = max(0, self.dog_x - 1)
        elif action == ACTION.DOWN:
            self.dog_x = min(self.grid_size - 1, self.dog_x + 1)
        elif action == ACTION.PICK_UP:
            # Pick up socks if dog is on the same cell as socks and doesn't have socks
            if (self.dog_x == self.socks_x and
                self.dog_y == self.socks_y and
                not self.has_socks):
                self.has_socks = True
            # Pick up snack (if use_snacks and snack still on floor)
            elif self.use_snacks and self.snack_x >= 0 and self.dog_x == self.snack_x and self.dog_y == self.snack_y:
                reward += 0.1
                self.snack_x = self.snack_y = -1
            # Pick up trash (if use_trash and trash still on floor)
            elif self.use_trash and self.trash_x >= 0 and self.dog_x == self.trash_x and self.dog_y == self.trash_y :
                
                reward -= 0.1
                self.trash_x = self.trash_y = -1
        elif action == ACTION.PUT:
            # Put down socks
            if self.has_socks:
                # Check if putting on target location
                if self.dog_x == self.target_x and self.dog_y == self.target_y:
                    # Successfully put socks in target place
                    reward = 1
                    terminated = True
                else:
                    # # Put socks in wrong place
                    # reward = -1
                    # Place socks at current location
                    self.socks_x = self.dog_x
                    self.socks_y = self.dog_y
                    self.has_socks = False
        
        observation = self._get_observation()
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def render(self, show_coords=False):
        """Render the environment.

        Args:
            show_coords: If True, draw cell coordinates (row, col) in the center of each cell.
        """
        if self.render_mode == "human":
            # Text-based rendering
            grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            
            # Place target
            grid[self.target_x][self.target_y] = 'T'
            
            # Place socks (if not being carried)
            if not self.has_socks:
                grid[self.socks_x][self.socks_y] = 'S'
            
            # Place snack and trash (if enabled and still on floor)
            if self.use_snacks and self.snack_x >= 0:
                grid[self.snack_x][self.snack_y] = 'N'  # snack
            if self.use_trash and self.trash_x >= 0:
                grid[self.trash_x][self.trash_y] = 'R'  # trash
            
            # Place dog
            if grid[self.dog_x][self.dog_y] == '.':
                grid[self.dog_x][self.dog_y] = 'D'
            elif grid[self.dog_x][self.dog_y] == 'S':
                grid[self.dog_x][self.dog_y] = 'DS'
            elif grid[self.dog_x][self.dog_y] == 'T':
                grid[self.dog_x][self.dog_y] = 'DT'
            elif grid[self.dog_x][self.dog_y] == 'N':
                grid[self.dog_x][self.dog_y] = 'DN'
            elif grid[self.dog_x][self.dog_y] == 'R':
                grid[self.dog_x][self.dog_y] = 'DR'
            
            # Print grid
            print("\n" + "=" * 20)
            for row in grid:
                print(" ".join(f"{cell:>3}" for cell in row))
            print(f"Dog has socks: {self.has_socks}")
            print("=" * 20 + "\n")
        
        # Always return image array (for plt.imshow etc.); only skip when no rendering possible
        if self.render_mode in ["human", "rgb_array"] or self.render_mode is None:
            return self._render_image(show_coords=show_coords)

        return None

    def _render_image(self, show_coords=False):
        """Render the environment as an image using textures.

        Args:
            show_coords: If True, draw cell coordinates (row, col) in the center of each cell.
        """
        # Create a blank image for the grid
        img_width = self.grid_size * self.cell_size
        img_height = self.grid_size * self.cell_size
        img = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 255))
        
        # Draw each cell
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_x = y * self.cell_size  # x position in image
                cell_y = x * self.cell_size  # y position in image
                
                # Check if this is the target cell
                if x == self.target_x and y == self.target_y:
                    # Use cell.png and draw a circle on top
                    cell_img = self.cell_texture.copy()
                    # Draw a circle on the cell
                    draw = ImageDraw.Draw(cell_img)
                    # Circle in the center of the cell
                    center = self.cell_size // 2
                    radius = self.cell_size // 3
                    circle_bbox = [
                        center - radius,
                        center - radius,
                        center + radius,
                        center + radius
                    ]
                    draw.ellipse(circle_bbox, outline=(255, 0, 0, 255), width=20)
                    img.paste(cell_img, (cell_x, cell_y), cell_img)
                else:
                    # Regular cell
                    img.paste(self.cell_texture, (cell_x, cell_y), self.cell_texture)
                
                # Place socks if not being carried and this is the socks position
                if not self.has_socks and x == self.socks_x and y == self.socks_y:
                    socks_offset_x = (self.cell_size - self.socks_texture.size[0]) // 2
                    socks_offset_y = (self.cell_size - self.socks_texture.size[1]) // 2
                    img.paste(
                        self.socks_texture,
                        (cell_x + socks_offset_x, cell_y + socks_offset_y),
                        self.socks_texture
                    )
                
                # Place snack (if use_snacks and snack still on floor)
                if self.use_snacks and self.snack_texture is not None and self.snack_x >= 0 and x == self.snack_x and y == self.snack_y:
                    snack_offset_x = (self.cell_size - self.snack_texture.size[0]) // 2
                    snack_offset_y = (self.cell_size - self.snack_texture.size[1]) // 2
                    img.paste(
                        self.snack_texture,
                        (cell_x + snack_offset_x, cell_y + snack_offset_y),
                        self.snack_texture
                    )
                
                # Place trash (if use_trash and trash still on floor)
                if self.use_trash and self.trash_texture is not None and self.trash_x >= 0 and x == self.trash_x and y == self.trash_y:
                    trash_offset_x = (self.cell_size - self.trash_texture.size[0]) // 2
                    trash_offset_y = (self.cell_size - self.trash_texture.size[1]) // 2
                    img.paste(
                        self.trash_texture,
                        (cell_x + trash_offset_x, cell_y + trash_offset_y),
                        self.trash_texture
                    )
                
                # Place dog
                if x == self.dog_x and y == self.dog_y:
                    dog_offset_x = (self.cell_size - self.dog_texture.size[0]) // 2
                    dog_offset_y = (self.cell_size - self.dog_texture.size[1]) // 2

                    # Use dog_socks texture if dog has socks, otherwise use dog texture
                    dog_img = self.dog_socks_texture if self.has_socks else self.dog_texture
                    img.paste(
                        dog_img,
                        (cell_x + dog_offset_x, cell_y + dog_offset_y),
                        dog_img
                    )

        # Draw cell coordinates in center of each cell if requested
        if show_coords:
            font_size = max(12, self.cell_size // 6)
            font = None
            for path in [
                "/System/Library/Fonts/Helvetica.ttc",  # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "C:\\Windows\\Fonts\\arial.ttf",  # Windows
            ]:
                try:
                    font = ImageFont.truetype(path, size=font_size)
                    break
                except (OSError, IOError):
                    continue
            if font is None:
                font = ImageFont.load_default()
            draw = ImageDraw.Draw(img)
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    cell_x = y * self.cell_size
                    cell_y = x * self.cell_size
                    center_x = cell_x + self.cell_size // 2
                    center_y = cell_y + self.cell_size // 2
                    coord_text = f"({x},{y})"
                    # anchor="mm" = middle-center (Pillow 8.0+)
                    draw.text((center_x, center_y), coord_text, fill=(0, 0, 0, 255), font=font, anchor="mm")

        # Always return RGB numpy array (so plt.imshow works); convert RGBA to RGB
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        arr = np.array(rgb_img)

        # For human mode, also display the image
        if self.render_mode == "human":
            img.show()

        return arr
