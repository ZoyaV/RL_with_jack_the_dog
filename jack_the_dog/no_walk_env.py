import gymnasium as gym
from gymnasium import spaces, Wrapper
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from enum import IntEnum
import os


class ACTION(IntEnum):
    """Actions available: move only (dog runs from Zoya to safety)."""
    LEFT = 0
    RIGHT = 1
    TOP = 2
    DOWN = 3


class NoWalkEnv(gym.Env):
    """
    Grid environment where Jack (the dog) must run from Zoya and hide in the safety cell.
    "No walk" — the dog doesn't want to go for a walk and hides instead.

    - Jack: dog (pitty_jack.png), moves with actions.
    - Zoya: static MOB (zoya.png), does not move.
    - Safety cell: one cell with a bed (bed.png); reaching it ends the episode with success.
    - Snack (приманка/bait): reward +0.1 when Jack reaches it. With probability 0.5 the snack
      is a lure — Jack moves one step toward Zoya instead of the intended cell; otherwise he
      moves normally and consumes the snack.

    Grid: 4x4
    Actions: left, right, top, down
    Reward: +1 for reaching the safety cell, +0.1 for consuming snack.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        seed=None,
        more_danger=False,
        dog_start=None,
        zoya_start=None,
        safety_start=None,
        snack_start=None,
    ):
        super().__init__()

        self.grid_size = 4
        self.render_mode = render_mode
        self._seed = seed
        # If True, snack is more dangerous:
        # when Jack steps on the snack, he always eats it and then:
        #   50%  -> teleports directly to Zoya (episode ends, dog is caught)
        #   40%  -> moves one step toward Zoya ("near Zoya")
        #   10%  -> stays in the snack cell
        # If False (default), after eating the snack:
        #   50%  -> moves one step toward Zoya
        #   50%  -> stays in the snack cell
        self.more_danger = more_danger

        # Optional fixed starting positions as (x, y) or flat index 0..15.
        # If None, positions are sampled randomly in reset().
        self._dog_start = dog_start
        self._zoya_start = zoya_start
        self._safety_start = safety_start
        self._snack_start = snack_start

        self.action_space = spaces.Discrete(len(ACTION))

        # Observation: [dog_x, dog_y, zoya_x, zoya_y, safety_x, safety_y, snack_x, snack_y]
        # snack_x, snack_y = grid_size when snack is consumed
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(8,), dtype=np.int32
        )

        self._load_textures()
        self.reset()

    def _load_textures(self):
        """Load texture images from the textures directory."""
        textures_dir = os.path.join(os.path.dirname(__file__), "textures")

        cell_path = os.path.join(textures_dir, "cell.png")
        self.cell_texture = Image.open(cell_path).convert("RGBA")
        self.cell_size = self.cell_texture.size[0]

        # Jack (dog) - pitty_jack.png
        jack_path = os.path.join(textures_dir, "pitty_jack.png")
        self.jack_texture = Image.open(jack_path).convert("RGBA")
        jack_size = int(self.cell_size * 0.8)
        self.jack_texture = self.jack_texture.resize(
            (jack_size, jack_size), Image.Resampling.LANCZOS
        )

        # Zoya (static MOB)
        zoya_path = os.path.join(textures_dir, "zoya.png")
        self.zoya_texture = Image.open(zoya_path).convert("RGBA")
        zoya_size = int(self.cell_size * 0.8)
        self.zoya_texture = self.zoya_texture.resize(
            (zoya_size, zoya_size), Image.Resampling.LANCZOS
        )

        # Safety cell: bed
        bed_path = os.path.join(textures_dir, "bed.png")
        self.bed_texture = Image.open(bed_path).convert("RGBA")
        bed_size = int(self.cell_size * 0.8)
        self.bed_texture = self.bed_texture.resize(
            (bed_size, bed_size), Image.Resampling.LANCZOS
        )

        # Snack (приманка/bait)
        snack_path = os.path.join(textures_dir, "snack.png")
        self.snack_texture = Image.open(snack_path).convert("RGBA")
        snack_size = int(self.cell_size * 0.6)
        self.snack_texture = self.snack_texture.resize(
            (snack_size, snack_size), Image.Resampling.LANCZOS
        )

    def reset(self, seed=None, options=None):
        if seed is None and self._seed is not None:
            seed = self._seed
        super().reset(seed=seed)

        # Helper to convert various coordinate formats into flat index 0..(grid_size^2-1)
        def _to_index(pos, name):
            if pos is None:
                return None
            # Allow (x, y) tuple/list
            if isinstance(pos, (tuple, list)) and len(pos) == 2:
                x, y = int(pos[0]), int(pos[1])
                if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                    raise ValueError(f"{name} coordinates {pos} are out of bounds")
                return x * self.grid_size + y
            # Allow flat integer index
            if isinstance(pos, (int, np.integer)):
                idx = int(pos)
                if not (0 <= idx < self.grid_size * self.grid_size):
                    raise ValueError(f"{name} index {idx} is out of bounds")
                return idx
            raise TypeError(
                f"{name} must be None, (x, y) or int index, got {type(pos)}"
            )

        # Allow overriding positions per-reset via options, if provided.
        dog_opt = None
        zoya_opt = None
        safety_opt = None
        snack_opt = None
        if options is not None:
            dog_opt = options.get("dog_start")
            zoya_opt = options.get("zoya_start")
            safety_opt = options.get("safety_start")
            snack_opt = options.get("snack_start")

        dog_pos = _to_index(dog_opt, "dog_start") if dog_opt is not None else _to_index(
            self._dog_start, "dog_start"
        )
        zoya_pos = _to_index(zoya_opt, "zoya_start") if zoya_opt is not None else _to_index(
            self._zoya_start, "zoya_start"
        )
        safety_pos = _to_index(safety_opt, "safety_start") if safety_opt is not None else _to_index(
            self._safety_start, "safety_start"
        )
        snack_pos = _to_index(snack_opt, "snack_start") if snack_opt is not None else _to_index(
            self._snack_start, "snack_start"
        )

        # If any of them are still None, sample the missing ones randomly without replacement.
        all_indices = list(range(self.grid_size * self.grid_size))
        fixed = [p for p in [dog_pos, zoya_pos, safety_pos, snack_pos] if p is not None]
        # Remove already chosen fixed indices so random ones are different
        for idx in fixed:
            if idx in all_indices:
                all_indices.remove(idx)

        missing = []
        if dog_pos is None:
            missing.append("dog")
        if zoya_pos is None:
            missing.append("zoya")
        if safety_pos is None:
            missing.append("safety")
        if snack_pos is None:
            missing.append("snack")

        if missing:
            sampled = self.np_random.choice(len(all_indices), size=len(missing), replace=False)
            sampled_indices = [all_indices[i] for i in sampled]
            for name, idx in zip(missing, sampled_indices):
                if name == "dog":
                    dog_pos = idx
                elif name == "zoya":
                    zoya_pos = idx
                elif name == "safety":
                    safety_pos = idx
                elif name == "snack":
                    snack_pos = idx

        self.dog_x = dog_pos // self.grid_size
        self.dog_y = dog_pos % self.grid_size

        self.zoya_x = zoya_pos // self.grid_size
        self.zoya_y = zoya_pos % self.grid_size

        self.safety_x = safety_pos // self.grid_size
        self.safety_y = safety_pos % self.grid_size

        self.snack_x = snack_pos // self.grid_size
        self.snack_y = snack_pos % self.grid_size
        self._layout_snack_x = self.snack_x
        self._layout_snack_y = self.snack_y
        self._initial_state_id = f"s{self._state_to_idx(self.dog_x, self.dog_y, 0)}"

        observation = self._get_observation()
        info = {}
        return observation, info

    def seed(self, seed=None):
        """Set the seed for the environment's random number generator."""
        self._seed = seed
        if hasattr(self, "np_random") and self.np_random is not None:
            self.np_random = np.random.default_rng(seed)
        return [seed]

    def _get_observation(self):
        """Get current observation."""
        snack_x = self.snack_x if self.snack_x >= 0 else self.grid_size
        snack_y = self.snack_y if self.snack_y >= 0 else self.grid_size
        return np.array(
            [
                self.dog_x,
                self.dog_y,
                self.zoya_x,
                self.zoya_y,
                self.safety_x,
                self.safety_y,
                snack_x,
                snack_y,
            ],
            dtype=np.int32,
        )

    def _state_to_idx(self, dog_x, dog_y, snack_consumed):
        """Convert (dog_x, dog_y, snack_consumed) to state index 0..31."""
        return (dog_x * self.grid_size + dog_y) * 2 + (1 if snack_consumed else 0)

    def _idx_to_state(self, idx):
        """Convert state index to (dog_x, dog_y, snack_consumed)."""
        v = idx // 2
        dog_x, dog_y = v // self.grid_size, v % self.grid_size
        snack_consumed = idx % 2
        return dog_x, dog_y, snack_consumed

    def _compute_move_toward_zoya(self, dog_x, dog_y):
        """Compute one step toward Zoya from (dog_x, dog_y). Returns (new_x, new_y)."""
        dx = np.sign(self.zoya_x - dog_x)
        dy = np.sign(self.zoya_y - dog_y)
        new_x, new_y = dog_x, dog_y
        if abs(self.zoya_x - dog_x) >= abs(self.zoya_y - dog_y) and dx != 0:
            new_x = np.clip(dog_x + dx, 0, self.grid_size - 1)
        elif dy != 0:
            new_y = np.clip(dog_y + dy, 0, self.grid_size - 1)
        elif dx != 0:
            new_x = np.clip(dog_x + dx, 0, self.grid_size - 1)
        return new_x, new_y

    def _compute_intended_pos(self, dog_x, dog_y, action_int):
        """Compute intended next position from action. Returns (new_x, new_y)."""
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
        """Return tuple of all state names ('s0', 's1', ...) for the current layout."""
        return tuple(f"s{i}" for i in range(32))

    def get_possible_actions(self, state):
        """Return tuple of possible action names for the state."""
        return ("a0", "a1", "a2", "a3")  # LEFT, RIGHT, TOP, DOWN

    def get_next_states(self, state, action):
        """Return dict mapping next_state -> transition probability for (state, action)."""
        idx = int(state[1:])  # 's5' -> 5
        dog_x, dog_y, snack_consumed = self._idx_to_state(idx)
        action_int = int(action[1:])  # 'a2' -> 2

        # Terminal state: dog on safety
        if (dog_x, dog_y) == (self.safety_x, self.safety_y):
            return {state: 1.0}

        intended_x, intended_y = self._compute_intended_pos(dog_x, dog_y, action_int)
        result = {}

        if (not snack_consumed
            and intended_x == self.snack_x
            and intended_y == self.snack_y):
            # Jack steps onto the snack: he always eats it (snack_consumed=1),
            # then his final position after eating depends on `more_danger`.
            #
            # Baseline (more_danger=False):
            #   50% -> one step toward Zoya ("near Zoya")
            #   50% -> stay in the snack cell
            #
            # Dangerous mode (more_danger=True):
            #   50% -> teleport directly to Zoya (dog caught)
            #   40% -> one step toward Zoya
            #   10% -> stay in the snack cell

            # Starting position after eating: snack cell
            snack_cell_x, snack_cell_y = self.snack_x, self.snack_y

            # "Stay in snack cell"
            s_stay = f"s{self._state_to_idx(snack_cell_x, snack_cell_y, 1)}"

            # "Near Zoya": one step from snack cell toward Zoya
            near_x, near_y = self._compute_move_toward_zoya(snack_cell_x, snack_cell_y)
            s_near = f"s{self._state_to_idx(near_x, near_y, 1)}"

            if self.more_danger:
                # "Directly to Zoya"
                s_zoya = f"s{self._state_to_idx(self.zoya_x, self.zoya_y, 1)}"
                result[s_zoya] = result.get(s_zoya, 0.0) + 0.5
                result[s_near] = result.get(s_near, 0.0) + 0.4
                result[s_stay] = result.get(s_stay, 0.0) + 0.1
            else:
                result[s_near] = result.get(s_near, 0.0) + 0.5
                result[s_stay] = result.get(s_stay, 0.0) + 0.5
        else:
            # Deterministic move (no snack at intended, or snack already consumed)
            s_next = f"s{self._state_to_idx(intended_x, intended_y, snack_consumed)}"
            result[s_next] = 1.0

        return result

    def get_reward(self, state, action, next_state):
        """Return reward for (state, action, next_state)."""
        idx_next = int(next_state[1:])
        dog_x, dog_y, snack_consumed_next = self._idx_to_state(idx_next)
        idx = int(state[1:])
        _, _, snack_consumed = self._idx_to_state(idx)

        reward = 0.0
        if (dog_x, dog_y) == (self.safety_x, self.safety_y):
            reward += 1.0
        if (dog_x, dog_y) == (self.zoya_x, self.zoya_y):
            reward -= 1.0
        if snack_consumed_next and not snack_consumed:
            reward += 0.1
        return reward

    def get_transition_prob(self, state, action, next_state):
        """Return P(next_state | state, action)."""
        next_states = self.get_next_states(state, action)
        return next_states.get(next_state, 0.0)

    def is_terminal(self, state):
        """Return True if state is terminal: dog caught by Zoya or dog reached safe place (under the bed)."""
        idx = int(state.replace("s", ""))
        dog_x, dog_y, _ = self._idx_to_state(idx)
        return (dog_x, dog_y) == (self.zoya_x, self.zoya_y) or (dog_x, dog_y) == (self.safety_x, self.safety_y)
    
    @property
    def _initial_state(self):
        """Return the state id that the env is in right after reset()."""
        return self._initial_state_id

    def load_from_state_id(self, state_id):
        """Load the environment into the given state. Sets dog_x, dog_y, snack_x, snack_y."""
        idx = int(state_id.replace("s", ""))
        dog_x, dog_y, snack_consumed = self._idx_to_state(idx)
        self.dog_x = dog_x
        self.dog_y = dog_y
        if snack_consumed:
            self.snack_x = -1
            self.snack_y = -1
        else:
            self.snack_x = self._layout_snack_x
            self.snack_y = self._layout_snack_y

    def _move_toward_zoya(self):
        """Move Jack one step toward Zoya (for lure effect)."""
        dx = np.sign(self.zoya_x - self.dog_x)
        dy = np.sign(self.zoya_y - self.dog_y)
        # Prefer the axis with larger distance; if tied, prefer x
        if abs(self.zoya_x - self.dog_x) >= abs(self.zoya_y - self.dog_y) and dx != 0:
            self.dog_x = np.clip(self.dog_x + dx, 0, self.grid_size - 1)
        elif dy != 0:
            self.dog_y = np.clip(self.dog_y + dy, 0, self.grid_size - 1)
        elif dx != 0:
            self.dog_x = np.clip(self.dog_x + dx, 0, self.grid_size - 1)

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0

        # Compute intended position from action
        new_x, new_y = self.dog_x, self.dog_y
        if action == ACTION.LEFT:
            new_y = max(0, self.dog_y - 1)
        elif action == ACTION.RIGHT:
            new_y = min(self.grid_size - 1, self.dog_y + 1)
        elif action == ACTION.TOP:
            new_x = max(0, self.dog_x - 1)
        elif action == ACTION.DOWN:
            new_x = min(self.grid_size - 1, self.dog_x + 1)

        # Snack (приманка): if intended cell is snack and snack exists
        if self.snack_x >= 0 and new_x == self.snack_x and new_y == self.snack_y:
            # Jack always reaches the snack cell, eats the snack, and gets +0.1 reward.
            self.dog_x, self.dog_y = new_x, new_y
            reward += 0.1
            self.snack_x, self.snack_y = -1, -1

            r = self.np_random.random()
            if self.more_danger:
                # 50% -> teleport directly to Zoya (dog caught)
                # 40% -> move one step toward Zoya
                # 10% -> stay in the snack cell
                if r < 0.5:
                    self.dog_x, self.dog_y = self.zoya_x, self.zoya_y
                elif r < 0.9:
                    self._move_toward_zoya()
                else:
                    # stay on the snack cell (already there)
                    pass
            else:
                # Baseline behaviour:
                # 50% -> move one step toward Zoya
                # 50% -> stay in the snack cell
                if r < 0.5:
                    self._move_toward_zoya()
                else:
                    # stay on the snack cell (already there)
                    pass
        else:
            self.dog_x, self.dog_y = new_x, new_y

        # Success: Jack reached the safety cell (hid from Zoya)
        if self.dog_x == self.safety_x and self.dog_y == self.safety_y:
            reward = 1.0
            terminated = True
        
            

        # Failure: Jack is caught by Zoya
        if self.dog_x == self.zoya_x and self.dog_y == self.zoya_y:
            reward = -1.0
            terminated = True

        observation = self._get_observation()
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self, show_coords=False):
        """Render the environment."""
        if self.render_mode == "human":
            grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            grid[self.safety_x][self.safety_y] = "S"  # safety
            grid[self.zoya_x][self.zoya_y] = "Z"      # Zoya
            if self.snack_x >= 0:
                grid[self.snack_x][self.snack_y] = "N"  # snack
            if grid[self.dog_x][self.dog_y] == ".":
                grid[self.dog_x][self.dog_y] = "J"    # Jack
            else:
                grid[self.dog_x][self.dog_y] = "J"
            print("\n" + "=" * 20)
            for row in grid:
                print(" ".join(f"{cell:>3}" for cell in row))
            print("=" * 20 + "\n")

        if self.render_mode in ["human", "rgb_array"] or self.render_mode is None:
            return self._render_image(show_coords=show_coords)
        return None

    def _render_image(self, show_coords=False):
        """Render the environment as an image using textures."""
        img_width = self.grid_size * self.cell_size
        img_height = self.grid_size * self.cell_size
        img = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 255))

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_x = y * self.cell_size
                cell_y = x * self.cell_size

                # Safety cell: bed
                if x == self.safety_x and y == self.safety_y:
                    img.paste(self.cell_texture, (cell_x, cell_y), self.cell_texture)
                    bed_offset_x = (self.cell_size - self.bed_texture.size[0]) // 2
                    bed_offset_y = (self.cell_size - self.bed_texture.size[1]) // 2
                    img.paste(
                        self.bed_texture,
                        (cell_x + bed_offset_x, cell_y + bed_offset_y),
                        self.bed_texture,
                    )
                else:
                    img.paste(self.cell_texture, (cell_x, cell_y), self.cell_texture)

                # Zoya (static)
                if x == self.zoya_x and y == self.zoya_y:
                    z_offset_x = (self.cell_size - self.zoya_texture.size[0]) // 2
                    z_offset_y = (self.cell_size - self.zoya_texture.size[1]) // 2
                    img.paste(
                        self.zoya_texture,
                        (cell_x + z_offset_x, cell_y + z_offset_y),
                        self.zoya_texture,
                    )

                # Snack (if not consumed)
                if self.snack_x >= 0 and x == self.snack_x and y == self.snack_y:
                    snack_offset_x = (self.cell_size - self.snack_texture.size[0]) // 2
                    snack_offset_y = (self.cell_size - self.snack_texture.size[1]) // 2
                    img.paste(
                        self.snack_texture,
                        (cell_x + snack_offset_x, cell_y + snack_offset_y),
                        self.snack_texture,
                    )

                # Jack (dog)
                if x == self.dog_x and y == self.dog_y:
                    j_offset_x = (self.cell_size - self.jack_texture.size[0]) // 2
                    j_offset_y = (self.cell_size - self.jack_texture.size[1]) // 2
                    img.paste(
                        self.jack_texture,
                        (cell_x + j_offset_x, cell_y + j_offset_y),
                        self.jack_texture,
                    )

        if show_coords:
            font_size = max(12, self.cell_size // 6)
            font = None
            for path in [
                "/System/Library/Fonts/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:\\Windows\\Fonts\\arial.ttf",
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
                    draw.text(
                        (center_x, center_y),
                        coord_text,
                        fill=(0, 0, 0, 255),
                        font=font,
                        anchor="mm",
                    )

        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        arr = np.array(rgb_img)

        if self.render_mode == "human":
            img.show()

        return arr


class StateValueOverlayWrapper(Wrapper):
    """
    Wrapper that renders state values as a semi-transparent red-green overlay on top
    of the environment. Higher values appear greener, lower values appear redder.

    State values format: {'s0': 0.81, 's1': 0.81, ..., 's31': 1.0}
    Each cell gets the max value of its two states (snack consumed / not consumed).
    """

    def __init__(self, env, state_values=None, alpha=0.45):
        super().__init__(env)
        self._state_values = state_values or {}
        self._alpha = alpha  # overlay transparency (0=invisible, 1=opaque)

    def set_state_values(self, state_values):
        """Update the state values dict. Format: {'s0': float, 's1': float, ...}"""
        self._state_values = dict(state_values)

    def render(self, show_coords=False):
        """Render with state value overlay (red=low, green=high)."""
        unwrapped = self.env.unwrapped
        saved_mode = getattr(unwrapped, "render_mode", None)
        unwrapped.render_mode = "rgb_array"
        try:
            base_arr = self.env.render(show_coords=show_coords)
        finally:
            unwrapped.render_mode = saved_mode

        if not self._state_values:
            if saved_mode == "human":
                Image.fromarray(base_arr).show()
            return base_arr

        grid_size = getattr(unwrapped, "grid_size", 4)
        cell_size = getattr(unwrapped, "cell_size", 64)
        n_cells = grid_size * grid_size

        # Compute value per cell: max of the two states (snack 0/1) for that cell
        cell_values = []
        for cell_idx in range(n_cells):
             s0 = self._state_values.get(f"s{cell_idx * 2}", 0.0)
             s1 = self._state_values.get(f"s{cell_idx * 2 + 1}", 0.0)
             cell_values.append(max(s0, s1))

        vals = np.array(cell_values, dtype=float)
        v_min, v_max = vals.min(), vals.max()
        if v_max - v_min > 1e-9:
            normalized = (vals - v_min) / (v_max - v_min)
        else:
            normalized = np.ones_like(vals) * 0.5

        # Create RGBA overlay: red (0) -> green (1)
        overlay = np.zeros(
            (grid_size * cell_size, grid_size * cell_size, 4), dtype=np.uint8
        )
        for x in range(grid_size):
            for y in range(grid_size):
                cell_idx = x * grid_size + y
                t = float(normalized[cell_idx])
                # Red to green: (1-t)*red + t*green
                r = int(255 * (1 - t))
                g = int(255 * t)
                cell_x = y * cell_size
                cell_y = x * cell_size
                overlay[
                    cell_y : cell_y + cell_size,
                    cell_x : cell_x + cell_size,
                    0,
                ] = r
                overlay[
                    cell_y : cell_y + cell_size,
                    cell_x : cell_x + cell_size,
                    1,
                ] = g
                overlay[
                    cell_y : cell_y + cell_size,
                    cell_x : cell_x + cell_size,
                    2,
                ] = 0
                overlay[
                    cell_y : cell_y + cell_size,
                    cell_x : cell_x + cell_size,
                    3,
                ] = int(255 * self._alpha)

        base_pil = Image.fromarray(base_arr).convert("RGBA")
        overlay_pil = Image.fromarray(overlay, mode="RGBA")
        composed = Image.alpha_composite(base_pil, overlay_pil)
        result = np.array(composed.convert("RGB"))

        if saved_mode == "human":
            Image.fromarray(result).show()

        return result


def _get_action_value(mdp, state_values, state, action, gamma):
    """Compute Q(s,a) = sum_s' P(s'|s,a) * (r + gamma * V(s'))."""
    t = [
        prob * (mdp.get_reward(state, action, s) + gamma * state_values.get(s, 0.0))
        for s, prob in mdp.get_next_states(state, action).items()
    ]
    return sum(t)


def _get_optimal_action(mdp, state_values, state, gamma):
    """Return best action for state, or None if terminal."""
    if mdp.is_terminal(state):
        return None
    actions = mdp.get_possible_actions(state)
    q_values = [_get_action_value(mdp, state_values, state, a, gamma) for a in actions]
    return actions[np.argmax(q_values)]


def _draw_arrow(draw, cx, cy, size, direction, fill=(0, 0, 0, 220)):
    """Draw an arrow (triangle) at (cx, cy) pointing in direction. direction: 'a0'=LEFT, 'a1'=RIGHT, 'a2'=TOP, 'a3'=DOWN."""
    # Arrow as triangle: base opposite to tip
    h = size * 0.6
    w = size * 0.4
    if direction == "a0":  # LEFT
        points = [(cx - h, cy), (cx + h, cy - w), (cx + h, cy + w)]
    elif direction == "a1":  # RIGHT
        points = [(cx + h, cy), (cx - h, cy - w), (cx - h, cy + w)]
    elif direction == "a2":  # TOP
        points = [(cx, cy - h), (cx - w, cy + h), (cx + w, cy + h)]
    else:  # a3 DOWN
        points = [(cx, cy + h), (cx - w, cy - h), (cx + w, cy - h)]
    draw.polygon(points, fill=fill, outline=(255, 255, 255, 255))


class PolicyOverlayWrapper(Wrapper):
    """
    Wrapper that renders the optimal policy as arrows in each cell.
    Strategy is computed from state_values using Q(s,a) = sum P(s'|s,a)*(r + gamma*V(s')).
    Each cell shows an arrow indicating the best action (LEFT/RIGHT/TOP/DOWN).
    """

    def __init__(self, env, state_values=None, gamma=0.9):
        super().__init__(env)
        self._state_values = state_values or {}
        self._gamma = gamma

    def set_state_values(self, state_values):
        """Update the state values dict. Format: {'s0': float, 's1': float, ...}"""
        self._state_values = dict(state_values)

    def render(self, show_coords=False):
        """Render with policy arrows overlay."""
        unwrapped = self.env.unwrapped
        saved_mode = getattr(unwrapped, "render_mode", None)
        unwrapped.render_mode = "rgb_array"
        try:
            base_arr = self.env.render(show_coords=show_coords)
        finally:
            unwrapped.render_mode = saved_mode

        mdp = unwrapped
        grid_size = getattr(unwrapped, "grid_size", 4)
        cell_size = getattr(unwrapped, "cell_size", 64)

        # Compute optimal action for each cell (use snack-not-consumed state)
        cell_actions = []
        for cell_idx in range(grid_size * grid_size):
            state = f"s{cell_idx * 2}"  # snack not consumed
            action = _get_optimal_action(mdp, self._state_values, state, self._gamma)
            cell_actions.append(action)

        base_pil = Image.fromarray(base_arr).convert("RGBA")
        draw = ImageDraw.Draw(base_pil)

        arrow_size = cell_size * 0.2
        for x in range(grid_size):
            for y in range(grid_size):
                cell_idx = x * grid_size + y
                action = cell_actions[cell_idx]
                if action is None:
                    continue
                cell_x = y * cell_size
                cell_y = x * cell_size
                cx = cell_x + cell_size // 2
                cy = cell_y + cell_size // 2
                _draw_arrow(draw, cx, cy, arrow_size, action)

        result = np.array(base_pil.convert("RGB"))

        if saved_mode == "human":
            Image.fromarray(result).show()

        return result

