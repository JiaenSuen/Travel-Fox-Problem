from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class FoxRenderer:
    """Presentation renderer with an optional task-neutral robot-state drawer."""

    STATUS_BUTTON = (856, 430, 1092, 468)

    def __init__(self, canvas_size: int = 832, panel_width: int = 288) -> None:
        self.canvas_size = canvas_size
        self.panel_width = panel_width
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 14)
            self.font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
            self.font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 19)
        except OSError:
            self.font = self.font_small = self.font_title = ImageFont.load_default()

    def render(self, env, action_name: str = "RESET", reward: float = 0.0, status_open: bool = False) -> np.ndarray:
        grid_h, grid_w = env.grid.shape
        total_w = self.canvas_size + self.panel_width
        image = Image.new("RGB", (total_w, self.canvas_size), (244, 247, 251))
        draw = ImageDraw.Draw(image)

        margin = 34
        available = self.canvas_size - margin * 2
        tile = max(8, min(available // grid_h, available // grid_w))
        map_w = tile * grid_w
        map_h = tile * grid_h
        ox = (self.canvas_size - map_w) // 2
        oy = (self.canvas_size - map_h) // 2

        draw.rounded_rectangle((ox - 10, oy - 10, ox + map_w + 10, oy + map_h + 10), radius=18, fill=(219, 225, 233))
        draw.rectangle((ox, oy, ox + map_w, oy + map_h), fill=(236, 239, 243))

        for r in range(grid_h):
            for c in range(grid_w):
                x0, y0 = ox + c * tile, oy + r * tile
                x1, y1 = x0 + tile - 1, y0 + tile - 1
                if env.grid[r, c] == 1:
                    draw.rounded_rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), radius=max(2, tile // 8), fill=(61, 70, 84))
                    draw.line((x0 + 4, y0 + 4, x1 - 4, y0 + 4), fill=(95, 106, 123), width=max(1, tile // 12))
                else:
                    draw.rectangle((x0, y0, x1, y1), fill=(249, 250, 252))
                    draw.line((x0, y1, x1, y1), fill=(230, 233, 238), width=1)
                    draw.line((x1, y0, x1, y1), fill=(230, 233, 238), width=1)

        # Optional motion track for dynamic cargo tasks.
        if hasattr(env, "render_track"):
            for tr, tc in env.render_track():
                x0, y0 = ox + tc * tile, oy + tr * tile
                x1, y1 = x0 + tile - 1, y0 + tile - 1
                mid_y = (y0 + y1) // 2
                draw.line((x0 + 2, mid_y, x1 - 2, mid_y), fill=(116, 126, 139), width=max(2, tile // 10))
                if tile >= 18:
                    draw.line((x0 + 3, y0 + 4, x0 + 3, y1 - 4), fill=(172, 180, 190), width=1)
                    draw.line((x1 - 3, y0 + 4, x1 - 3, y1 - 4), fill=(172, 180, 190), width=1)

        # Optional stateful doors for multi-room tasks. Closed doors are drawn as
        # solid panels; open doors keep a visible frame while exposing the floor.
        if hasattr(env, "render_doors"):
            for (dr, dc), is_open in env.render_doors():
                x0, y0 = ox + dc * tile, oy + dr * tile
                x1, y1 = x0 + tile - 1, y0 + tile - 1
                pad = max(2, tile // 7)
                if is_open:
                    draw.rectangle((x0 + pad, y0 + 2, x1 - pad, y1 - 2), outline=(139, 99, 65), width=max(2, tile // 10))
                    draw.line((x0 + pad + 1, y0 + 3, x0 + pad + 1, y1 - 3), fill=(180, 139, 94), width=max(1, tile // 14))
                else:
                    draw.rounded_rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), radius=max(2, tile // 10), fill=(151, 105, 69), outline=(94, 61, 42), width=max(1, tile // 12))
                    knob = max(1, tile // 16)
                    draw.ellipse((x1 - pad - knob, (y0 + y1)//2 - knob, x1 - pad + knob, (y0 + y1)//2 + knob), fill=(232, 201, 126))

        # Optional colored access doors. Drawn after ordinary doors so the access
        # constraint remains visually salient even when a task shares the same doorway cells.
        if hasattr(env, "render_colored_doors"):
            access_palette = [(206, 75, 75), (72, 161, 104), (66, 117, 201)]
            for (dr, dc), is_open, color, is_locked in env.render_colored_doors():
                x0, y0 = ox + dc * tile, oy + dr * tile
                x1, y1 = x0 + tile - 1, y0 + tile - 1
                pad = max(2, tile // 7)
                base = access_palette[int(color) % len(access_palette)]
                if is_open:
                    draw.rectangle((x0 + pad, y0 + 2, x1 - pad, y1 - 2), outline=base, width=max(2, tile // 9))
                else:
                    fill = tuple(min(255, int(v + (255 - v) * (0.18 if is_locked else 0.42))) for v in base)
                    draw.rounded_rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), radius=max(2, tile // 10), fill=fill, outline=base, width=max(2, tile // 11))
                    if is_locked:
                        lock_w = max(4, tile // 4)
                        cx, cy = (x0 + x1)//2, (y0 + y1)//2
                        draw.rectangle((cx-lock_w//2, cy, cx+lock_w//2, cy+lock_w//2), fill=(247, 244, 229), outline=(83, 84, 87))
                        draw.arc((cx-lock_w//2, cy-lock_w//2, cx+lock_w//2, cy+lock_w//3), 180, 360, fill=(83,84,87), width=max(1,tile//16))

        entities = env.render_entities() if hasattr(env, "render_entities") else {"objects": [], "goals": []}
        color_palette = [
            (220, 82, 82),   # red
            (83, 166, 112),  # green
            (72, 128, 201),  # blue
            (224, 179, 63),  # yellow
            (151, 101, 190), # purple
        ]
        for color, (gr, gc) in entities.get("goals", []):
            gx0, gy0 = ox + gc * tile, oy + gr * tile
            pad = max(2, tile // 7)
            if env.TASK_CODE in {"LOCAL-TRANSPORT", "ROOM-DOOR-TRANSPORT", "MOVING-CARGO-EVASION", "KEYED-HAZARD-LOGISTICS"}:
                fill, outline = (104, 187, 139), (59, 139, 98)
            else:
                base = color_palette[int(color) % len(color_palette)]
                fill = tuple(min(255, int(v + (255 - v) * 0.55)) for v in base)
                outline = base
            draw.rounded_rectangle(
                (gx0 + pad, gy0 + pad, gx0 + tile - pad, gy0 + tile - pad),
                radius=max(3, tile // 5), fill=fill, outline=outline, width=max(1, tile // 10),
            )

        if hasattr(env, "render_keys"):
            for color, (kr, kc) in env.render_keys():
                x0, y0 = ox + kc * tile, oy + kr * tile
                base = color_palette[int(color) % len(color_palette)]
                cx, cy = x0 + tile // 2, y0 + tile // 2
                rad = max(3, tile // 7)
                draw.ellipse((cx-rad, cy-rad, cx+rad, cy+rad), outline=base, width=max(2, tile//10))
                draw.line((cx+rad, cy, x0+tile-max(3,tile//8), cy), fill=base, width=max(2,tile//10))
                tooth = max(2, tile//9)
                draw.line((x0+tile-max(3,tile//8)-tooth, cy, x0+tile-max(3,tile//8)-tooth, cy+tooth), fill=base, width=max(2,tile//12))

        for color, (rr, rc) in entities.get("objects", []):
            x0, y0 = ox + rc * tile, oy + rr * tile
            p = max(3, tile // 5)
            if env.TASK_CODE in {"LOCAL-TRANSPORT", "ROOM-DOOR-TRANSPORT", "MOVING-CARGO-EVASION", "KEYED-HAZARD-LOGISTICS"}:
                fill, outline = (244, 178, 75), (181, 116, 36)
            else:
                fill = color_palette[int(color) % len(color_palette)]
                outline = tuple(max(0, int(v * 0.65)) for v in fill)
            draw.rounded_rectangle(
                (x0 + p, y0 + p, x0 + tile - p, y0 + tile - p),
                radius=max(2, tile // 8), fill=fill, outline=outline, width=max(1, tile // 12),
            )

        if hasattr(env, "render_vehicle") and not getattr(env, "carrying", False):
            vr, vc = env.render_vehicle()
            x0, y0 = ox + vc * tile, oy + vr * tile
            pad = max(2, tile // 8)
            draw.rounded_rectangle((x0 + pad, y0 + tile//2, x0 + tile - pad, y0 + tile - pad), radius=max(2, tile//10), outline=(72, 82, 96), width=max(2, tile//12))
            wheel = max(2, tile // 10)
            draw.ellipse((x0 + pad, y0 + tile - pad - wheel, x0 + pad + 2*wheel, y0 + tile - pad + wheel), fill=(55, 62, 72))
            draw.ellipse((x0 + tile - pad - 2*wheel, y0 + tile - pad - wheel, x0 + tile - pad, y0 + tile - pad + wheel), fill=(55, 62, 72))

        if hasattr(env, "render_hazards"):
            for kind, (hr, hc) in env.render_hazards():
                x0, y0 = ox + hc * tile, oy + hr * tile
                cx, cy = x0 + tile // 2, y0 + tile // 2
                rad = max(4, int(tile * 0.28))
                if kind == "wolf":
                    draw.polygon([(cx-rad, cy-rad//2), (cx-rad//2, cy-rad-3), (cx, cy-rad//2)], fill=(246, 247, 249), outline=(112, 120, 132))
                    draw.polygon([(cx+rad, cy-rad//2), (cx+rad//2, cy-rad-3), (cx, cy-rad//2)], fill=(246, 247, 249), outline=(112, 120, 132))
                    draw.ellipse((cx-rad, cy-rad, cx+rad, cy+rad), fill=(246, 247, 249), outline=(112, 120, 132), width=max(1, tile//14))
                    eye = max(1, tile // 22)
                    draw.ellipse((cx-rad//3-eye, cy-eye, cx-rad//3+eye, cy+eye), fill=(35, 39, 46))
                    draw.ellipse((cx+rad//3-eye, cy-eye, cx+rad//3+eye, cy+eye), fill=(35, 39, 46))

        ar, ac = env.agent_pos
        x0, y0 = ox + ac * tile, oy + ar * tile
        cx, cy = x0 + tile // 2, y0 + tile // 2
        radius = max(4, int(tile * 0.30))
        orange = (229, 112, 55) if not env.carrying else (207, 91, 42)
        ear = max(3, int(tile * 0.19))
        draw.polygon([(cx - radius, cy - radius // 2), (cx - radius // 2, cy - radius - ear), (cx, cy - radius // 2)], fill=orange)
        draw.polygon([(cx + radius, cy - radius // 2), (cx + radius // 2, cy - radius - ear), (cx, cy - radius // 2)], fill=orange)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=orange, outline=(147, 63, 31), width=max(1, tile // 14))
        eye_r = max(1, tile // 20)
        draw.ellipse((cx - radius // 3 - eye_r, cy - eye_r, cx - radius // 3 + eye_r, cy + eye_r), fill=(34, 37, 43))
        draw.ellipse((cx + radius // 3 - eye_r, cy - eye_r, cx + radius // 3 + eye_r, cy + eye_r), fill=(34, 37, 43))
        if env.carrying:
            box_r = max(3, tile // 6)
            if hasattr(env, "carrying_color") and env.carrying_color is not None:
                carried_fill = color_palette[int(env.carrying_color) % len(color_palette)]
                carried_outline = tuple(max(0, int(v * 0.65)) for v in carried_fill)
            else:
                carried_fill, carried_outline = (244, 178, 75), (181, 116, 36)
            draw.rounded_rectangle((cx - box_r, cy + radius // 3, cx + box_r, cy + radius), radius=2, fill=carried_fill, outline=carried_outline)

        if env.observation_mode == "local":
            view_radius = env.view_size // 2
            vr0, vc0 = max(0, ar - view_radius), max(0, ac - view_radius)
            vr1, vc1 = min(grid_h - 1, ar + view_radius), min(grid_w - 1, ac + view_radius)
            draw.rectangle(
                (ox + vc0 * tile + 2, oy + vr0 * tile + 2, ox + (vc1 + 1) * tile - 3, oy + (vr1 + 1) * tile - 3),
                outline=(75, 132, 198), width=max(2, tile // 14),
            )

        panel_x = self.canvas_size + 24
        draw.text((panel_x, 30), "TFP", fill=(31, 38, 48), font=self.font_title)
        draw.text((panel_x, 60), env.TASK_NAME, fill=(80, 88, 101), font=self.font)
        draw.text((panel_x, 84), env.TASK_CODE, fill=(111, 120, 134), font=self.font_small)
        lines = [
            f"Map  {env.current_map.name if env.current_map else '-'}",
            f"Size  {grid_h} x {grid_w}",
            f"View  {env.observation_mode} {env.view_size}x{env.view_size}",
            f"Action  {action_name}",
            f"Step  {env.steps} / {env.max_steps}",
            f"Reward  {reward:+.3f}",
            f"Carrying  {'YES' if env.carrying else 'NO'}",
            f"Collisions  {env.collisions}",
            f"Invalid  {env.invalid_actions}",
        ]
        if hasattr(env, "door_positions"):
            lines.append(f"Doors  {len(getattr(env, 'open_doors', ()))}/{len(env.door_positions)} open")
        if hasattr(env, "keys_owned"):
            lines.append(f"Keys  {len(getattr(env, 'keys_owned', ()) )}/{len(getattr(env, 'lock_order_colors', ()))}")
        if hasattr(env, "delivered_count") and hasattr(env, "total_cargo"):
            lines.append(f"Delivered  {env.delivered_count}/{env.total_cargo}")
        if hasattr(env, "wolf_positions"):
            lines.append(f"Hazards  {len(env.wolf_positions)}")
        lines.extend([
            f"Oracle  {env.oracle_steps}",
            f"Reward fn  {env.reward_module}",
        ])
        y = 126
        for line in lines:
            draw.text((panel_x, y), line, fill=(54, 61, 72), font=self.font_small)
            y += 25

        bx0, by0, bx1, by1 = self.STATUS_BUTTON
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=9, fill=(50, 93, 151) if status_open else (226, 233, 243))
        draw.text((bx0 + 14, by0 + 10), f"ROBOT STATE {'▲' if status_open else '▼'}", fill=(255, 255, 255) if status_open else (49, 76, 112), font=self.font_small)
        draw.text((panel_x, self.canvas_size - 56), "Click ROBOT STATE to inspect task state", fill=(90, 99, 112), font=self.font_small)

        if status_open:
            self._draw_status_drawer(draw, env, total_w)
        return np.asarray(image, dtype=np.uint8)

    def _draw_status_drawer(self, draw: ImageDraw.ImageDraw, env, total_w: int) -> None:
        x0, y0, x1, y1 = 32, self.canvas_size - 150, total_w - 32, self.canvas_size - 22
        draw.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=(31, 39, 51), outline=(95, 117, 146), width=2)
        draw.text((x0 + 18, y0 + 14), "ROBOT STATE", fill=(236, 242, 250), font=self.font)
        fields = list(env.robot_status().items())
        columns = 4
        col_w = (x1 - x0 - 32) // columns
        for i, (label, value) in enumerate(fields[:8]):
            col = i % columns
            row = i // columns
            px = x0 + 18 + col * col_w
            py = y0 + 44 + row * 38
            draw.text((px, py), str(label), fill=(148, 164, 185), font=self.font_small)
            draw.text((px, py + 16), str(value), fill=(247, 249, 252), font=self.font_small)


class LiveWindow:
    """Pygame live display with optional fullscreen presentation.

    F11 toggles fullscreen at runtime; Escape returns to windowed mode.  Frames are
    aspect-preserving scaled and centered, so the research display can fill a projector
    or monitor without changing the evaluator's rendered pixel geometry.
    """

    def __init__(self, fullscreen: bool = False) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        self.screen = None
        self.closed = False
        self.status_open = False
        self.fullscreen = bool(fullscreen)
        self._window_size: tuple[int, int] | None = None
        self._last_viewport = (0, 0, 1, 1)

    def _set_mode(self, frame_w: int, frame_h: int) -> None:
        pygame = self.pygame
        if self._window_size is None:
            self._window_size = (frame_w, frame_h)
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF)
        else:
            self.screen = pygame.display.set_mode(self._window_size, pygame.RESIZABLE | pygame.DOUBLEBUF)
        pygame.display.set_caption("TFP Research Visualizer · F11 fullscreen · Esc windowed")

    def _toggle_fullscreen(self, frame_w: int, frame_h: int) -> None:
        self.fullscreen = not self.fullscreen
        self._set_mode(frame_w, frame_h)

    def _frame_coordinates(self, pos: tuple[int, int], frame_w: int, frame_h: int) -> tuple[float, float]:
        vx, vy, vw, vh = self._last_viewport
        if vw <= 0 or vh <= 0:
            return -1.0, -1.0
        return (pos[0] - vx) * frame_w / vw, (pos[1] - vy) * frame_h / vh

    def show(self, frame: np.ndarray) -> bool:
        if self.closed:
            return False
        pygame = self.pygame
        h, w = frame.shape[:2]
        if self.screen is None:
            self._set_mode(w, h)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self._toggle_fullscreen(w, h)
                elif event.key == pygame.K_ESCAPE and self.fullscreen:
                    self.fullscreen = False
                    self._set_mode(w, h)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = self._frame_coordinates(event.pos, w, h)
                x0, y0, x1, y1 = FoxRenderer.STATUS_BUTTON
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.status_open = not self.status_open

        assert self.screen is not None
        screen_w, screen_h = self.screen.get_size()
        scale = min(screen_w / max(1, w), screen_h / max(1, h))
        draw_w = max(1, int(round(w * scale)))
        draw_h = max(1, int(round(h * scale)))
        ox = (screen_w - draw_w) // 2
        oy = (screen_h - draw_h) // 2
        self._last_viewport = (ox, oy, draw_w, draw_h)
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        if (draw_w, draw_h) != (w, h):
            surface = pygame.transform.smoothscale(surface, (draw_w, draw_h))
        self.screen.fill((16, 18, 22))
        self.screen.blit(surface, (ox, oy))
        pygame.display.flip()
        pygame.time.delay(45)
        return True

    def close(self) -> None:
        if not self.closed:
            self.pygame.quit()
            self.closed = True


class VideoRecorder:
    """Write playback-robust CFR H.264 MP4 episodes with an explicit outro.

    Every simulator state is encoded exactly once (including reset).  The terminal
    frame is held before a separate fade/end-card segment, so the actual terminal
    state occurs well before the container EOF.  This is deliberately redundant with
    CFR metadata: several desktop players visually truncate the final GOP even when
    ffprobe reports the correct duration.
    """

    def __init__(
        self,
        output_dir: str | Path,
        fps: int = 8,
        terminal_hold_seconds: float = 1.0,
        outro_seconds: float = 1.5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = max(1, int(fps))
        self.terminal_hold_seconds = max(0.0, float(terminal_hold_seconds))
        self.outro_seconds = max(0.0, float(outro_seconds))
        self.writer = None
        self.current_path: Path | None = None
        self.last_frame: np.ndarray | None = None
        self.frame_count = 0
        self.last_frame_count = 0
        self.episode_label = "Episode complete"

    def start_episode(self, map_stem: str, seed: int) -> None:
        self.finish_episode()
        import imageio.v2 as imageio

        path = self.output_dir / f"{map_stem}_seed{seed}.mp4"
        self.current_path = path
        self.last_frame = None
        self.frame_count = 0
        self.episode_label = f"Episode complete · {map_stem} · seed {seed}"
        self.writer = imageio.get_writer(
            path,
            fps=self.fps,
            codec="libx264",
            quality=7,
            pixelformat="yuv420p",
            macro_block_size=1,
            ffmpeg_log_level="error",
            output_params=[
                "-movflags", "+faststart",
                "-vsync", "cfr",
                "-r", str(self.fps),
                # Keep GOPs short and disable B-frame reordering. Some desktop players
                # visually stop at the final GOP even when MP4 duration metadata is
                # correct; a one-second GOP plus an explicit outro makes EOF robust.
                "-g", str(self.fps),
                "-keyint_min", str(self.fps),
                "-sc_threshold", "0",
                "-bf", "0",
            ],
        )

    def append(self, frame: np.ndarray) -> None:
        if self.writer is None:
            return
        stable = np.ascontiguousarray(frame, dtype=np.uint8)
        self.writer.append_data(stable)
        self.last_frame = stable.copy()
        self.frame_count += 1

    def _outro_frame(self, alpha: float) -> np.ndarray:
        assert self.last_frame is not None
        alpha = float(max(0.0, min(1.0, alpha)))
        base = Image.fromarray(self.last_frame).convert("RGB")
        dark = Image.new("RGB", base.size, (17, 20, 27))
        image = Image.blend(base, dark, alpha)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, image.height // 32))
            small = ImageFont.truetype("DejaVuSans.ttf", max(12, image.height // 50))
        except OSError:
            font = small = ImageFont.load_default()
        if alpha >= 0.55:
            title = "EPISODE COMPLETE"
            bbox = draw.textbbox((0, 0), title, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((image.width - tw) / 2, image.height * 0.44), title, fill=(245, 247, 250), font=font)
            bbox2 = draw.textbbox((0, 0), self.episode_label, font=small)
            tw2 = bbox2[2] - bbox2[0]
            draw.text(((image.width - tw2) / 2, image.height * 0.51), self.episode_label, fill=(191, 201, 214), font=small)
        return np.asarray(image, dtype=np.uint8)

    def finish_episode(self) -> None:
        if self.writer is None:
            return
        if self.last_frame is not None:
            hold = int(round(self.fps * self.terminal_hold_seconds))
            for _ in range(hold):
                self.writer.append_data(self.last_frame)
                self.frame_count += 1
            outro = int(round(self.fps * self.outro_seconds))
            for i in range(outro):
                # Fade during the first ~60%, then keep a stable end card.  Players
                # that visually clip the final few frames still show the terminal
                # state and completion cue before EOF.
                alpha = min(0.82, 0.82 * (i + 1) / max(1, int(outro * 0.6)))
                self.writer.append_data(self._outro_frame(alpha))
                self.frame_count += 1
        self.writer.close()
        self.writer = None
        self.last_frame_count = self.frame_count

    @property
    def encoded_duration_seconds(self) -> float:
        return float(self.last_frame_count if self.writer is None else self.frame_count) / float(self.fps)

    def close(self) -> None:
        self.finish_episode()
