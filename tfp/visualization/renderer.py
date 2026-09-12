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
            if env.TASK_CODE in {"LOCAL-TRANSPORT", "ROOM-DOOR-TRANSPORT", "MOVING-CARGO-EVASION"}:
                fill, outline = (104, 187, 139), (59, 139, 98)
            else:
                base = color_palette[int(color) % len(color_palette)]
                fill = tuple(min(255, int(v + (255 - v) * 0.55)) for v in base)
                outline = base
            draw.rounded_rectangle(
                (gx0 + pad, gy0 + pad, gx0 + tile - pad, gy0 + tile - pad),
                radius=max(3, tile // 5), fill=fill, outline=outline, width=max(1, tile // 10),
            )

        for color, (rr, rc) in entities.get("objects", []):
            x0, y0 = ox + rc * tile, oy + rr * tile
            p = max(3, tile // 5)
            if env.TASK_CODE in {"LOCAL-TRANSPORT", "ROOM-DOOR-TRANSPORT", "MOVING-CARGO-EVASION"}:
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
    """Pygame live display with a clickable robot-state drawer toggle."""

    def __init__(self) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        self.screen = None
        self.closed = False
        self.status_open = False

    def show(self, frame: np.ndarray) -> bool:
        if self.closed:
            return False
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                x0, y0, x1, y1 = FoxRenderer.STATUS_BUTTON
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.status_open = not self.status_open
        h, w = frame.shape[:2]
        if self.screen is None:
            self.screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("TFP Research Visualizer")
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        pygame.time.delay(45)
        return True

    def close(self) -> None:
        if not self.closed:
            self.pygame.quit()
            self.closed = True


class VideoRecorder:
    """Write one MP4 per representative evaluated map/seed episode."""

    def __init__(self, output_dir: str | Path, fps: int = 8) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writer = None

    def start_episode(self, map_stem: str, seed: int) -> None:
        self.finish_episode()
        import imageio.v2 as imageio

        path = self.output_dir / f"{map_stem}_seed{seed}.mp4"
        self.writer = imageio.get_writer(path, fps=self.fps, codec="libx264", quality=7)

    def append(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            self.writer.append_data(frame)

    def finish_episode(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def close(self) -> None:
        self.finish_episode()
