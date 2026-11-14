"""Pygame front end: a menu, then a game screen (you vs a bot, or bot vs bot).

Keys: Esc = back to menu (quit from the menu), R = restart, Space = pause or
resume (bot vs bot), Right arrow = one move while paused.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import pygame

from app.bots import BOARDS, DEFAULT_BOT, BotMove, BotOption, BotWorker, available_bots, make_bot
from app.match import Match, Seat

W, H = 1120, 760
PANEL_X = 770
BOARD_RECT = pygame.Rect(24, 24, PANEL_X - 48, H - 48)

# Palette: a green games-table felt with chalk-coloured ink. Text colours keep a contrast of at
# least 4.5:1 on the felt; the mint accent marks choices and buttons, so it never reads as a seat.
BG = (21, 46, 38)
SURFACE = (28, 59, 49)
BORDER = (49, 89, 75)
INK = (238, 243, 236)
INK2 = (190, 209, 199)
MUTED = (138, 166, 153)
SLOT = (52, 94, 79)  # undrawn lines: faint guides on the felt
ACCENT = (126, 214, 166)
ERROR = (255, 143, 163)
SEAT_COLORS = ((242, 193, 78), (236, 112, 150))  # gold (player 1), rose (player 2)

THINK_TIMES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
SPEEDS: tuple[tuple[str, float], ...] = (("Fast", 0.15), ("Normal", 0.7), ("Slow", 1.6))
HUMAN_MIN_DELAY = 0.35  # seconds a bot "takes" at least in human games, so instant bots do not jump


def tint(color: tuple[int, int, int], amount: float, base: tuple[int, int, int] = SURFACE) -> tuple[int, int, int]:
    """``amount`` of ``color`` mixed into ``base``."""
    return tuple(round(b + (c - b) * amount) for c, b in zip(color, base))  # type: ignore[return-value]


@dataclass
class Button:
    rect: pygame.Rect
    action: Callable[[], None]
    enabled: bool = True


class App:
    def __init__(self, seed: int = 0, screen: pygame.Surface | None = None) -> None:
        pygame.init()
        self.screen = screen if screen is not None else pygame.display.set_mode((W, H))
        pygame.display.set_caption("Dots and Boxes")
        self.fonts = {
            "title": pygame.font.SysFont("segoeui,arial", 40, bold=True),
            "h2": pygame.font.SysFont("segoeui,arial", 22, bold=True),
            "body": pygame.font.SysFont("segoeui,arial", 18),
            "body_bold": pygame.font.SysFont("segoeui,arial", 18, bold=True),
            "small": pygame.font.SysFont("segoeui,arial", 15),
            "score": pygame.font.SysFont("segoeui,arial", 44, bold=True),
            "tag": pygame.font.SysFont("segoeui,arial", 16, bold=True),
        }
        self.seed = seed
        self.game_no = 0
        self.running = True
        self.view = "menu"
        self.buttons: list[Button] = []
        self.error = ""
        # menu choices
        self.mode = "human"  # "human" | "bots"
        self.board = (3, 3)
        self.think = 1.0
        self.human_first = True
        self.picks = {"opponent": DEFAULT_BOT, "p1": DEFAULT_BOT, "p2": ""}
        self.bots: list[BotOption] = []
        self.boards: list[tuple[int, int]] = []
        self.refresh_boards()
        self.refresh_bots()
        # game state
        self.match: Match | None = None
        self.worker = BotWorker()
        self.thinking_since = 0.0
        self.held: BotMove | None = None
        self.last_bot: tuple[str, BotMove] | None = None
        self.paused = False
        self.step = False
        self.speed = SPEEDS[1][1]
        self.note = ""
        self.hover: int | None = None

    # ----- menu data -------------------------------------------------------
    def refresh_boards(self) -> None:
        """Boards with at least one bot (larger boards appear once bots are optimised or pass the Greedy test)."""
        self.boards = [b for b in BOARDS if available_bots(*b)]
        if self.board not in self.boards:
            self.board = (3, 3) if (3, 3) in self.boards else self.boards[0]

    def refresh_bots(self) -> None:
        self.bots = available_bots(*self.board)
        keys = [b.key for b in self.bots]
        for slot in ("opponent", "p1"):
            if self.picks[slot] not in keys:
                self.picks[slot] = DEFAULT_BOT if DEFAULT_BOT in keys else keys[0]
        if self.picks["p2"] not in keys:  # a different bot from player 1 when there is one
            others = [k for k in keys if k != self.picks["p1"]]
            self.picks["p2"] = others[0] if others else keys[0]

    def bot(self, key: str) -> BotOption:
        return next(b for b in self.bots if b.key == key)

    def cycle(self, slot: str, step: int) -> None:
        keys = [b.key for b in self.bots]
        i = keys.index(self.picks[slot])
        self.picks[slot] = keys[(i + step) % len(keys)]

    def set_board(self, board: tuple[int, int]) -> None:
        self.board = board
        self.refresh_bots()

    # ----- game control ----------------------------------------------------
    def start_game(self) -> None:
        self.worker.cancel()
        self.game_no += 1
        base = self.seed * 1000 + self.game_no
        try:
            if self.mode == "human":
                bot = Seat(self.bot(self.picks["opponent"]).label, make_bot(self.picks["opponent"], 2 * base, self.board))
                you = Seat("You")
                seats = (you, bot) if self.human_first else (bot, you)
            else:
                seats = (Seat(self.bot(self.picks["p1"]).label, make_bot(self.picks["p1"], 2 * base, self.board)),
                         Seat(self.bot(self.picks["p2"]).label, make_bot(self.picks["p2"], 2 * base + 1, self.board)))
        except Exception as exc:  # e.g. a model file that fails to load
            self.error = f"Could not start: {type(exc).__name__}: {exc}"
            self.view = "menu"
            return
        self.error = ""
        self.match = Match(*self.board, seats)
        self.held = None
        self.last_bot = None
        self.paused = False
        self.step = False
        self.note = ""
        self.thinking_since = time.perf_counter()
        self.view = "game"

    def to_menu(self) -> None:
        self.worker.cancel()
        self.match = None
        self.view = "menu"
        self.refresh_boards()  # pick up bots optimised while the app was open
        self.refresh_bots()

    def who(self, seat: int) -> str:
        """Short name for a seat in status lines (the seat cards show the full bot names)."""
        assert self.match is not None
        if self.match.seats[seat].is_human:
            return "You"
        return "The bot" if self.mode == "human" else f"Player {seat + 1}"

    def play(self, edge: int) -> None:
        assert self.match is not None
        seat = self.match.to_move
        k = self.match.play(edge)
        if k and not self.match.over:
            verb = "complete" if self.match.seats[seat].is_human else "completes"
            self.note = f"{self.who(seat)} {verb} {k} box{'es' if k > 1 else ''} – extra turn"
        else:
            self.note = ""
        self.thinking_since = time.perf_counter()

    def update(self) -> None:
        """Advance bot turns: start thinking, collect the move, apply it after the display delay."""
        m = self.match
        if self.view != "game" or m is None or m.over or m.current.is_human:
            return
        bots_mode = self.mode == "bots"
        if self.held is None:
            busy = self.worker.busy  # read before poll: a move finishing in between is still collected
            self.held = self.worker.poll()
            if self.held is None:
                if not busy and not (bots_mode and self.paused and not self.step):
                    self.worker.start(m.current.player, m.state, self.think)  # type: ignore[arg-type]
                return
        if self.held.error:
            self.error = f"{m.current.name} failed: {self.held.error}"
            self.paused = True
            self.held = None
            return
        if bots_mode and self.paused and not self.step:
            return
        wait = self.speed if bots_mode else HUMAN_MIN_DELAY
        if time.perf_counter() - self.thinking_since < wait and not self.step:
            return
        move, self.held = self.held, None
        self.last_bot = (m.current.name, move)
        self.step = False
        self.play(move.move)

    # ----- input -----------------------------------------------------------
    def handle(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.view == "game":
                    self.to_menu()
                else:
                    self.running = False
            elif self.view == "game" and event.key == pygame.K_r:
                self.start_game()
            elif self.view == "game" and self.mode == "bots" and event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif self.view == "game" and self.mode == "bots" and event.key == pygame.K_RIGHT and self.paused:
                self.step = True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for b in self.buttons:
                if b.enabled and b.rect.collidepoint(event.pos):
                    b.action()
                    return
            m = self.match
            if self.view == "game" and m is not None and not m.over and m.current.is_human:
                edge = self.edge_at(event.pos)
                if edge is not None:
                    self.play(edge)

    # ----- board geometry --------------------------------------------------
    def geometry(self) -> tuple[float, float, float]:
        m, n = self.board
        top = BOARD_RECT.y + 110  # room for the title / result line
        area_w, area_h = BOARD_RECT.w - 80, BOARD_RECT.bottom - top - 36
        s = min(area_w / n, area_h / m)
        ox = BOARD_RECT.centerx - s * n / 2
        oy = top + (area_h - s * m) / 2
        return ox, oy, s

    def dot(self, r: int, c: int) -> tuple[float, float]:
        ox, oy, s = self.geometry()
        return ox + c * s, oy + r * s

    def endpoints(self, edge: int) -> tuple[tuple[float, float], tuple[float, float]]:
        m, n = self.board
        h = (m + 1) * n
        if edge < h:
            r, c = divmod(edge, n)
            return self.dot(r, c), self.dot(r, c + 1)
        r, c = divmod(edge - h, n + 1)
        return self.dot(r, c), self.dot(r + 1, c)

    def edge_at(self, pos: tuple[int, int]) -> int | None:
        """The undrawn edge nearest to ``pos`` within a click radius, if any."""
        if self.match is None:
            return None
        _, _, s = self.geometry()
        best, best_d = None, s * 0.3
        px, py = pos
        for e in self.match.legal():
            (x1, y1), (x2, y2) = self.endpoints(e)
            dx, dy = x2 - x1, y2 - y1
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
            d = math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
            if d < best_d:
                best, best_d = e, d
        return best

    # ----- drawing helpers -------------------------------------------------
    def text(self, s: str, pos: tuple[float, float], font: str = "body", color: tuple[int, int, int] = INK,
             anchor: str = "topleft") -> pygame.Rect:
        surf = self.fonts[font].render(s, True, color)
        rect = surf.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
        self.screen.blit(surf, rect)
        return rect

    def fit(self, s: str, font: str, max_w: int) -> str:
        """``s`` cut with an ellipsis so it renders no wider than ``max_w`` pixels."""
        f = self.fonts[font]
        if f.size(s)[0] <= max_w:
            return s
        while s and f.size(s + "…")[0] > max_w:
            s = s[:-1]
        return s.rstrip() + "…"

    def wrap(self, s: str, font: str, max_w: int, max_lines: int) -> list[str]:
        """``s`` broken into at most ``max_lines`` lines no wider than ``max_w`` (the last one cut if needed)."""
        f, lines, line = self.fonts[font], [], ""
        for word in s.split():
            trial = f"{line} {word}".strip()
            if line and f.size(trial)[0] > max_w:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        if len(lines) > max_lines:
            lines = lines[:max_lines - 1] + [" ".join(lines[max_lines - 1:])]
        return [self.fit(ln, font, max_w) for ln in lines]

    def button(self, rect: pygame.Rect, label: str, action: Callable[[], None], *, selected: bool = False,
               primary: bool = False, enabled: bool = True, font: str = "body") -> None:
        hover = enabled and rect.collidepoint(pygame.mouse.get_pos())
        if primary:
            fill, fg, edge = (ACCENT if enabled else BORDER), (BG if enabled else MUTED), None
            font = "body_bold" if font == "body" else font  # thin dark text on mint reads faint
        elif selected:
            fill, fg, edge = tint(ACCENT, 0.18), INK, ACCENT
        else:
            fill, fg, edge = (tint(INK, 0.05) if hover else SURFACE), (INK if enabled else MUTED), BORDER
        pygame.draw.rect(self.screen, fill, rect)
        if edge is not None:
            pygame.draw.rect(self.screen, edge, rect, width=2 if selected else 1)
        self.text(label, rect.center, font, fg, "center")
        self.buttons.append(Button(rect, action, enabled))

    def chips(self, x: int, y: int, options: list[tuple[str, bool, Callable[[], None]]], width: int = 0) -> int:
        for label, selected, action in options:
            w = width or self.fonts["body"].size(label)[0] + 32
            self.button(pygame.Rect(x, y, w, 38), label, action, selected=selected)
            x += w + 10
        return y + 38

    def selector(self, x: int, y: int, slot: str, caption: str, color: tuple[int, int, int] | None = None) -> int:
        self.text(caption, (x, y), "body", INK2)
        if color is not None:
            pygame.draw.rect(self.screen, color, (x - 21, y + 5, 14, 14))
        y += 28
        keys = [b.key for b in self.bots]
        opt = self.bot(self.picks[slot])
        self.button(pygame.Rect(x, y, 40, 40), "<", lambda: self.cycle(slot, -1))
        box = pygame.Rect(x + 48, y, 520, 40)
        pygame.draw.rect(self.screen, SURFACE, box)
        pygame.draw.rect(self.screen, BORDER, box, width=1)
        self.text(opt.label, (box.x + 14, box.centery), "body", INK, "midleft")
        self.text(f"{keys.index(opt.key) + 1}/{len(keys)}", (box.right - 12, box.centery), "small", MUTED, "midright")
        self.button(pygame.Rect(box.right + 8, y, 40, 40), ">", lambda: self.cycle(slot, 1))
        for i, line in enumerate(self.wrap(opt.description, "small", W - 24 - (x + 48), 2)):
            self.text(line, (x + 48, y + 48 + 19 * i), "small", MUTED)
        return y + 92  # room for two description lines, so the layout does not jump between bots

    # ----- screens ---------------------------------------------------------
    def draw_menu(self) -> None:
        x = 190
        self.text("Dots and Boxes", (x, 40), "title")
        self.text("Play against one of the project's solvers, or watch two of them play.", (x, 96), "body", INK2)
        y = 140
        self.text("Mode", (x, y), "h2")
        self.chips(x, y + 34, [("Play against a bot", self.mode == "human", lambda: setattr(self, "mode", "human")),
                               ("Watch two bots", self.mode == "bots", lambda: setattr(self, "mode", "bots"))])
        y += 90
        self.text("Board (boxes)", (x, y), "h2")
        self.chips(x, y + 34, [(f"{m}×{n}", self.board == (m, n), lambda b=(m, n): self.set_board(b))
                               for m, n in self.boards], width=76)
        y += 90
        if self.mode == "human":
            self.text("Opponent", (x, y), "h2")
            y = self.selector(x, y + 34, "opponent", "Bot")
            self.text("Who moves first", (x, y), "h2")
            self.chips(x, y + 34, [("You", self.human_first, lambda: setattr(self, "human_first", True)),
                                   ("The bot", not self.human_first, lambda: setattr(self, "human_first", False))],
                       width=120)
            y += 90
        else:
            self.text("Bots", (x, y), "h2")
            y = self.selector(x, y + 34, "p1", "Player 1 (moves first)", SEAT_COLORS[0])
            y = self.selector(x, y, "p2", "Player 2", SEAT_COLORS[1])
            y += 4
        self.text("Thinking time per move", (x, y), "h2")
        self.text("(an upper limit for the bots that search or simulate)", (x + 270, y + 5), "small", MUTED)
        self.chips(x, y + 34, [(f"{t:g} s", self.think == t, lambda t=t: setattr(self, "think", t))
                               for t in THINK_TIMES], width=90)
        y += 90
        self.button(pygame.Rect(x, y, 220, 50), "Start", self.start_game, primary=True, font="h2")
        if self.error:  # beside the button: below it would leave the window in bot-vs-bot mode
            for i, line in enumerate(self.wrap(self.error, "small", W - 24 - (x + 240), 2)):
                self.text(line, (x + 240, y + 6 + 19 * i), "small", ERROR)
        self.text("Esc quits", (W - 24, H - 24), "small", MUTED, "bottomright")

    def draw_board(self) -> None:
        m = self.match
        assert m is not None
        pygame.draw.rect(self.screen, SURFACE, BOARD_RECT)
        pygame.draw.rect(self.screen, BORDER, BOARD_RECT, width=1)
        bm, bn = self.board
        if m.over:
            self.text(self.result_text(), (BOARD_RECT.centerx, BOARD_RECT.y + 56), "title", INK, "center")
        else:
            title = self.fit(f"{m.seats[0].name}  vs  {m.seats[1].name}", "h2", BOARD_RECT.w - 40)
            self.text(title, (BOARD_RECT.centerx, BOARD_RECT.y + 56), "h2", INK, "center")
        ox, oy, s = self.geometry()
        if self.mode == "human":
            tags = tuple("You" if seat.is_human else "Bot" for seat in m.seats)
        else:
            tags = ("P1", "P2")
        board = m.state.board
        for box, owner in enumerate(m.owners):
            if owner < 0:
                continue
            r, c = divmod(box, bn)
            rect = pygame.Rect(round(ox + c * s), round(oy + r * s), round(s), round(s))
            tile = rect.inflate(-2 * max(6, round(s * 0.16)), -2 * max(6, round(s * 0.16)))
            pygame.draw.rect(self.screen, SEAT_COLORS[owner], tile)
            self.text(tags[owner], tile.center, "tag", BG, "center")
        width = max(4, round(s * 0.06))
        last = m.last
        if last is not None:  # halo under the most recent line
            p1, p2 = self.endpoints(last.edge)
            pygame.draw.line(self.screen, tint(INK, 0.22), p1, p2, width + 10)  # soft chalk glow
        for e in range(board.E):
            p1, p2 = self.endpoints(e)
            owner = m.edge_owner.get(e)
            if owner is None:
                color = tint(SEAT_COLORS[m.to_move], 0.75) if e == self.hover else SLOT
                pygame.draw.line(self.screen, color, p1, p2, width if e == self.hover else 2)
            else:
                pygame.draw.line(self.screen, SEAT_COLORS[owner], p1, p2, width)
        for r in range(bm + 1):
            for c in range(bn + 1):
                pygame.draw.circle(self.screen, INK, self.dot(r, c), max(5, round(s * 0.05)))

    def result_text(self) -> str:
        m = self.match
        assert m is not None
        s0, s1 = m.state.scores
        w = m.winner
        if w == -1:
            return f"Draw {s0}–{s1}"
        assert w is not None
        hi, lo = max(s0, s1), min(s0, s1)
        seat = m.seats[w]
        if self.mode == "human":
            return f"You win {hi}–{lo}!" if seat.is_human else f"The bot wins {hi}–{lo}"
        return f"Player {w + 1} wins {hi}–{lo}"

    def draw_panel(self) -> None:
        m = self.match
        assert m is not None
        x, w = PANEL_X, W - PANEL_X - 24
        mode = "You vs a bot" if self.mode == "human" else "Bot vs bot"
        self.text(mode, (x, 28), "h2")
        self.text(f"{self.board[0]}×{self.board[1]} board · {self.think:g} s per move", (x, 60), "small", MUTED)
        y = 96
        for i, seat in enumerate(m.seats):
            card = pygame.Rect(x, y, w, 92)
            active = not m.over and m.to_move == i
            pygame.draw.rect(self.screen, tint(SEAT_COLORS[i], 0.10) if active else SURFACE, card)
            pygame.draw.rect(self.screen, SEAT_COLORS[i] if active else BORDER, card, width=2 if active else 1)
            pygame.draw.rect(self.screen, SEAT_COLORS[i], (card.x + 14, card.y + 18, 16, 16))
            label = self.fit("You" if seat.is_human else seat.name, "body", card.w - 40 - 80)
            self.text(label, (card.x + 40, card.y + 14), "body")
            self.text("moves first" if i == 0 else "moves second", (card.x + 40, card.y + 42), "small", MUTED)
            self.text(str(m.state.scores[i]), (card.right - 18, card.centery), "score",
                      tint(SEAT_COLORS[i], 0.9, INK), "midright")
            y += 104
        y += 4
        if m.over:
            status = self.result_text()
        elif m.current.is_human:
            status = "Your move: click a line"
        elif self.mode == "bots" and self.paused:
            status = "Paused"
        else:
            dots = "." * (1 + int(time.perf_counter() * 3) % 3)
            status = f"{self.who(m.to_move)} is thinking{dots}"
        self.text(self.fit(status, "body", w), (x, y), "body", INK)
        if self.note:
            self.text(self.fit(self.note, "small", w), (x, y + 28), "small", INK2)
        if self.last_bot is not None:
            name, mv = self.last_bot
            info = f"{mv.seconds:.2f} s"
            if mv.nodes:
                info += f" · {mv.nodes:,} node{'s' if mv.nodes != 1 else ''}"
            if mv.depth:
                info += f" · depth {mv.depth}"
            self.text(self.fit(f"Last bot move: {info}", "small", w), (x, y + 54), "small", MUTED)
        if self.error:
            self.text(self.fit(self.error, "small", w), (x, y + 76), "small", ERROR)
        y += 110
        if self.mode == "bots":
            self.text("Playback", (x, y), "h2")
            y += 34
            self.button(pygame.Rect(x, y, 150, 40), "Resume" if self.paused else "Pause",
                        lambda: setattr(self, "paused", not self.paused), enabled=not m.over)
            self.button(pygame.Rect(x + 160, y, 150, 40), "Step", lambda: setattr(self, "step", True),
                        enabled=self.paused and not m.over)
            y += 52
            self.chips(x, y, [(label, self.speed == d, lambda d=d: setattr(self, "speed", d))
                              for label, d in SPEEDS], width=96)
            y += 56
        bottom = H - 120
        self.button(pygame.Rect(x, bottom, 150, 44), "Play again" if m.over else "Restart", self.start_game,
                    primary=m.over)
        self.button(pygame.Rect(x + 160, bottom, 150, 44), "Menu", self.to_menu)
        keys = "Esc menu · R restart" + (" · Space pause · → step" if self.mode == "bots" else "")
        self.text(keys, (x, H - 40), "small", MUTED)

    def render(self) -> None:
        self.buttons = []
        self.screen.fill(BG)
        if self.view == "menu":
            self.draw_menu()
        else:
            m = self.match
            mouse = pygame.mouse.get_pos()
            self.hover = (self.edge_at(mouse) if m is not None and not m.over and m.current.is_human else None)
            self.draw_board()
            self.draw_panel()

    def run(self) -> None:
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                self.handle(event)
            self.update()
            self.render()
            pygame.display.flip()
            clock.tick(60)
        self.worker.cancel()
        pygame.quit()
