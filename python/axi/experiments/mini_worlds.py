"""Small INVENTED grid worlds, one mechanic each, to check the transition ledger (engine/ledger.py, item 132, test 158).

These are NOT ARC-AGI-3 games and are not claimed to resemble one. They are written here because the ledger's logic can be checked exactly
only against worlds whose rules we know. The mechanics are the ones a pasted AI chat listed for a made-up "heavy" level (a far effect of a
switch, a global mode, a crushed object, a change that cannot be undone); that level itself cannot be won (its socket is walled in), so
only its list of mechanics is used, never the level.

Symbols: '#' wall, '.' floor, 'P' the player, 'B' a pushable block, 'T' a pad, 'r' / 'g' gates (r is solid in mode 0, g in mode 1),
'o' / 'e' the gauge (mode 0 / 1) in the top-left corner. Actions: U D L R (a step) and A (press: nothing happens in these worlds).
PushWorld: steps and pushes. PadWorld: stepping onto a pad flips the mode; a gate that becomes solid under a block crushes it. With
`visible=True` the mode shows (the gauge, and an open gate is drawn as floor); with `visible=False` it does not (the gauge never
changes and both gates are always drawn), which is the hidden-state world.
With `strip=True` a row is added at the bottom: a strip of 'z' that is used up from the right, one cell for every move that changed the screen, the used part drawn 'k'
(the shape the saved ARC-AGI-3 runs showed on 25 Oct: a one-row strip along the screen edge, eaten each move). It is an invented copy of that shape, not the game."""
from __future__ import annotations
import random
from typing import List, Tuple

STEP = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
ACTIONS = ("U", "D", "L", "R", "A")

PUSH_LEVEL = ("########",
              "#P..B..#",
              "#..#...#",
              "#...B..#",
              "#......#",
              "########")
PAD_LEVEL = ("##########",
             "#P.T..r..#",
             "#..B..r..#",
             "#.....g..#",
             "#.T...g.B#",
             "#........#",
             "##########")


class World:
    def __init__(self, level=PAD_LEVEL, visible: bool = True, pads: bool = True, strip: bool = False):
        self.h = len(level); self.w = len(level[0]); self.visible = visible; self.pads = pads; self.mode = 0; self.entries = 0; self.crushed = 0; self.strip = strip; self.used = 0
        self.cells = [list(r) for r in level]; self.player = next((x, y) for y, r in enumerate(level) for x, c in enumerate(r) if c == "P")
        self.cells[self.player[1]][self.player[0]] = "."
        self.blocks = {(x, y) for y, r in enumerate(level) for x, c in enumerate(r) if c == "B"}
        for x, y in self.blocks: self.cells[y][x] = "."

    def solid(self, x, y) -> bool:
        c = self.cells[y][x]
        return c == "#" or (c == "r" and self.mode == 0) or (c == "g" and self.mode == 1)

    def frame(self) -> Tuple[str, ...]:
        rows = []
        for y in range(self.h):
            row = []
            for x in range(self.w):
                c = self.cells[y][x]
                if (x, y) == self.player: row.append("P")
                elif (x, y) in self.blocks: row.append("B")
                elif c in "rg" and self.visible and not self.solid(x, y): row.append(".")
                else: row.append(c)
            rows.append("".join(row))
        if self.visible and self.pads: rows[0] = ("o" if self.mode == 0 else "e") + rows[0][1:]
        if self.strip: rows.append("z" * max(0, self.w - self.used) + "k" * min(self.w, self.used))
        return tuple(rows)

    def step(self, a: str) -> Tuple[str, ...]:
        before = (self.player, frozenset(self.blocks), self.mode)
        self._step(a)
        if self.strip and (self.player, frozenset(self.blocks), self.mode) != before: self.used += 1      # only a move that changed the screen is counted
        return self.frame()

    def _step(self, a: str) -> Tuple[str, ...]:
        if a in STEP:
            dx, dy = STEP[a]; nx, ny = self.player[0] + dx, self.player[1] + dy
            if not self.solid(nx, ny):
                if (nx, ny) in self.blocks:
                    bx, by = nx + dx, ny + dy
                    if self.solid(bx, by) or (bx, by) in self.blocks: return self.frame()
                    self.blocks.remove((nx, ny)); self.blocks.add((bx, by))
                self.player = (nx, ny)
                if self.pads and self.cells[ny][nx] == "T":
                    self.mode = 1 - self.mode; self.entries += 1; n = len(self.blocks)
                    self.blocks = {b for b in self.blocks if not self.solid(*b)}              # a block under a gate that turned solid is crushed
                    self.crushed += n - len(self.blocks)
        return self.frame()


def wide(level, w: int = 64) -> Tuple[str, ...]:
    """The same level on a screen `w` cells wide (walls fill the rest), so that a strip along the bottom is as wide as a real screen's."""
    return tuple(r.ljust(w, "#") for r in level)


def walk(world: World, seed: int, moves: int, ledger=None) -> List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]]:
    rng = random.Random(seed); out = []
    for _ in range(moves):
        a = rng.choice(ACTIONS); before = world.frame(); after = world.step(a); out.append((a, before, after))
        if ledger is not None: ledger.record(a, before, after)
    return out
