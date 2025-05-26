from dataclasses import dataclass

from dm_control.viewer import user_input


@dataclass
class KeyCallback:
    reset: bool = False
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_ENTER:
            self.reset = True
        elif key == user_input.KEY_SPACE:
            self.pause = not self.pause
