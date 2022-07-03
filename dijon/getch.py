import sys


class __DefaultGetch:
    def __init__(self):
        self.buffer = b''

    def getch(self):
        if not self.buffer:
            self.buffer = sys.stdin.read().encode('ascii')

        ch = self.buffer[0]
        self.buffer = self.buffer[1:]
        return ch


def __win_getch():
    import msvcrt
    return msvcrt.getch()[0]


def __unix_getch():
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1).encode('ascii')[0]
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# Choose an appropriate implementation
__impl = __DefaultGetch().getch
try:
    import msvcrt
    __impl = __win_getch
except ImportError:
    try:
        import termios
        termios.tcgetattr(sys.stdin.fileno())
        __impl = __unix_getch
    except termios.error:
        pass


def getch() -> int:
    return __impl()
