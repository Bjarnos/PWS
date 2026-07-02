from colorama import init, Fore, Style

init(autoreset=True)

class Colors:
    PROGRESS     = Style.BRIGHT + Fore.CYAN
    BAR_FILLED   = Fore.GREEN
    BAR_EMPTY    = Style.DIM + Fore.WHITE
    HEADER       = Style.BRIGHT + Fore.MAGENTA
    LABEL        = Style.BRIGHT + Fore.YELLOW
    VALUE        = Fore.WHITE
    FINISH       = Style.BRIGHT + Fore.GREEN
    TIME         = Fore.CYAN
    RESET        = Style.RESET_ALL

theme = Colors()
