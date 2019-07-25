
"""
ref to https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
"""
HEADER_SIGN = "⯈"

FMT = "\x1b[6;30;4{c}m \x1b[0m\x1b[6;3{c};40m ⯈\x1b[0m \x1b[1;3{c};40m {{s}}\x1b[0m"  

WARMN = FMT.format(c=1)
TRAIN = FMT.format(c=2)
VALID = FMT.format(c=3)
HINTS = FMT.format(c=4)
BUILD = FMT.format(c=5)

def cprint(f, context):
    print(f.format(s=context))



from tabulate import tabulate
from collections import defaultdict

losses_history = defaultdict(list)

def get_changing_str(number):
    arrow = '⮥' if number > 0 else '⮧'
    return arrow + '  ' + '%.5f' % (abs(number))

def tabulate_print_losses(losses, trace, mode = 'train'):
    assert mode in ['train', 'valid']  

    historys = losses_history[trace]

    items = [
        (c[0], c[1], get_changing_str(c[1]-h[1])) for (h, c) in zip(historys[-1], losses)
    ] if len(historys) > 0 else [
        (c[0], c[1], 'NONE') for c in losses
    ]
    historys.append(losses)

    table = tabulate(
        items,
        tablefmt="grid",
    )
    lines = table.split('\n')
    log_mode =  TRAIN if mode is 'train' else VALID
    for l in lines:
        cprint(log_mode, l)


if __name__ == "__main__":
    cprint(TRAIN, "aaa ") 
    cprint(TRAIN, "dd") 
    cprint(TRAIN, "ff") 
    cprint(VALID, "aaaa") 
    cprint(WARMN, "aaaa") 
    cprint(HINTS, "aaaa") 



