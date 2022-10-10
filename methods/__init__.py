from .BYOL import BYOL
from .RSA import RSA


METHOD_LIST = ["byol", "rsa"]


def create_method(args):
    assert args.method in METHOD_LIST

    if args.method == 'byol':
        model = BYOL(args.arch, args.max_step)
    elif args.method == 'rsa':
        model = RSA(args.arch, max_step=args.max_step, beta=args.beta)

    return model
