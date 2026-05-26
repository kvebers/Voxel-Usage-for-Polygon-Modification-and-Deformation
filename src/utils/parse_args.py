import os
import argparse


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("scene", nargs="?", default="scene.jsonc", help="Scene config")
    arg_parser.add_argument("--record", nargs="?", const=True, metavar="OUTPUT.mp4", help="Record video")
    args = arg_parser.parse_args()
    record = os.path.splitext(os.path.basename(args.scene))[0] + ".mp4" if args.record is True else args.record
    return args.scene, record
