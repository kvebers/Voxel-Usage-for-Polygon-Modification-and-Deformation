import json


def load_config(path="scene.json"):
    with open(path, 'r') as f:
        return json.load(f)