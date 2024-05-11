import json


def load_json(file):
    """load json file"""
    with open(file=file, mode='r') as fp:
        data = json.load(fp=fp)
    return data


def dump_json(file, obj):
    """dump json file"""
    with open(file=file, mode='w') as fp:
        json.dump(obj=obj, fp=fp)
    return file