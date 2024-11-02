import argparse
import logging

from argparse import Namespace
from types import SimpleNamespace
from typing import List

from jsonargparse import ArgumentParser


def merge(parser, config_files: List[str], target_namespace: Namespace):
    """
    Parameters
    ----------
    config_files : list of str

    target_namespace : argparse.Namespace

    Returns
    -------
    target_namespace : argparse.Namespace
    """
    if config_files:
        parser.required_args = set()
        for config_file in config_files:
            cfg = parser.parse_path(config_file, defaults=False)
            _copy_from_dict_to_namespace(cfg, target_namespace)

    return target_namespace


def _copy_from_dict_to_namespace(source_namespace, target_namespace):

    queue = list(ConfigEntry(k, getattr(source_namespace, k), 0) for k in vars(source_namespace).keys())
    stack = []
    visited = set()

    while len(queue) > 0:
        entry = queue.pop()
        while len(stack) > 0 and stack[-1].level >= entry.level:
            stack.pop()

        k, v = entry.key, entry.val
        if isinstance(v, SimpleNamespace) or isinstance(v, Namespace):
            if k not in visited:
                queue.extend([ConfigEntry(kk, getattr(v, kk), entry.level+1)
                              for kk in vars(v).keys()])
                visited.add(k)
                target_sub_namespace = _get_sub_namespace(target_namespace, stack)
                if not hasattr(target_sub_namespace, k) or getattr(target_sub_namespace, k) is None:
                    setattr(target_sub_namespace, k, Namespace())

        elif not k.startswith('__'):
            if v is not None:
                target_sub_namespace = _get_sub_namespace(target_namespace, stack)
                setattr(target_sub_namespace, k, v)
        else:
            logging.info(f'Ignoring config key: {k}')

        if len(queue) > 0 and entry.level < queue[-1].level:
            stack.append(entry)

    return target_namespace


class ConfigEntry(object):
    def __init__(self, key, val, level):
        self.key = key
        self.val = val
        self.level = level

    def __str__(self):
        return f'{self.key}={self.val} (level: {self.level})'


def _get_sub_namespace(root_namespace, stack):

    if stack == []:
        return root_namespace

    sub_namespace = root_namespace
    for s in stack:
        sub_namespace = getattr(sub_namespace, s.key)

    return sub_namespace


def namespace_to_dict(source_namespace):

    source_namespace = vars(source_namespace)
    queue = [(None, k, v) for k, v in source_namespace.items()]

    while len(queue) > 0:
        parent, k, v = queue.pop()
        parent = source_namespace if parent is None else parent

        if isinstance(v, SimpleNamespace) or isinstance(v, Namespace):
            new_dict = vars(v)
            parent[k] = new_dict

            queue.extend([(parent[k], kk, vv) for kk, vv in new_dict.items()])

    return source_namespace


class KeywordArgs(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        kwargs = dict()
        for kw_entry in values.split(','):
            key, value = kw_entry.strip().split('=')
            if key in kwargs:
                raise ValueError(f'Multiple assignments for argument \'{key}\'')
            else:
                kwargs[key] = value
        setattr(namespace, self.dest, kwargs)


class GroupingArgParser(ArgumentParser):

    def parse_args(self, args=None, namespace=None):

        args = super().parse_args(args=args, namespace=namespace)

        args_grouped = self._group_args(args)
        #args_grouped = merge(self, args.config_files, args_grouped)

        args_grouped = namespace_to_dict(args_grouped)

        return args_grouped

    def parse_path(self, path, defaults=True):
        args = super().parse_path(path, defaults=defaults)
        args_grouped = self._group_args(args)

        return args_grouped

    def _group_args(self, args):

        args_grouped = Namespace()
        for group in self._action_groups:
            if group.title not in set(['positional arguments', 'optional arguments']):
                title_normalized = group.title.lower().replace(' ', '_')
                group_namespace = Namespace()

                for action in group._group_actions:
                    if hasattr(args, action.dest):
                        setattr(group_namespace, action.dest, getattr(args, action.dest))

                setattr(args_grouped, title_normalized, group_namespace)
            else:
                for action in group._group_actions:
                    if action.dest != 'help' and hasattr(args, action.dest):
                        setattr(args_grouped, action.dest, getattr(args, action.dest))
        return args_grouped
