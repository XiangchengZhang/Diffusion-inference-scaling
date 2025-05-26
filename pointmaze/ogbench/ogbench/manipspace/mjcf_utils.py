from pathlib import Path
from typing import Any

import numpy as np
from dm_control import mjcf
from lxml import etree


def attach(
    parent_xml_or_model: Any,
    child_xml_or_model: Any,
    attach_site: Any = None,
    remove_keyframes: bool = True,
    add_freejoint: bool = False,
) -> mjcf.Element:
    if isinstance(parent_xml_or_model, Path):
        assert parent_xml_or_model.exists()
        parent = mjcf.from_path(parent_xml_or_model.as_posix())
    else:
        assert isinstance(parent_xml_or_model, mjcf.RootElement)
        parent = parent_xml_or_model

    if isinstance(child_xml_or_model, Path):
        assert child_xml_or_model.exists()
        child = mjcf.from_path(child_xml_or_model.as_posix())
    else:
        assert isinstance(child_xml_or_model, mjcf.RootElement)
        child = child_xml_or_model

    if attach_site is not None:
        if isinstance(attach_site, str):
            attachment_site = parent.find('site', attach_site)
            assert attachment_site is not None
        else:
            assert isinstance(attach_site, mjcf.Element)
            attachment_site = attach_site
        frame = attachment_site.attach(child)
    else:
        frame = parent.attach(child)
        if add_freejoint:
            frame.add('freejoint')

    if remove_keyframes:
        keyframes = parent.find_all('key')
        if keyframes is not None:
            for key in keyframes:
                key.remove()

    return frame


def to_string(
    root: mjcf.RootElement,
    precision: float = 17,
    zero_threshold: float = 0.0,
    pretty: bool = False,
) -> str:
    xml_string = root.to_xml_string(precision=precision, zero_threshold=zero_threshold)
    root = etree.XML(xml_string, etree.XMLParser(remove_blank_text=True))

    # Remove hashes from asset filenames.
    tags = ['mesh', 'texture']
    for tag in tags:
        assets = [asset for asset in root.find('asset').iter() if asset.tag == tag and 'file' in asset.attrib]
        for asset in assets:
            name, extension = asset.get('file').split('.')
            asset.set('file', '.'.join((name[:-41], extension)))  # Remove hash.

    if not pretty:
        return etree.tostring(root, pretty_print=True).decode()

    # Remove auto-generated names.
    for elem in root.iter():
        for key in elem.keys():
            if key == 'name' and 'unnamed' in elem.get(key):
                elem.attrib.pop(key)

    # Get string from lxml.
    xml_string = etree.tostring(root, pretty_print=True)

    # Remove redundant attributes.
    xml_string = xml_string.replace(b' gravcomp="0"', b'')

    # Insert spaces between top-level elements.
    lines = xml_string.splitlines()
    newlines = []
    for line in lines:
        newlines.append(line)
        if line.startswith(b'  <'):
            if line.startswith(b'  </') or line.endswith(b'/>'):
                newlines.append(b'')
    newlines.append(b'')
    xml_string = b'\n'.join(newlines)

    return xml_string.decode()


def get_assets(root: mjcf.RootElement) -> dict:
    assets = {}
    for file, payload in root.get_assets().items():
        name, extension = file.split('.')
        assets['.'.join((name[:-41], extension))] = payload  # Remove hash.
    return assets


def safe_find_all(root: mjcf.RootElement, namespace: str, *args, **kwargs):
    """Find all given elements or throw an error if none are found."""
    features = root.find_all(namespace, *args, **kwargs)
    if not features:
        raise ValueError(f'{namespace} not found in the MJCF model.')
    return features


def safe_find(root: mjcf.RootElement, namespace: str, identifier: str):
    """Find the given element or throw an error if not found."""
    feature = root.find(namespace, identifier)
    if feature is None:
        raise ValueError(f'{namespace} {identifier} not found.')
    return feature


def add_bounding_box_site(body: mjcf.Element, lower: np.ndarray, upper: np.ndarray, **kwargs) -> mjcf.Element:
    """Visualize a bounding box as a box site attached to the given body."""
    pos = (lower + upper) / 2
    size = (upper - lower) / 2
    size += 1e-7
    return body.add('site', type='box', pos=pos, size=size, **kwargs)
