import importlib.util

import cv2
import numpy as np
import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location("prepare_mar20", ROOT / "scripts" / "prepare_mar20.py")
prepare_mar20 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_mar20)

XML = """<annotation><filename>{id}.jpg</filename><size><width>800</width><height>400</height><depth>3</depth></size>
<object><type>robndbox</type><name>A14</name><robndbox>
<x_left_top>400</x_left_top><y_left_top>200</y_left_top><x_right_top>560</x_right_top><y_right_top>200</y_right_top>
<x_right_bottom>560</x_right_bottom><y_right_bottom>300</y_right_bottom><x_left_bottom>400</x_left_bottom><y_left_bottom>300</y_left_bottom>
</robndbox></object>
<object><type>robndbox</type><name>A1</name><robndbox>
<x_left_top>-8</x_left_top><y_left_top>10</y_left_top><x_right_top>80</x_right_top><y_right_top>10</y_right_top>
<x_right_bottom>80</x_right_bottom><y_right_bottom>50</y_right_bottom><x_left_bottom>-8</x_left_bottom><y_left_bottom>50</y_left_bottom>
</robndbox></object></annotation>"""


@pytest.fixture
def mar20_root(tmp_path):
    root = tmp_path / "MAR20"
    ann = root / "Annotations" / "Oriented Bounding Boxes"
    for d in (ann, root / "JPEGImages", root / "ImageSets" / "Main"):
        d.mkdir(parents=True)
    ids = [str(i) for i in range(1, 11)]
    for image_id in ids:
        (ann / f"{image_id}.xml").write_text(XML.format(id=image_id))
        cv2.imwrite(str(root / "JPEGImages" / f"{image_id}.jpg"), np.zeros((400, 800, 3), np.uint8))
    (root / "ImageSets" / "Main" / "train.txt").write_text("\n".join(ids[:6]))
    (root / "ImageSets" / "Main" / "test.txt").write_text("\n".join(ids[6:]))
    return root


def test_convert_mar20(mar20_root, tmp_path):
    out = tmp_path / "yolo"
    stats = prepare_mar20.convert(mar20_root, out, val_frac=0.34)
    assert stats["images"] == {"train": 4, "val": 2, "test": 4} and stats["missing"] == 0
    assert stats["instances"]["test"]["KC-135"] == 4 and stats["instances"]["test"]["SU-35"] == 4

    label = (out / "labels" / "test" / "7.txt").read_text().splitlines()
    cls, *coords = label[0].split()
    assert cls == "13"  # A14 → KC-135
    assert np.allclose([float(v) for v in coords], [0.5, 0.5, 0.7, 0.5, 0.7, 0.75, 0.5, 0.75])
    assert float(label[1].split()[1]) == 0.0  # координаты за краем обрезаны до [0, 1]

    assert (out / "images" / "test" / "7.jpg").exists()
    yaml = (out / "mar20.yaml").read_text()
    assert f"path: {out.resolve()}" in yaml and "  13: KC-135" in yaml and "test: images/test" in yaml
