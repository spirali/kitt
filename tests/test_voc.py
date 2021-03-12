from kitt.image.objdetect.annotation import BoundingBox, AnnotationType
from kitt.image.objdetect.voc import voc_to_annotated_image
from tests.conftest import data_path


def test_load_voc_xml():
    annotated = voc_to_annotated_image(data_path("example.xml"))
    width, height = annotated.width, annotated.height
    assert width == 500
    assert height == 375

    assert annotated.annotations[0].class_name == "dog"
    assert annotated.annotations[0].confidence is None
    assert annotated.annotations[0].annotation_type == AnnotationType.GROUND_TRUTH
    assert annotated.annotations[0].bbox.denormalize(
        width, height
    ).to_int() == BoundingBox(144, 255, 90, 201)
    assert annotated.annotations[1].class_name == "dog"
    assert annotated.annotations[1].bbox.denormalize(
        width, height
    ).to_int() == BoundingBox(264, 380, 73, 180)
