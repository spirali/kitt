from kitt.dataloading.preprocessing import ScalePreprocessing


def test_scale_preprocessing_roundtrip():
    fn = ScalePreprocessing(2.0)

    input = 5
    assert fn.denormalize(fn.normalize(input)) == input
