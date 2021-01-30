from tensorflow.python.keras.applications.resnet import ResNet50

from kitt.model import load_model_from_bytes, save_model_to_bytes


def test_save_load_model():
    model = ResNet50(weights=None)
    data = save_model_to_bytes(model)
    loaded = load_model_from_bytes(data)
    assert loaded.get_config() == model.get_config()

    w_orig = model.get_weights()
    w_loaded = loaded.get_weights()
    assert len(w_orig) == len(w_loaded)
    for (orig, loaded) in zip(w_orig, w_loaded):
        assert (orig == loaded).all()
