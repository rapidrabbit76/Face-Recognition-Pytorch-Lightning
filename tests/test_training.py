import pytest


@pytest.mark.parametrize("train", [True, False])
def test_backbone_forward(args, batch, backbone, train):
    x, _ = batch
    backbone = backbone if train else backbone.eval()
    x = backbone(x)
    assert list(x.shape) == [args.batch_size, args.embedding_size]


class TestTrainer:
    @pytest.fixture(scope="class")
    def val_batch_output(self, test_batch, trainer):
        return trainer.validation_step(test_batch, 0)

    @pytest.fixture(scope="class")
    def test_batch_output(self, test_batch, trainer):
        return trainer.test_step(test_batch, 0)

    def test_pl_training_step(self, batch, trainer):
        x = trainer.training_step(batch, 0)
        assert list(x.shape) == []

    def test_pl_validation_step(self, args, val_batch_output):
        check_shape_test_validation_step(args, *val_batch_output)

    def test_pl_test_step(self, args, test_batch_output):
        check_shape_test_validation_step(args, *test_batch_output)

    def test_pl_module_validation_epoch_end(self, val_batch_output, trainer):
        log = trainer.validation_epoch_end([val_batch_output for _ in range(2)])
        item = log["val/acc"].item()
        assert item == 0 or item == 1
        assert isinstance(log, dict)

    def test_pl_module_test_epoch_end(self, test_batch_output, trainer):
        log = trainer.test_epoch_end([test_batch_output for _ in range(2)])
        item = log["test/acc"].item()
        assert item == 0 or item == 1
        assert isinstance(log, dict)


def check_shape_test_validation_step(args, feature_a, feature_b, label):
    assert list(feature_a.shape) == [args.batch_size, args.embedding_size]
    assert feature_a.shape == feature_b.shape
    assert list(label.shape) == [args.batch_size]
