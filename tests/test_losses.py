from losses import FocalLoss


def test_loss(batch):
    x, y = batch
    focal_loss = FocalLoss()
    loss = focal_loss(x, x)
    assert loss.item() == 0


def test_metric_func(args, batch, trainer):
    x, y = batch
    backbone = trainer.backbone
    metric_fn = trainer.metric_fn

    feature = backbone(x)
    logits = metric_fn(feature, y)
    assert list(logits.shape) == [args.batch_size, args.num_classes]
