import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from taxon_vision.inference.gradcam import generate_heatmap
from taxon_vision.models.factory import (
    create_feature_extractor,
    extract_and_cache_features,
    extract_features,
    get_feature_dimension,
)
from taxon_vision.models.head import TaxonClassifier
from taxon_vision.models.loss import ClassBalancedLoss
from taxon_vision.models.trainer import (
    EmbeddingSplit,
    HeadTrainingConfig,
    TuningConfig,
    evaluate_head,
    train_head_epoch,
    train_head_on_cached_embeddings,
    tune_hyperparameters,
)


def test_feature_extractor_factory_and_caching() -> None:
    backbone = create_feature_extractor("mobilenetv4_conv_small", pretrained=False)
    dim = get_feature_dimension(backbone)
    assert dim > 0

    dummy_images = torch.randn(2, 3, 224, 224)
    features = extract_features(backbone, dummy_images)
    assert features.shape[0] == 2
    assert features.shape[1] == dim

    dataloader = [
        (torch.randn(2, 3, 224, 224), torch.tensor([0, 1])),
        (torch.randn(2, 3, 224, 224), torch.tensor([1, 2])),
    ]
    cached_feats, cached_lbls = extract_and_cache_features(backbone, dataloader)
    assert cached_feats.shape == (4, dim)
    assert cached_lbls.shape == (4,)


def test_feature_extractor_and_head() -> None:
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 64))
    classifier = TaxonClassifier(backbone=backbone, feature_dim=64, num_classes=5)

    dummy_input = torch.randn(2, 3, 224, 224)
    logits = classifier(dummy_input)
    assert logits.shape == (2, 5)

    # Test direct embedding forward pass
    dummy_feats = torch.randn(2, 64)
    head_logits = classifier.forward_head(dummy_feats)
    assert head_logits.shape == (2, 5)


def test_train_head_epoch_and_evaluate() -> None:
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 16))
    classifier = TaxonClassifier(backbone=backbone, feature_dim=16, num_classes=3)
    optimizer = optim.SGD(classifier.head.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    dataloader = [(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 1]))]

    avg_loss = train_head_epoch(classifier, dataloader, optimizer, criterion)
    assert avg_loss >= 0.0

    val_loss, val_acc = evaluate_head(classifier, dataloader, criterion)
    assert val_loss >= 0.0
    assert 0.0 <= val_acc <= 1.0


def test_train_head_on_cached_embeddings() -> None:
    feature_dim = 16
    num_classes = 3
    head = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(feature_dim, num_classes))
    optimizer = optim.AdamW(head.parameters(), lr=0.01)
    criterion = ClassBalancedLoss(samples_per_class=[20, 15, 5], beta=0.99)

    train_z = torch.randn(40, feature_dim)
    train_y = torch.randint(0, num_classes, (40,))
    val_z = torch.randn(10, feature_dim)
    val_y = torch.randint(0, num_classes, (10,))

    data = EmbeddingSplit(
        train_embeddings=train_z,
        train_labels=train_y,
        val_embeddings=val_z,
        val_labels=val_y,
    )
    config = HeadTrainingConfig(epochs=3, batch_size=16)

    history = train_head_on_cached_embeddings(
        head=head,
        data=data,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
    )
    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3
    assert len(history["val_accuracy"]) == 3


def test_tune_hyperparameters_optuna(tmp_path: pytest.TempPathFactory) -> None:
    feature_dim = 8
    num_classes = 2
    train_z = torch.randn(20, feature_dim)
    train_y = torch.randint(0, num_classes, (20,))
    val_z = torch.randn(10, feature_dim)
    val_y = torch.randint(0, num_classes, (10,))

    local_mlflow_uri = f"file://{tmp_path}/mlruns"

    data = EmbeddingSplit(
        train_embeddings=train_z,
        train_labels=train_y,
        val_embeddings=val_z,
        val_labels=val_y,
        samples_per_class=[12, 8],
    )
    config = TuningConfig(
        n_trials=2,
        epochs_per_trial=2,
        batch_size=8,
        tracking_uri=local_mlflow_uri,
        experiment_name="test-tuning",
    )

    best_params, study = tune_hyperparameters(
        data=data,
        num_classes=num_classes,
        config=config,
    )
    assert "lr" in best_params
    assert "weight_decay" in best_params
    assert "dropout" in best_params
    assert "beta" in best_params
    assert len(study.trials) == 2


def test_embedding_split_validation() -> None:
    # Dimension mismatch check
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        EmbeddingSplit(
            train_embeddings=torch.randn(10, 16),
            train_labels=torch.zeros(10, dtype=torch.long),
            val_embeddings=torch.randn(5, 32),
            val_labels=torch.zeros(5, dtype=torch.long),
        )

    # Sample count mismatch check
    with pytest.raises(ValueError, match="Train embeddings count"):
        EmbeddingSplit(
            train_embeddings=torch.randn(10, 16),
            train_labels=torch.zeros(8, dtype=torch.long),
            val_embeddings=torch.randn(5, 16),
            val_labels=torch.zeros(5, dtype=torch.long),
        )


def test_gradcam_heatmap() -> None:
    img = Image.new("RGB", (100, 100))
    heatmap = generate_heatmap(img)
    assert heatmap.shape == (100, 100)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
