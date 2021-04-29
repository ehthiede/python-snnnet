import pytest
import torch

from snnnet import metrics


def test_binary_classification_summary_all_correct():
    summary = metrics.BinaryClassificationSummary()

    true_labels = torch.ones(10, dtype=torch.int64)
    predicted_labels = torch.ones(10, dtype=torch.int64)

    summary.record_results(
        true_labels, predicted_labels)

    assert summary.count == 10
    assert torch.allclose(summary.accuracy, torch.tensor(1.))
    assert torch.allclose(summary.precision, torch.tensor(1.))
    assert torch.allclose(summary.recall, torch.tensor(1.))


def test_binary_classification_summary_mixed():
    summary = metrics.BinaryClassificationSummary()

    ones = torch.ones(10, dtype=torch.int64)
    zeros = torch.zeros(10, dtype=torch.int64)

    summary.record_results(ones, zeros)
    summary.record_results(ones, ones)

    assert summary.count == 20
    assert torch.allclose(summary.accuracy, torch.tensor(0.5))
    assert torch.allclose(summary.recall, torch.tensor(0.5))
    assert torch.allclose(summary.precision, torch.tensor(1.))


def test_categorical_summary():
    generator = torch.manual_seed(42)
    num_classes = 10

    summary = metrics.ClassificationSummary(num_classes=num_classes)
    predicted_labels = torch.randint(0, num_classes-1, (20,), generator=generator)
    true_labels = torch.randint(0, num_classes-1, (20,), generator=generator)

    summary.record_results(true_labels, predicted_labels)

    assert summary.count == 20
