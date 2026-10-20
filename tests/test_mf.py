"""Check paired predictions, repeated-ID gradients, and batch training."""

import pytest
import torch

from memrecagent import mf


def fixed_model():
    model = mf.MatrixFactorization(3, 4, factors=2, seed=7)
    with torch.no_grad():
        model.user_vectors.weight.copy_(torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]]))
        model.item_vectors.weight.copy_(
            torch.tensor([[3.0, 1.0], [1.0, 4.0], [2.0, 3.0], [0.0, 2.0]])
        )
    return model


def test_forward_returns_one_prediction_per_pair_not_all_cross_combinations():
    model = fixed_model()
    result = model(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
    torch.testing.assert_close(result, torch.tensor([5.0, 6.0, 5.0]))
    assert result.shape == (3,)


@pytest.mark.parametrize(
    "users,items",
    [
        ([0, 1], [0]),
        ([[0], [1]], [[0], [1]]),
        ([], []),
    ],
)
def test_forward_rejects_shapes_that_could_silently_broadcast(users, items):
    model = fixed_model()
    with pytest.raises(ValueError):
        model(torch.tensor(users, dtype=torch.long), torch.tensor(items, dtype=torch.long))


@pytest.mark.parametrize("ratings", [[[4.0], [5.0]], [4.0]])
def test_loss_requires_one_rating_per_prediction(ratings):
    model = fixed_model()
    with pytest.raises(ValueError):
        model.rating_loss(torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor(ratings))


def test_repeated_user_id_sums_gradients_then_mse_averages_records():
    model = fixed_model()
    users, items = torch.tensor([0, 0]), torch.tensor([0, 1])
    # 预测分别为5、9；真实评分4、5，误差1、4。
    loss = model.rating_loss(users, items, torch.tensor([4.0, 5.0]), l2=0)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(8.5))
    # (2*1*[3,1] + 2*4*[1,4])/2 = [7,17]。
    torch.testing.assert_close(model.user_vectors.weight.grad[0], torch.tensor([7.0, 17.0]))
    torch.testing.assert_close(model.user_vectors.weight.grad[1:], torch.zeros(2, 2))
    torch.testing.assert_close(model.item_vectors.weight.grad[0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(model.item_vectors.weight.grad[1], torch.tensor([4.0, 8.0]))


def test_l2_is_mean_of_per_record_vector_squared_norms():
    model = fixed_model()
    users, items = torch.tensor([0, 0]), torch.tensor([0, 1])
    # 两条记录的向量平方和为15、22；均值18.5，λ=.1贡献1.85。
    loss = model.rating_loss(users, items, torch.tensor([5.0, 5.0]), l2=0.1)
    assert loss.item() == pytest.approx(9.85)


def test_initialization_is_reproducible_without_changing_global_rng():
    before = torch.random.get_rng_state().clone()
    first = mf.MatrixFactorization(3, 4, factors=2, seed=11)
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    second = mf.MatrixFactorization(3, 4, factors=2, seed=11)
    torch.testing.assert_close(first.user_vectors.weight, second.user_vectors.weight)
    torch.testing.assert_close(first.item_vectors.weight, second.item_vectors.weight)


def test_training_keeps_last_partial_batch_and_lowers_training_mse():
    model = mf.MatrixFactorization(2, 3, factors=2, seed=42)
    train = [(0, 0, 5.0), (0, 1, 2.0), (1, 0, 4.0), (1, 1, 1.0), (1, 2, 3.0)]
    rows = mf.fit_mf(model, train, [(0, 2, 3.0)], epochs=40, batch_size=2, lr=0.05)
    assert len(rows) == 41
    assert rows[0].epoch == 0
    assert all(row.batch_sizes == (2, 2, 1) for row in rows[1:])
    assert rows[-1].train_mse < rows[0].train_mse * 0.2


def test_validation_ratings_do_not_affect_parameter_updates():
    first = mf.MatrixFactorization(2, 3, factors=2, seed=42)
    second = mf.MatrixFactorization(2, 3, factors=2, seed=42)
    train = [(0, 0, 5.0), (1, 1, 1.0)]
    a = mf.fit_mf(first, train, [(0, 2, 5.0)], epochs=3, batch_size=1)
    b = mf.fit_mf(second, train, [(0, 2, 1.0)], epochs=3, batch_size=1)
    torch.testing.assert_close(first.user_vectors.weight, second.user_vectors.weight)
    torch.testing.assert_close(first.item_vectors.weight, second.item_vectors.weight)
    assert a[-1].validation_mse != b[-1].validation_mse
    # 未参与任何训练记录的物品2也不能被验证集更新或正则项全表收缩。
    initial = mf.MatrixFactorization(2, 3, factors=2, seed=42)
    torch.testing.assert_close(first.item_vectors.weight[2], initial.item_vectors.weight[2])


def test_shuffle_and_training_are_reproducible():
    train = [(0, 0, 5.0), (0, 1, 2.0), (1, 0, 4.0), (1, 1, 1.0)]
    first = mf.MatrixFactorization(2, 3, factors=2, seed=42)
    second = mf.MatrixFactorization(2, 3, factors=2, seed=42)
    a = mf.fit_mf(first, train, [(0, 2, 3.0)], epochs=3, batch_size=3, seed=42)
    b = mf.fit_mf(second, train, [(0, 2, 3.0)], epochs=3, batch_size=3, seed=42)
    assert a == b
    torch.testing.assert_close(first.user_vectors.weight, second.user_vectors.weight)


@pytest.mark.parametrize(
    "train,validation",
    [
        ([], [(0, 1, 3.0)]),
        ([(0, 0, 5.0)], []),
        ([(0, 0, 5.0)], [(0, 0, 4.0)]),
        ([(0, 0, 5.0), (0, 0, 5.0)], [(0, 1, 3.0)]),
        ([(2, 0, 5.0)], [(0, 1, 3.0)]),
        ([(0, 0, 0.0)], [(0, 1, 3.0)]),
    ],
)
def test_invalid_or_overlapping_records_are_rejected(train, validation):
    model = mf.MatrixFactorization(2, 3, factors=2, seed=42)
    with pytest.raises(ValueError):
        mf.fit_mf(model, train, validation, epochs=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"lr": 0},
        {"l2": -1},
    ],
)
def test_invalid_training_configuration_is_rejected(kwargs):
    model = mf.MatrixFactorization(2, 3, factors=2, seed=42)
    with pytest.raises(ValueError):
        mf.fit_mf(model, [(0, 0, 5.0)], [(0, 1, 3.0)], **kwargs)
