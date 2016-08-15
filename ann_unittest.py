import ann
import numpy as np
import unittest

SMALL=0.000001

class TestLogisticCost(unittest.TestCase):
  def test_single_positive(self):
    predicted = np.array([[0.3]])
    actual = np.array([[1.0]])
    gold = -np.log(0.3)

    cost = ann.logisticCost.compute(actual, predicted)
    self.assertAlmostEquals(gold, cost)

    cost1 = ann.logisticCost.compute(actual, predicted + SMALL)
    dcost_numeric = (cost1 - cost) / SMALL

    dcost = ann.logisticCost.dcostdy(actual, predicted)
    self.assertAlmostEquals(dcost_numeric, dcost, places=4)

  def test_single_negative(self):
    predicted = np.array([[0.3]])
    actual = np.array([[0.0]])
    gold = -np.log(1 - 0.3)

    cost = ann.logisticCost.compute(actual, predicted)
    self.assertAlmostEquals(gold, cost)

    cost1 = ann.logisticCost.compute(actual, predicted + SMALL)
    dcost_numeric = (cost1 - cost) / SMALL

    dcost = ann.logisticCost.dcostdy(actual, predicted)
    self.assertAlmostEquals(dcost_numeric, dcost, places=4)

  def test_multi(self):
    predicted = np.array([0.1, 0.3, 0.7, 0.8])
    actual = np.array([1.0, 0.0, 1.0, 0.0])
    
    gold = [
        -np.log(predicted[0]),
        -np.log(1-predicted[1]),
        -np.log(predicted[2]),
        -np.log(1-predicted[3]),
        ]

    predicted.shape = (4, 1)
    actual.shape = (4, 1)

    cost = ann.logisticCost.compute(actual, predicted)
    self.assertAlmostEquals(gold[0], cost[0])
    self.assertAlmostEquals(gold[1], cost[1])
    self.assertAlmostEquals(gold[2], cost[2])
    self.assertAlmostEquals(gold[3], cost[3])

    cost = sum(cost) / 4

    dcost = ann.logisticCost.dcostdy(actual, predicted)

    idx = 0
    predicted[idx][0] += SMALL
    cost1 = sum(ann.logisticCost.compute(actual, predicted)) / 4
    predicted[idx][0] -= SMALL
    dcost_numeric = (cost1 - cost) / SMALL
    self.assertAlmostEquals(dcost_numeric, dcost[idx][0], places=4)

    idx = 1
    predicted[idx][0] += SMALL
    cost1 = sum(ann.logisticCost.compute(actual, predicted)) / 4
    predicted[idx][0] -= SMALL
    dcost_numeric = (cost1 - cost) / SMALL
    self.assertAlmostEquals(dcost_numeric, dcost[idx][0], places=4)

    idx = 2
    predicted[idx][0] += SMALL
    cost1 = sum(ann.logisticCost.compute(actual, predicted)) / 4
    predicted[idx][0] -= SMALL
    dcost_numeric = (cost1 - cost) / SMALL
    self.assertAlmostEquals(dcost_numeric, dcost[idx][0], places=4)

    idx = 3
    predicted[idx][0] += SMALL
    cost1 = sum(ann.logisticCost.compute(actual, predicted)) / 4
    predicted[idx][0] -= SMALL
    dcost_numeric = (cost1 - cost) / SMALL
    self.assertAlmostEquals(dcost_numeric, dcost[idx][0], places=4)


class TestTraining(unittest.TestCase):
  def data1(self):
    np.random.seed(10)

    N = 100
    X = np.round(np.random.rand(N, 2) * 2 - 1, decimals=2)
    y = np.zeros((N, 1))

    for i, x in enumerate(X):
        if np.sum(x) > 0:
            y[i][0] = 1
            
    np.sum(y, axis=0)

    return (X, y)

  def data2(self):
    np.random.seed(10)
    N = 100

    X = np.round(np.random.rand(N, 4) * 2 - 1, decimals=2)
    y = np.zeros((N, 2))

    for i, x in enumerate(X):
      if x[0] + x[1] > 0:
        y[i][0] = 1

      if x[2] * x[3] > 0.5:
        y[i][1] = 1

    return (X, y)

  def test_train_1(self):
    X, y = self.data1()

    np.random.seed(10)
    nn = ann.NewNeuralNet(2, 1)
    original_cost = nn.compute_cost(X, y)
    ann.StochasticGradientDescent(nn, X, y, iterations=2000, learning_rate=0.3)
    new_cost = nn.compute_cost(X, y)
    self.assertLess(new_cost, original_cost/10)

  def test_train_2(self):
    X, y = self.data2()

    np.random.seed(10)
    nn = ann.NewNeuralNet(4, 6, 2)
    original_cost = nn.compute_cost(X, y)
    ann.StochasticGradientDescent(nn, X, y, iterations=6000, learning_rate=0.5)
    new_cost = nn.compute_cost(X, y)
    self.assertLess(new_cost, original_cost/10)
