import matplotlib.pyplot as plt
from avalanche.logging.tensorboard_logger import TensorboardLogger
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import (
    Naive,
    Cumulative,
    JointTraining,
)


EPOCHS = 1
BATCH_SIZE = 100
LR = 0.001

scenario = SplitMNIST(n_experiences=5, fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
model = SimpleMLP(num_classes=scenario.n_classes)

tb_joint = TensorboardLogger("./tb_data/joint")
tb_cumulative = TensorboardLogger("./tb_data/cumulative")
tb_naive = TensorboardLogger("./tb_data/naive")

acc = accuracy_metrics(experience=True, stream=True)

joint_strategy = JointTraining(
    model=model,
    optimizer=Adam(model.parameters(), lr=LR),
    criterion=CrossEntropyLoss(),
    train_mb_size=BATCH_SIZE,
    train_epochs=EPOCHS,
    eval_mb_size=BATCH_SIZE,
    evaluator=EvaluationPlugin(acc, loggers=[tb_joint]),
)

naive_strategy = Naive(
    model=model,
    optimizer=Adam(model.parameters(), lr=LR),
    criterion=CrossEntropyLoss(),
    train_mb_size=BATCH_SIZE,
    train_epochs=EPOCHS,
    eval_mb_size=BATCH_SIZE,
    evaluator=EvaluationPlugin(acc, loggers=[tb_naive]),
)

cumulative_strategy = Cumulative(
    model=model,
    optimizer=Adam(model.parameters(), lr=LR),
    criterion=CrossEntropyLoss(),
    train_mb_size=BATCH_SIZE,
    train_epochs=EPOCHS,
    eval_mb_size=BATCH_SIZE,
    evaluator=EvaluationPlugin(acc, loggers=[tb_cumulative]),
)

key = "Top1_Acc_Stream/eval_phase/test_stream/Task000"

joint_strategy.train(scenario.train_stream)
joint_res = joint_strategy.eval(scenario.test_stream)
cum_res = []
naive_res = []

for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    naive_strategy.train(experience)
    nr = naive_strategy.eval(scenario.test_stream)
    naive_res.append(nr[key])

    cumulative_strategy.train(experience)
    cr = cumulative_strategy.eval(scenario.test_stream)
    cum_res.append(cr[key])


tasks = [1, 2, 3, 4, 5]
ticks = ["Exp00" + str(i) for i in range(5)]
plt.figure(figsize=(10, 6))
plt.xticks(tasks, ticks)
plt.plot(tasks, [joint_res[key]] * len(tasks), marker="o", label="Joint Training")
plt.plot(tasks, naive_res, marker="o", label="Naive Training")
plt.plot(tasks, cum_res, marker="o", label="Cumulative Training")

plt.xlabel("Experiences")
plt.ylabel("Average Accuracy")
plt.title("Joint, Naive, Cumulative Training Rusults for SplitMNIST")
plt.legend()

plt.show()
