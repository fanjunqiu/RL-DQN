# RL-DQN
Preliminary study DQN
张伟楠动手学强化学习第七章：DQN算法。
2022年5月4日
pip install pygame


cuda检测方法
# cuda
if  torch.cuda.is_available():
    print("choose to use gpu...")
    device = torch.device("cuda:0")
else:
    print("choose to use cpu...")
    device = torch.device("cpu")
