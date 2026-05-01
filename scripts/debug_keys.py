import pickle
with open("results/main/CartPole-v1/seed_0.pkl", "rb") as f:
    log = pickle.load(f)["log"]
    print(log.keys())
