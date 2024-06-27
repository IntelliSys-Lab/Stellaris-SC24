import utils
import config
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_line(
    scheduler_names,
    env_names,
    algo_names,
):
    for env_name in env_names:
        for algo_name in algo_names:
            for scheduler_name in scheduler_names:
                file_name = f"{scheduler_name}~{env_name}~{algo_name}~timeline"
                df = pd.read_csv(f"{config.log_path}/{file_name}.csv")
                fig = sns.lineplot(
                    data=df,
                    x="Wall clock time (s)",
                    y="Episodic reward",
                    label=scheduler_name,
                ).get_figure()

            fig.savefig(f"{config.figure_path}/timeline~{env_name}~{algo_name}.png")
            fig.clear()

def plot_bar(
    scheduler_names,
    env_names,
    algo_names,
):
    csv_cost = [
        [
            "Framework",
            "Cost ($)",
            "Environment",
        ]
    ]

    for scheduler_name in scheduler_names:
        for env_name in env_names:
            for algo_name in algo_names:
                file_name = f"{scheduler_name}~{env_name}~{algo_name}~info"
                df = pd.read_csv(f"{config.log_path}/{file_name}.csv")
                csv_cost.append([scheduler_name, df["cost"].iloc[-1], env_name])

    fig = sns.barplot(
        data=pd.DataFrame(csv_cost[1:], columns=csv_cost[0]),
        x="Framework",
        y="Cost ($)",
        hue="Environment",
    ).get_figure()

    fig.savefig(f"{config.figure_path}/cost.png")
    fig.clear()



if __name__ == "__main__":
    utils.mkdir(config.figure_path)

    scheduler_names = ["stellaris", "rllib"]
    env_names = list(config.envs.keys())
    algo_names = list(config.algos.keys())

    # Plot timelines
    plot_line(
        scheduler_names=scheduler_names,
        env_names=env_names,
        algo_names=algo_names,
    )

    # Plot cost bars
    plot_bar(
        scheduler_names=scheduler_names,
        env_names=env_names,
        algo_names=algo_names,
    )
