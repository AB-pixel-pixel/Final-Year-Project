result = {
    "episode_results": {
        "0": {
            "finish": 7,
            "total": 10
        },
        "1": {
            "finish": 9,
            "total": 10
        },
        "2": {
            "finish": 8,
            "total": 10
        },
        "3": {
            "finish": 10,
            "total": 10
        },
        "4": {
            "finish": 7,
            "total": 10
        },
        "5": {
            "finish": 8,
            "total": 10
        },
        "6": {
            "finish": 0,
            "total": 10
        },
        "7": {
            "finish": 5,
            "total": 10
        },
        "8": {
            "finish": 5,
            "total": 10
        },
        "9": {
            "finish": 4,
            "total": 10
        },
        "10": {
            "finish": 8,
            "total": 10
        },
        "11": {
            "finish": 5,
            "total": 10
        },
        "12": {
            "finish": 7,
            "total": 10
        },
        "13": {
            "finish": 7,
            "total": 10
        },
        "14": {
            "finish": 6,
            "total": 10
        },
        "15": {
            "finish": 6,
            "total": 10
        },
        "16": {
            "finish": 2,
            "total": 10
        },
        "17": {
            "finish": 4,
            "total": 10
        },
        "18": {
            "finish": 1,
            "total": 10
        },
        "19": {
            "finish": 4,
            "total": 10
        },
        "20": {
            "finish": 3,
            "total": 10
        },
        "21": {
            "finish": 5,
            "total": 10
        },
        "22": {
            "finish": 4,
            "total": 10
        },
        "23": {
            "finish": 6,
            "total": 10
        }
    },
    "avg_finish": 0.5458333333333333
}

episode_results = result['episode_results']
# 计算平均名次
total_finish = sum(result["finish"] for result in episode_results.values())
print(total_finish)
total_episodes = len(episode_results)

avg_finish = total_finish / total_episodes
print("Average Finish:", avg_finish)

print(3.625+2.58)