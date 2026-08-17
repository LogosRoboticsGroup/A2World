import numpy as np


def sample_history_indices(actions: np.ndarray, budget: int) -> list[int]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected [T, 14] history actions, got {actions.shape}")
    if budget < 1:
        raise ValueError("History budget must be positive")
    if len(actions) == 0:
        return [0] * budget

    left = actions[:, :6]
    right = actions[:, 7:13]
    translation = np.square(left[:, :3]).sum(1) + np.square(right[:, :3]).sum(1)
    rotation = np.square(left[:, 3:]).sum(1) + np.square(right[:, 3:]).sum(1)
    step_length = np.sqrt(translation + 0.25 * rotation)
    arc = np.concatenate([[0.0], np.cumsum(step_length)])
    last = len(arc) - 1
    if budget == 1:
        return [last]
    if arc[-1] <= 1e-8:
        return [0] * (budget - 1) + [last]

    targets = np.linspace(0.0, arc[-1], budget)
    indices = [int(np.abs(arc - target).argmin()) for target in targets]
    indices[0] = 0
    indices[-1] = last
    return indices


def action_path_features(history_actions: np.ndarray, future_actions: np.ndarray, indices: list[int]) -> np.ndarray:
    history_actions = np.asarray(history_actions, dtype=np.float64)
    future_actions = np.asarray(future_actions, dtype=np.float64)
    edges = np.concatenate([history_actions, future_actions], axis=0)
    node_gripper = np.concatenate(
        [[history_actions[0, 6] if len(history_actions) else future_actions[0, 6]], edges[:, 6]]
    )
    prefix = np.concatenate([np.zeros((1, 6)), np.cumsum(edges[:, :6], axis=0)], axis=0)
    prefix_abs = np.concatenate([np.zeros((1, 6)), np.cumsum(np.abs(edges[:, :6]), axis=0)], axis=0)
    prefix_sq = np.concatenate([np.zeros((1, 6)), np.cumsum(np.square(edges[:, :6]), axis=0)], axis=0)
    toggles = np.concatenate([[0.0], np.cumsum(node_gripper[1:] != node_gripper[:-1])])
    history_nodes = len(history_actions) + 1
    features = np.zeros((len(future_actions), len(indices), 22), dtype=np.float32)

    for future_index in range(len(future_actions)):
        target = history_nodes + future_index
        for slot, history_index in enumerate(indices):
            start = min(history_index, history_nodes - 1)
            length = max(target - start, 1)
            mean = (prefix[target] - prefix[start]) / length
            mean[:3] = np.tanh(mean[:3] / 0.05)
            mean[3:] = np.tanh(mean[3:] / 0.25)
            mean_abs = (prefix_abs[target] - prefix_abs[start]) / length
            rms = np.sqrt(np.maximum((prefix_sq[target] - prefix_sq[start]) / length, 0.0))
            length_ratio = length / max(len(edges), 1)
            toggle_ratio = (toggles[target] - toggles[start]) / max(toggles[-1], 1.0)
            features[future_index, slot] = np.concatenate(
                [
                    mean,
                    mean_abs,
                    rms,
                    [length_ratio, node_gripper[start], node_gripper[target], toggle_ratio],
                ]
            )
    return features
