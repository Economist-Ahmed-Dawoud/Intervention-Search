# base_all_apths_v1.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict


class BaseConfig:
    """A minimal base configuration class."""

    pass


@dataclass
class RCAResults:
    """
    Container for RCA results.

    Attributes:
        root_cause_nodes: A list of tuples, each (node, score).
        root_cause_paths: A dictionary mapping a node to its path (if available).
    """

    root_cause_nodes: list = field(default_factory=list)
    root_cause_paths: dict = field(default_factory=dict)
    path_explanations: dict = field(default_factory=dict)
    outcome_pct_change: float = None
    node_abnormal_summary: dict = field(default_factory=dict)

    def to_dict(self):
        # Directly use self.root_cause_paths which now contains the severity stars.
        # Filter it to include only paths for the final selected root causes.
        final_output_paths = {}
        # Get the names of the root causes that are actually in the final output list
        valid_root_cause_names = {rc["root_cause"] for rc in self.root_cause_nodes}

        for root, path_list in self.root_cause_paths.items():
            # Only include paths for roots that are in the final root cause list
            if root in valid_root_cause_names:
                # IMPORTANT: Assume path_list already contains dicts like:
                # {'path': [...], 'score': 0.95, 'path_severity': '★★★★★'}
                # No need to rebuild from path_explanations anymore.
                final_output_paths[root] = path_list

        # Return the final dictionary structure for the JSON
        return {
            "root_cause_nodes": self.root_cause_nodes,
            "root_cause_paths": final_output_paths,  # Use the correctly formatted paths directly
            "outcome_pct_change": self.outcome_pct_change,
            "node_abnormal_summary": self.node_abnormal_summary,
            # path_explanations is intentionally omitted from the final JSON
        }

    def to_list(self):
        result_list = []
        for node, score in self.root_cause_nodes:
            result_dict = {
                "root_cause": node,
                "score": score,
                "paths": self.root_cause_paths.get(node, None),
            }
            if hasattr(self, "path_explanations") and node in self.path_explanations:
                result_dict["explanations"] = self.path_explanations[node]
            result_list.append(result_dict)
        return result_list


class BaseRCA(ABC):
    """Abstract base class for RCA algorithms."""

    @abstractmethod
    def train(self, **kwargs):
        """Train the model with normal (non-anomalous) data."""
        pass

    @abstractmethod
    def find_root_causes(self, **kwargs) -> RCAResults:
        """Identify root causes from data."""
        pass
